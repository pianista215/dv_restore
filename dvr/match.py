"""Emparejar frames entre varias capturas de la misma cinta.

Dos lecturas correctas del mismo frame de cinta dan bloques de video byte a
byte identicos, asi que se pueden emparejar sin timecode. El procedimiento es
el que ya funcionaba en dv_autofix.py, ampliado a N capturas:

  - huella FNV de cada bloque sano, saltando los 3 bytes de ID DIF, que
    difieren entre capturas aunque el contenido sea el mismo;
  - las huellas votan candidatos, y cada candidato se verifica exigiendo que
    coincidan TODOS los bloques de video sanos comunes;
  - ademas se exige que coincidan N bloques de audio, porque en una escena
    quieta dos frames distintos comparten bloques de video identicos y sin el
    audio se emparejan mal.

Con eso se construye un grafo de identidad; cada componente conexa es un frame
de cinta, con de 1 a N lecturas. Ordenarlas da a la vez la fusion y la
reconstruccion de la linea temporal.
"""

import numpy as np
from collections import defaultdict, Counter

PRIME = np.uint64(1099511628211)
SEED = np.uint64(1469598103934665603)
MAX_PER_KEY = 400
CHUNK = 2000


def fingerprints(data, probe, chunk=CHUNK):
    """Huella de cada bloque sonda y si esta sano, para todos los frames."""
    n = data.shape[0]
    idx = (probe[:, None] + np.arange(8, 80)[None, :]).ravel()
    hs = np.empty((n, len(probe)), dtype=np.uint64)
    ok = np.empty((n, len(probe)), dtype=bool)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        raw = np.ascontiguousarray(data[a:b][:, idx]).reshape(b - a, len(probe), 72)
        u = raw.view(np.uint64).reshape(b - a, len(probe), 9)
        with np.errstate(over="ignore"):
            h = np.full((b - a, len(probe)), SEED, dtype=np.uint64)
            for k in range(u.shape[2]):
                h = (h ^ u[:, :, k]) * PRIME
        hs[a:b] = h
        ok[a:b] = (data[a:b][:, probe + 3] >> 4) == 0
    return hs, ok


class Matcher:
    def __init__(self, caps, probes=162, min_audio=20, candidates=40):
        self.caps = caps
        self.prof = caps[0].prof
        self.min_audio = min_audio
        self.candidates = candidates
        # Las sondas se eligen con un sorteo fijo, NO a paso constante.
        # El dano de estas cintas tiene periodo 5 (dentro de cada segmento
        # sobreviven siempre los mismos macrobloques y mueren los mismos), asi
        # que un paso que sea multiplo de 5 hace que todas las sondas caigan en
        # una posicion que esta siempre rota y no se obtenga ni un voto aunque
        # los dos frames compartan cientos de bloques identicos.
        n = self.prof.n_video
        rng = np.random.default_rng(20240913)
        sel = np.sort(rng.choice(n, min(probes, n), replace=False))
        self.probe = self.prof.video[sel]
        self.hs, self.ok = [], []
        for c in caps:
            h, o = fingerprints(c.data, self.probe)
            self.hs.append(h)
            self.ok.append(o)
        self.index = defaultdict(list)
        for ci, h in enumerate(self.hs):
            o = self.ok[ci]
            for f in range(caps[ci].n):
                for p in np.nonzero(o[f])[0]:
                    self.index[int(h[f, p])].append((ci, f))
        for k in list(self.index):
            if len(self.index[k]) > MAX_PER_KEY:
                del self.index[k]

    def votes(self, ci, f, other=None):
        v = Counter()
        for p in np.nonzero(self.ok[ci][f])[0]:
            for (cj, g) in self.index.get(int(self.hs[ci][f, p]), ()):
                if cj == ci:
                    continue
                if other is not None and cj != other:
                    continue
                v[(cj, g)] += 1
        return v

    def same_tape_frame(self, ci, f, cj, g):
        """Verificacion estricta: todo bloque de video sano comun debe coincidir
        y el audio tiene que respaldarlo."""
        prof = self.prof
        fa = self.caps[ci].data[f]
        fb = self.caps[cj].data[g]
        both = ((fa[prof.sta] >> 4) == 0) & ((fb[prof.sta] >> 4) == 0)
        n = int(both.sum())
        if n < 6:
            return False, 0, 0
        idx = prof.vdata[both]
        if not bool((fa[idx] == fb[idx]).all()):
            return False, n, 0
        aud = int((fa[prof.adata] == fb[prof.adata]).all(axis=1).sum())
        return aud >= self.min_audio, n, aud


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return ra


class TapeIndex:
    """Frames de cinta ordenados, cada uno con sus lecturas."""

    def __init__(self, caps, clusters, order, stats):
        self.caps = caps
        self.clusters = clusters          # lista de [(ci, f), ...]
        self.order = order                # indices de clusters, en orden
        self.stats = stats

    def __len__(self):
        return len(self.order)

    def reads(self, k):
        return self.clusters[self.order[k]]


def build_index(caps, probes=162, min_audio=20, candidates=40, verbose=True,
                refine=True, window=40, audio_only=80):
    m = Matcher(caps, probes, min_audio, candidates)
    base = np.cumsum([0] + [c.n for c in caps])
    total = int(base[-1])
    uf = Union(total)
    gid = lambda ci, f: int(base[ci]) + f

    links = conflicts = 0
    for ci, cap in enumerate(caps):
        for f in range(cap.n):
            v = m.votes(ci, f)
            if not v:
                continue
            done = set()
            for (cj, g), _ in v.most_common(candidates):
                if cj in done:
                    continue
                ok, _, _ = m.same_tape_frame(ci, f, cj, g)
                if ok:
                    if uf.find(gid(ci, f)) != uf.find(gid(cj, g)):
                        links += 1
                    uf.union(gid(ci, f), gid(cj, g))
                    done.add(cj)
        if verbose:
            print(f"  {cap.name}: {cap.n} frames emparejados")

    groups = defaultdict(list)
    for ci, cap in enumerate(caps):
        for f in range(cap.n):
            groups[uf.find(gid(ci, f))].append((ci, f))
    clusters = list(groups.values())
    cid = {}
    for k, g in enumerate(clusters):
        for r in g:
            cid[r] = k
        seen = Counter(c for c, _ in g)
        if any(v > 1 for v in seen.values()):
            conflicts += 1

    order = _topo_order(clusters, cid, caps)
    extra = 0
    if refine and len(caps) > 1:
        extra = _refine(m, caps, uf, gid, clusters, cid, order, window,
                        audio_only, verbose)
        if extra:
            groups = defaultdict(list)
            for ci, cap in enumerate(caps):
                for f in range(cap.n):
                    groups[uf.find(gid(ci, f))].append((ci, f))
            clusters = list(groups.values())
            cid = {}
            for k, g in enumerate(clusters):
                for r in g:
                    cid[r] = k
            order = _topo_order(clusters, cid, caps)
    stats = dict(clusters=len(clusters), links=links + extra, conflicts=conflicts,
                 total_reads=total, refined=extra,
                 shared=sum(1 for g in clusters if len(g) > 1))
    return TapeIndex(caps, clusters, order, stats)


def _refine(m, caps, uf, gid, clusters, cid, order, window, audio_only, verbose):
    """Segunda pasada: emparejar lo que quedo suelto usando el orden.

    Un frame que no encontro pareja por huella casi siempre esta tan danado que
    apenas le quedan bloques sanos. Pero su sitio en la cinta ya esta acotado
    por los vecinos que si casaron, asi que basta con probar a fuerza bruta la
    ventana correspondiente de la otra captura. La verificacion sigue siendo la
    estricta; solo cuando no quedan bloques de video sanos en comun se acepta
    el emparejamiento por audio, que no lleva marca de error.
    """
    ncap = len(caps)
    npos = len(order)
    seq = np.full((ncap, npos), -1, dtype=np.int64)
    for k, c in enumerate(order):
        for (ci, f) in clusters[c]:
            seq[ci, k] = f
    added = 0
    by_audio = 0
    prof = m.prof
    for k in range(npos):
        present = [ci for ci in range(ncap) if seq[ci, k] >= 0]
        if len(present) == ncap:
            continue
        for cj in range(ncap):
            if seq[cj, k] >= 0:
                continue
            prev = nxt = None
            for t in range(k - 1, max(-1, k - 400), -1):
                if seq[cj, t] >= 0:
                    prev = int(seq[cj, t])
                    break
            for t in range(k + 1, min(npos, k + 400)):
                if seq[cj, t] >= 0:
                    nxt = int(seq[cj, t])
                    break
            lo = 0 if prev is None else prev + 1
            hi = caps[cj].n - 1 if nxt is None else nxt - 1
            if hi < lo or hi - lo > window:
                continue
            ci = present[0]
            f = int(seq[ci, k])
            hit = None
            for g in range(lo, hi + 1):
                if uf.find(gid(cj, g)) == uf.find(gid(ci, f)):
                    continue
                ok, n, aud = m.same_tape_frame(ci, f, cj, g)
                if ok:
                    hit = (g, False)
                    break
                if n < 6 and aud >= audio_only:
                    hit = (g, True)
            if hit is None:
                continue
            g, only_aud = hit
            uf.union(gid(ci, f), gid(cj, g))
            seq[cj, k] = g
            added += 1
            by_audio += only_aud
    if verbose and added:
        print(f"  segunda pasada: {added} parejas mas "
              f"({by_audio} solo por audio)")
    return added


def _topo_order(clusters, cid, caps):
    """Orden de los frames de cinta a partir del orden interno de cada captura.

    Cada captura impone que su frame i va antes que el i+1. Se ordena el grafo
    resultante; si aparece un ciclo (senal de un emparejamiento falso) se rompe
    por la arista menos sostenida y se sigue.
    """
    n = len(clusters)
    succ = defaultdict(Counter)
    indeg = Counter()
    for ci, cap in enumerate(caps):
        for f in range(cap.n - 1):
            a, b = cid[(ci, f)], cid[(ci, f + 1)]
            if a != b:
                succ[a][b] += 1
    for a, d in succ.items():
        for b in d:
            indeg[b] += 1

    # desempate: posicion media de la lectura mas temprana
    key = {}
    for k, g in enumerate(clusters):
        key[k] = min(f for _, f in g)

    import heapq
    ready = [(key[k], k) for k in range(n) if indeg[k] == 0]
    heapq.heapify(ready)
    out = []
    seen = set()
    while ready:
        _, a = heapq.heappop(ready)
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
        for b in succ.get(a, ()):
            indeg[b] -= 1
            if indeg[b] <= 0 and b not in seen:
                heapq.heappush(ready, (key[b], b))
    if len(out) < n:
        # quedan ciclos: se anaden por orden de posicion, que es lo mas
        # parecido al orden real de cinta
        rest = sorted((k for k in range(n) if k not in seen), key=lambda k: key[k])
        out += rest
    return out


def save_index(idx, path):
    import json
    data = dict(
        captures=[c.path for c in idx.caps],
        order=[int(k) for k in idx.order],
        clusters=[[[int(c), int(f)] for c, f in g] for g in idx.clusters],
        stats={k: int(v) for k, v in idx.stats.items()},
    )
    with open(path, "w") as fh:
        json.dump(data, fh)


def load_index(path, caps=None):
    import json
    from .dvfile import Capture
    with open(path) as fh:
        d = json.load(fh)
    caps = caps or [Capture(p) for p in d["captures"]]
    clusters = [[(int(c), int(f)) for c, f in g] for g in d["clusters"]]
    return TapeIndex(caps, clusters, [int(k) for k in d["order"]], d["stats"])
