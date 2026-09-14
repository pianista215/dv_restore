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

import os

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


def verify_pair(prof, fa, fb, min_audio=20, min_blocks=20, min_ratio=0.90):
    """Son la misma lectura de cinta?

    Se exige que coincida byte a byte la GRAN MAYORIA de los bloques de video
    sanos que ambos comparten, no todos. Exigirlos todos parecia lo riguroso,
    pero rechazaba emparejamientos buenos: la camara marca como sanos algunos
    bloques que no lo son, asi que dos lecturas del mismo frame de cinta
    discrepan en un puñado. Medido en este material, el mismo frame da una
    proporcion de 1,000 (percentil 1 tambien 1,000) y dos frames de cinta
    distintos dan 0,000, asi que el umbral es holgado.

    Rechazar esos emparejamientos era caro: las lecturas no enlazadas quedaban
    como cadena paralela y se intercalaban una a una con las buenas, que es lo
    que se ve como microrepeticiones y parpadeos.

    Ademas hay que tener respaldo del audio, porque en una escena quieta dos
    frames distintos comparten bloques de video identicos.

    Devuelve (es_la_misma, bloques_sanos_comunes, bloques_de_audio_iguales).
    """
    both = ((fa[prof.sta] >> 4) == 0) & ((fb[prof.sta] >> 4) == 0)
    n = int(both.sum())
    if n < min_blocks:
        return False, n, 0
    idx = prof.vdata[both]
    same = int((fa[idx] == fb[idx]).all(axis=1).sum())
    if same < min_ratio * n:
        return False, n, 0
    aud = int((fa[prof.adata] == fb[prof.adata]).all(axis=1).sum())
    return aud >= min_audio, n, aud


def probe_set(prof, probes, seed=20240913):
    """Que bloques se usan como huella.

    Por sorteo fijo, NO a paso constante: el dano de estas cintas tiene periodo
    5 (dentro de cada segmento sobreviven siempre los mismos macrobloques y
    mueren los mismos), asi que un paso multiplo de 5 hace que todas las sondas
    caigan en una posicion siempre rota y no se obtenga ni un voto aunque los
    dos frames compartan cientos de bloques identicos.
    """
    n = prof.n_video
    rng = np.random.default_rng(seed)
    return prof.video[np.sort(rng.choice(n, min(probes, n), replace=False))]


class Matcher:
    def __init__(self, caps, probes=162, min_audio=20, candidates=40,
                 min_blocks=20, min_ratio=0.90):
        self.caps = caps
        self.prof = caps[0].prof
        self.min_audio = min_audio
        self.min_blocks = min_blocks
        self.min_ratio = min_ratio
        self.candidates = candidates
        self.probe = probe_set(self.prof, probes)
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
        return verify_pair(self.prof, self.caps[ci].data[f],
                           self.caps[cj].data[g], self.min_audio,
                           self.min_blocks, self.min_ratio)


class Union:
    """Union-find que se niega a juntar dos lecturas de la misma captura.

    Una captura no puede leer dos veces el mismo frame de cinta. Si un
    emparejamiento lo pretende, es falso, y ademas es venenoso: fusionar los
    frames f y f+5 de una captura convierte la cadena que los une en un CICLO,
    y con el orden topologico un ciclo temprano bloquea todo lo que viene
    detras. Medido en el arranque de esta cinta: dos fusiones asi dejaban el
    84% de los grupos sin ordenar por topologia.
    """

    def __init__(self, n, cap_of=None):
        self.p = list(range(n))
        self.caps = [{cap_of(i)} if cap_of else set() for i in range(n)]
        self.refused = 0

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.caps[ra] & self.caps[rb]:
            self.refused += 1
            return None
        self.p[rb] = ra
        self.caps[ra] |= self.caps[rb]
        self.caps[rb] = set()
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
                refine=True, window=40, audio_only=80, max_tries=4_000_000,
                drop_empty=0.95):
    m = Matcher(caps, probes, min_audio, candidates)
    base = np.cumsum([0] + [c.n for c in caps])
    total = int(base[-1])
    owner = np.zeros(total, dtype=np.int32)
    for ci, cap in enumerate(caps):
        owner[int(base[ci]):int(base[ci]) + cap.n] = ci
    uf = Union(total, cap_of=lambda i: int(owner[i]))
    gid = lambda ci, f: int(base[ci]) + f

    links = conflicts = 0
    import time
    t0 = time.time()
    for ci, cap in enumerate(caps):
        for f in range(cap.n):
            if verbose and f and f % 2000 == 0:
                print(f"    {cap.name}: {f}/{cap.n}  ({time.time()-t0:.0f}s)")
            v = m.votes(ci, f)
            if not v:
                continue
            done = set()
            for (cj, g), _ in v.most_common(candidates):
                if cj in done:
                    continue
                ok, _, _ = m.same_tape_frame(ci, f, cj, g)
                if ok:
                    same = uf.find(gid(ci, f)) == uf.find(gid(cj, g))
                    if same or uf.union(gid(ci, f), gid(cj, g)) is not None:
                        if not same:
                            links += 1
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
    dropped_empty = 0
    extra = 0
    if refine and len(caps) > 1:
        extra = _refine(m, caps, uf, gid, clusters, cid, order, window,
                        audio_only, verbose, max_tries)
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
    if drop_empty is not None:
        # Un frame leido por una sola captura y ocultado casi entero por ella no
        # lleva NI UN dato de cinta: es relleno que solto el deck al patinar.
        # No es un frame de cinta distinto, y como no tiene datos no puede
        # verificar contra nada, asi que nunca enlaza y el orden lo coloca donde
        # quiere, intercalado con el material bueno. Fuera.
        lim = drop_empty * caps[0].prof.n_video
        keep = []
        for k in order:
            g = clusters[k]
            if len(g) == 1:
                ci, f = g[0]
                if caps[ci].n_bad(f) >= lim:
                    dropped_empty += 1
                    continue
            keep.append(k)
        order = keep
    stats = dict(clusters=len(clusters), links=links + extra, conflicts=conflicts,
                 refused=uf.refused,
                 total_reads=total, refined=extra, dropped_empty=dropped_empty,
                 emitted=len(order),
                 shared=sum(1 for g in clusters if len(g) > 1))
    return TapeIndex(caps, clusters, order, stats)


def _refine(m, caps, uf, gid, clusters, cid, order, window, audio_only,
            verbose, max_tries=4_000_000):
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
    tried = 0
    # Frame siguiente de cada captura para cada posicion, calculado de una vez.
    # Barrerlo a mano por cada posicion y captura eran decenas de millones de
    # iteraciones con muchas capturas.
    nexts = np.full((ncap, npos), -1, dtype=np.int64)
    for cj in range(ncap):
        last = -1
        for k in range(npos - 1, -1, -1):
            nexts[cj, k] = last
            if seq[cj, k] >= 0:
                last = int(seq[cj, k])
    # el anterior se lleva al vuelo, asi recoge los emparejamientos que vamos
    # anadiendo en esta misma pasada
    prevs = [-1] * ncap
    for k in range(npos):
        present = [ci for ci in range(ncap) if seq[ci, k] >= 0]
        if len(present) == ncap:
            for cj in range(ncap):
                prevs[cj] = int(seq[cj, k])
            continue
        # se parte de la lectura menos danada: es la que mas posibilidades
        # tiene de verificar contra otra captura
        ci = min(present, key=lambda c: caps[c].n_bad(int(seq[c, k])))
        f = int(seq[ci, k])
        for cj in range(ncap):
            if seq[cj, k] >= 0:
                continue
            prev, nxt = prevs[cj], int(nexts[cj, k])
            lo = 0 if prev < 0 else prev + 1
            hi = caps[cj].n - 1 if nxt < 0 else nxt - 1
            if hi < lo or hi - lo > window:
                continue
            if max_tries and tried >= max_tries:
                continue
            tried += hi - lo + 1
            hit = None
            for g in range(lo, hi + 1):
                if uf.find(gid(cj, g)) == uf.find(gid(ci, f)):
                    continue
                ok, n, aud = m.same_tape_frame(ci, f, cj, g)
                if ok:
                    hit = (g, False)
                    break
                if n < m.min_blocks and aud >= audio_only:
                    hit = (g, True)
            if hit is None:
                continue
            g, only_aud = hit
            if uf.union(gid(ci, f), gid(cj, g)) is None:
                continue
            seq[cj, k] = g
            added += 1
            by_audio += only_aud
        for cj in range(ncap):
            if seq[cj, k] >= 0:
                prevs[cj] = int(seq[cj, k])
    if verbose:
        print(f"  segunda pasada: {added} parejas mas "
              f"({by_audio} solo por audio, {tried} comprobaciones)")
    return added


def estimate_position(clusters, cid, caps, spine=0, rounds=4):
    """Posicion estimada de cada frame de cinta, en la escala de la captura
    columna vertebral.

    Hace falta para desempatar el orden. Cuando dos cadenas de frames no estan
    enlazadas entre si, el orden topologico tiene libertad para intercalarlas, y
    si se desempata por el numero de frame CRUDO se intercalan momentos
    distintos de la cinta: cada captura arranca en un punto diferente, asi que
    ese numero no significa nada entre capturas.

    Aqui se ancla cada grupo que contenga una lectura de la columna vertebral a
    su numero de frame, y se propaga al resto interpolando dentro de cada
    captura, que si va en orden. Con eso el desempate es una posicion de cinta
    de verdad, comparable entre capturas.
    """
    n = len(clusters)
    pos = np.full(n, np.nan)
    for k, g in enumerate(clusters):
        for (ci, f) in g:
            if ci == spine:
                pos[k] = float(f)
    if not np.isfinite(pos).any():
        for k, g in enumerate(clusters):
            pos[k] = float(min(f for _, f in g))
        return pos
    for _ in range(rounds):
        changed = False
        for ci, cap in enumerate(caps):
            ks = np.array([cid[(ci, f)] for f in range(cap.n)])
            known = np.isfinite(pos[ks])
            if known.sum() < 2:
                continue
            fs = np.nonzero(known)[0].astype(float)
            ps = pos[ks[known]]
            # pendiente global de esta captura frente a la columna vertebral,
            # para poder estimar tambien fuera del tramo con anclas
            slope = (ps[-1] - ps[0]) / max(fs[-1] - fs[0], 1.0)
            miss = np.nonzero(~known)[0]
            if not len(miss):
                continue
            est = np.interp(miss.astype(float), fs, ps)
            left = miss < fs[0]
            right = miss > fs[-1]
            est[left] = ps[0] - (fs[0] - miss[left]) * slope
            est[right] = ps[-1] + (miss[right] - fs[-1]) * slope
            for m, v in zip(miss, est):
                k = ks[m]
                if not np.isfinite(pos[k]):
                    pos[k] = v
                    changed = True
        if not changed:
            break
    bad = ~np.isfinite(pos)
    if bad.any():
        pos[bad] = np.nanmax(pos[~bad]) + 1.0
    return pos


def _topo_order(clusters, cid, caps, spine=0):
    """Orden de los frames de cinta a partir del orden interno de cada captura.

    Cada captura impone que su frame i va antes que el i+1. Se ordena el grafo
    resultante respetando esas aristas, y cuando hay varios grupos disponibles a
    la vez se elige el de menor posicion estimada de cinta. Si queda algun ciclo
    (senal de un emparejamiento falso) se anaden por esa misma posicion.
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

    key = estimate_position(clusters, cid, caps, spine)

    import heapq
    ready = [(float(key[k]), k) for k in range(n) if indeg[k] == 0]
    heapq.heapify(ready)
    out = []
    seen = set()
    forced = 0
    pending = set(range(n))
    while len(out) < n:
        if not ready:
            # Queda un ciclo. Se fuerza SOLO el nodo mas prometedor (el de
            # menor posicion estimada entre los que menos les falta) y se
            # sigue: tirar de golpe todo lo que queda detras, como se hacia
            # antes, dejaba el 84% del arranque ordenado solo por posicion.
            if not pending:
                break
            a = min(pending, key=lambda k: (indeg[k], float(key[k])))
            heapq.heappush(ready, (float(key[a]), a))
            forced += 1
        _, a = heapq.heappop(ready)
        if a in seen:
            continue
        seen.add(a)
        pending.discard(a)
        out.append(a)
        for b in succ.get(a, ()):
            indeg[b] -= 1
            if indeg[b] <= 0 and b not in seen:
                heapq.heappush(ready, (float(key[b]), b))
    _topo_order.forced = forced
    return out



def triage(base, donor_paths, probes=162, min_audio=20, candidates=40,
           verbose=True):
    """Que donantes solapan de verdad con la captura base, y en que tramo.

    Con muchas capturas no se puede meter todo en un indice y verificar todas
    contra todas: se dispara la memoria y el tiempo. Aqui se indexa SOLO la
    base y se pasa cada donante por delante en streaming, asi que el coste es
    lineal en el material del donante y la memoria queda acotada.

    Sirve para quedarse solo con el material util antes del emparejado N-way
    de verdad. Devuelve una lista de dict, uno por donante.
    """
    from .dvfile import Capture
    import time

    prof = base.prof
    probe = probe_set(prof, probes)
    if verbose:
        print(f"indexando la base {base.name} ({base.n} frames)...")
    hb, okb = fingerprints(base.data, probe)
    index = defaultdict(list)
    for f in range(base.n):
        for p in np.nonzero(okb[f])[0]:
            index[int(hb[f, p])].append(f)
    for k in list(index):
        if len(index[k]) > MAX_PER_KEY:
            del index[k]
    if verbose:
        print(f"  {len(index)} huellas distintas\n")

    out = []
    for path in donor_paths:
        t0 = time.time()
        try:
            cap = Capture(path)
        except ValueError as e:
            if verbose:
                print(f"{os.path.basename(path)}: {e}")
            continue
        if cap.prof is not prof:
            if verbose:
                print(f"{cap.name}: perfil {cap.prof.name}, no encaja")
            continue
        hd, okd = fingerprints(cap.data, probe)
        matched = 0
        bmin = bmax = dmin = dmax = None
        for g in range(cap.n):
            v = Counter()
            for p in np.nonzero(okd[g])[0]:
                for f in index.get(int(hd[g, p]), ()):
                    v[f] += 1
            if not v:
                continue
            fb = cap.data[g]
            for f, _ in v.most_common(candidates):
                ok, _, _ = verify_pair(prof, base.data[f], fb, min_audio)
                if not ok:
                    continue
                matched += 1
                bmin = f if bmin is None else min(bmin, f)
                bmax = f if bmax is None else max(bmax, f)
                dmin = g if dmin is None else min(dmin, g)
                dmax = g if dmax is None else max(dmax, g)
                break
        rec = dict(path=path, name=cap.name, frames=cap.n, matched=matched,
                   base_lo=bmin, base_hi=bmax, don_lo=dmin, don_hi=dmax,
                   seconds=time.time() - t0)
        out.append(rec)
        del hd, okd
        if verbose:
            if matched:
                print(f"{cap.name:26s} {cap.n:6d} frames -> {matched:5d} casan "
                      f"({100*matched/cap.n:5.1f}%)  base {bmin}..{bmax}  "
                      f"donante {dmin}..{dmax}   [{rec['seconds']:.0f}s]")
            else:
                print(f"{cap.name:26s} {cap.n:6d} frames -> sin solape con la base"
                      f"   [{rec['seconds']:.0f}s]")
    return out


def window_clips(base, donor_paths, lo, hi, out_dir, tag, margin=30,
                 probes=162, min_audio=20, candidates=40, min_matched=15,
                 gap=120, verbose=True):
    """Recorta un tramo de la base y el tramo equivalente de cada donante.

    Igual que triage, indexa SOLO la base (y solo el tramo pedido), asi que el
    coste no depende de cuantos donantes haya ni de lo grandes que sean. Cada
    donante se recorre una vez y se queda con el intervalo de frames suyos que
    caen dentro del tramo, con margen para no cortar por lo sano.

    Es lo que permite trabajar con muchas pasadas sin que explote el
    emparejado N-way: se reduce antes el material a lo que de verdad solapa.
    """
    from .dvfile import Capture, clip
    import time

    prof = base.prof
    probe = probe_set(prof, probes)
    lo = max(0, lo)
    hi = min(base.n - 1, hi)
    if verbose:
        print(f"indexando {base.name} [{lo}..{hi}] ({hi-lo+1} frames)...")
    hb, okb = fingerprints(base.data[lo:hi + 1], probe)
    index = defaultdict(list)
    for f in range(hi - lo + 1):
        for p in np.nonzero(okb[f])[0]:
            index[int(hb[f, p])].append(lo + f)
    for k in list(index):
        if len(index[k]) > MAX_PER_KEY:
            del index[k]

    os.makedirs(out_dir, exist_ok=True)
    made = []
    dst = os.path.join(out_dir, f"{tag}_{os.path.splitext(base.name)[0]}.dv")
    clip(base, dst, lo, hi - lo + 1)
    made.append(dst)
    if verbose:
        print(f"  {base.name}: {lo}..{hi} ({hi-lo+1}) -> {os.path.basename(dst)}")

    for path in donor_paths:
        t0 = time.time()
        cap = Capture(path)
        if cap.prof is not prof:
            continue
        hd, okd = fingerprints(cap.data, probe)
        hits = []
        for g in range(cap.n):
            v = Counter()
            for p in np.nonzero(okd[g])[0]:
                for f in index.get(int(hd[g, p]), ()):
                    v[f] += 1
            if not v:
                continue
            fb = cap.data[g]
            for f, _ in v.most_common(candidates):
                ok, _, _ = verify_pair(prof, base.data[f], fb, min_audio)
                if ok:
                    hits.append(g)
                    break
        del hd, okd
        matched = len(hits)
        if matched < min_matched:
            if verbose:
                print(f"  {cap.name}: {matched} frames en el tramo, se descarta")
            continue
        # Quedarse con el GRUPO CONTIGUO mas grande, no con el minimo y el
        # maximo: un solo emparejamiento espurio lejano estiraba el recorte a
        # miles de frames de material de otra parte de la cinta, que entraba al
        # indice sin enlazar con nada y se intercalaba con el bueno.
        groups = []
        cur = [hits[0]]
        for g in hits[1:]:
            if g - cur[-1] <= gap:
                cur.append(g)
            else:
                groups.append(cur)
                cur = [g]
        groups.append(cur)
        best = max(groups, key=len)
        dropped = matched - len(best)
        dlo, dhi = best[0], best[-1]
        if len(best) < min_matched:
            if verbose:
                print(f"  {cap.name}: {matched} sueltos sin grupo contiguo, se descarta")
            continue
        a = max(0, dlo - margin)
        n = min(cap.n - a, dhi - a + 1 + margin)
        dst = os.path.join(out_dir, f"{tag}_{os.path.splitext(cap.name)[0]}.dv")
        clip(cap, dst, a, n)
        made.append(dst)
        if verbose:
            extra = f", {dropped} sueltos descartados" if dropped else ""
            print(f"  {cap.name}: {len(best)} casan{extra} -> recorte {a}..{a+n-1} "
                  f"({n} frames, {n/25:.0f}s)  [{time.time()-t0:.0f}s]")
    return made


def align_to_tape(idx, cap_index, out_path):
    """Escribe una captura estirada a la linea temporal real de la cinta.

    Por cada frame de cinta emite la lectura de esa captura si la tiene, y si
    no repite la ultima. Asi se ve como se reproduciria de verdad: donde la
    captura perdio frames, la imagen se queda congelada. Puesto al lado del
    resultado fundido, es lo que ensena de un vistazo cuanto material se ha
    recuperado.
    """
    prof = idx.caps[cap_index].prof
    last = None
    missing = 0
    with open(out_path, "wb") as fh:
        for k in range(len(idx)):
            mine = [f for c, f in idx.reads(k) if c == cap_index]
            if mine:
                last = idx.caps[cap_index].frame(mine[0])
            else:
                missing += 1
            if last is None:
                continue
            fh.write(last.tobytes())
    return missing
