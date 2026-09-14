"""Tapar los macrobloques que estan rotos en todas las lecturas.

No se inventa imagen: se copia el mismo macrobloque de un frame vecino del
propio material, y solo cuando hay pruebas de que esa zona del cuadro esta
quieta. La prueba sale del plano DC, que se lee sin decodificar el bitstream:
se comparan los macrobloques SANOS que rodean al roto entre el frame de destino
y el candidato.

El umbral no compara contra la perfeccion, sino contra la ALTERNATIVA, que no
es dejarlo en blanco: es la estimacion que ya hizo la camara, y que suele ser
un trozo de un frame viejo. Medido con verdad de campo, la copia tiene menos
error que esa estimacion en todos los tramos de movimiento hasta 40 (a 12-20,
8,0 frente a 12,0) y solo pierde por encima. Con un umbral prudente se
rechazaban copias que eran mejores, y el resultado eran parpadeos: un frame
rancio entre dos buenos.

El alcance de la busqueda (max_dist) importa mas de lo que parece: el dano
llega en RACHAS de frames seguidos. Con una cinta que pierde las secuencias DIF
pares durante diez frames, buscar origen solo a +-6 deja sin arreglo los del
medio, y salen como bandas horizontales con imagen de otro momento.

El problema practico es que el dano viene en manchas grandes: seis de cada diez
macrobloques rotos no tienen NI UN vecino sano a su alrededor con el que medir.
Por eso la ventana crece por escalones hasta encontrar apoyo suficiente, y el
umbral se relaja a medida que la medida se vuelve mas global y menos fiable.

Y el caso peor son los frames ocultados ENTEROS por la camara, que no tienen ni
un macrobloque sano: ahi no hay nada dentro del frame con que medir. Son justo
los que mas se notan, porque caen entre frames bien restaurados y el ojo ve un
tiron. Para esos se mide el movimiento entre los dos frames BUENOS que lo
rodean: si la escena no se mueve de uno a otro, tampoco se mueve en medio, y
rellenar es seguro. El error que se espera cometer se estima repartiendo ese
movimiento en proporcion a la distancia.
"""

import numpy as np

from . import dcplane, native, shuffle
from .fixup import sanitize_mb, fit_segment

# escalones de la ventana de medida, en macrobloques de radio, y cuanto se
# endurece el umbral segun la medida es mas lejana (y por tanto menos fiable)
WINDOWS = (3, 6, 12, 0)          # 0 = frame entero
WIN_FACTOR = (1.0, 0.85, 0.7, 0.55)


class FrameInfo:
    __slots__ = ("bad", "grid_dc", "grid_ok", "dcb")

    def __init__(self, frame, prof, table):
        self.bad = dcplane.sta(frame, prof) != 0
        dc = dcplane.dc_raw(frame, prof)[:, :4].mean(axis=1)
        self.dcb = dc
        g = np.zeros(prof.mb_rows * prof.mb_cols, np.float32)
        g[table] = dc
        self.grid_dc = g.reshape(prof.mb_rows, prof.mb_cols)
        ok = np.zeros(prof.mb_rows * prof.mb_cols, bool)
        ok[table] = ~self.bad
        self.grid_ok = ok.reshape(prof.mb_rows, prof.mb_cols)


def _integral(a):
    return np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))


def _win_sum(I, r0, r1, c0, c1):
    return I[r1, c1] - I[r0, c1] - I[r1, c0] + I[r0, c0]


class PairMotion:
    """Diferencia media de DC entre dos frames, consultable por ventanas.

    Se guardan las sumas acumuladas para que preguntar por cualquier ventana
    alrededor de cualquier macrobloque cueste lo mismo, sea del tamano que sea.
    """

    def __init__(self, a, b, prof):
        v = a.grid_ok & b.grid_ok
        d = np.abs(a.grid_dc - b.grid_dc) * v
        self.Iv = _integral(v.astype(np.float64))
        self.Id = _integral(d.astype(np.float64))
        self.rows, self.cols = prof.mb_rows, prof.mb_cols
        self.total_v = float(v.sum())
        self.total_d = float(d.sum())

    def at(self, r, c, win):
        if win == 0:
            if self.total_v == 0:
                return None, 0
            return self.total_d / self.total_v, int(self.total_v)
        r0, r1 = max(0, r - win), min(self.rows, r + win + 1)
        c0, c1 = max(0, c - win), min(self.cols, c + win + 1)
        n = _win_sum(self.Iv, r0, r1, c0, c1)
        if n == 0:
            return None, 0
        return _win_sum(self.Id, r0, r1, c0, c1) / n, int(n)


def conceal_sequence(frames, prof, max_dist=25, thr=20.0, min_nb=6,
                     table=None, progress=None, stats=None, verify=True,
                     outlier_factor=3.0):
    if table is None:
        table = shuffle.load(prof)
    pos = np.asarray(table, np.int64)
    rows, cols = pos // prof.mb_cols, pos % prof.mb_cols

    n = len(frames)
    info = [FrameInfo(np.asarray(frames[i]), prof, table) for i in range(n)]
    out = [np.array(frames[i]) for i in range(n)]
    ids = prof.video[:, None] + np.arange(3)[None, :]

    check = [0, 0]
    trimmed = 0
    bridged = bridged_try = rescued = 0
    filled = motion_reject = no_source = no_support = 0
    seg_copied = seg_repacked = 0
    used_win = np.zeros(len(WINDOWS), np.int64)
    diffs = []

    for i in range(n):
        bad = info[i].bad
        if not bad.any():
            continue
        cand = [j for d in range(1, max_dist + 1) for j in (i - d, i + d)
                if 0 <= j < n]
        pm = {}
        pm2 = {}
        # el frame no tiene con que medir por dentro: hara falta el puente
        blind = int((~bad).sum()) < min_nb
        src = np.full(prof.n_video, -1, np.int64)
        for k in np.nonzero(bad)[0]:
            r, c = int(rows[k]), int(cols[k])
            best = None
            for j in cand:
                if info[j].bad[k]:
                    continue
                if j not in pm:
                    pm[j] = PairMotion(info[i], info[j], prof)
                for w, (win, fac) in enumerate(zip(WINDOWS, WIN_FACTOR)):
                    d, cnt = pm[j].at(r, c, win)
                    if d is None or cnt < min_nb:
                        continue
                    score = d / fac
                    if best is None or score < best[0]:
                        best = (score, j, w, d)
                    break
                if best is not None and best[1] == j and best[0] < 0.5:
                    break
            if best is None and blind:
                # puente: los dos frames buenos que rodean a este
                j1 = next((j for j in range(i - 1, max(-1, i - max_dist - 1), -1)
                           if not info[j].bad[k]), None)
                j2 = next((j for j in range(i + 1, min(n, i + max_dist + 1))
                           if not info[j].bad[k]), None)
                if j1 is not None and j2 is not None:
                    key = (j1, j2)
                    if key not in pm2:
                        pm2[key] = PairMotion(info[j1], info[j2], prof)
                    for w, (win, fac) in enumerate(zip(WINDOWS, WIN_FACTOR)):
                        dd, cnt = pm2[key].at(r, c, win)
                        if dd is None or cnt < min_nb:
                            continue
                        near = j1 if (i - j1) <= (j2 - i) else j2
                        # el movimiento medido cubre j2-j1 frames; al copiar
                        # desde el mas cercano solo se hereda su parte
                        est = dd * abs(i - near) / max(j2 - j1, 1)
                        best = (est / fac, near, w, est)
                        break
                    bridged_try += 1
            if best is None:
                if not any(not info[j].bad[k] for j in cand):
                    no_source += 1
                else:
                    no_support += 1
                continue
            diffs.append(best[3])
            # Copiar tambien cuando lo que hay puesto es un disparate.
            # La puerta de movimiento mide cuanto se mueve la escena entre los
            # dos frames: si la estimacion de la camara para ESTE macrobloque
            # se aleja mucho mas que eso, no es una estimacion mala, es imagen
            # de otro momento. Pasa cuando la cinta pierde secuencias DIF
            # enteras durante una racha larga: salen bandas horizontales con
            # una escena distinta, y rechazarlas por movimiento no tiene
            # sentido porque lo que se conserva es peor que cualquier copia.
            d_keep = abs(float(info[i].dcb[k]) - float(info[best[1]].dcb[k]))
            outlier = d_keep > outlier_factor * max(best[3], 4.0)
            if best[0] > thr and not outlier:
                motion_reject += 1
            else:
                src[k] = best[1]
                used_win[best[2]] += 1
                filled += 1
                if outlier and best[0] > thr:
                    rescued += 1
                if blind:
                    bridged += 1

        segs = np.nonzero((src.reshape(prof.n_seg, 5) >= 0).any(axis=1))[0]
        if len(segs) == 0:
            continue
        base_ids = out[i][ids]
        for s in segs:
            off = int(prof.seg[s, 0])
            ss = src[s * 5:(s + 1) * 5]
            uniq = set(int(x) for x in ss if x >= 0)
            if len(uniq) == 1 and (ss >= 0).all():
                j = uniq.pop()
                out[i][off:off + 400] = frames[j][off:off + 400]
                for m in range(5):
                    o = int(prof.seg[s, m])
                    out[i][o:o + 3] = base_ids[s * 5 + m]
                    out[i][o + 3] &= 0x0F
                seg_copied += 1
                continue
            dst = native.parse(bytes(out[i][off:off + 400]))
            trust = bool(dst.ok)
            cache = {}
            for m in range(5):
                j = int(ss[m])
                if j < 0:
                    # no se sustituye, pero hay que dejarlo emitible igual
                    sanitize_mb(dst.mb[m], trust)
                    continue
                if j not in cache:
                    cache[j] = native.parse(bytes(frames[j][off:off + 400]))
                dst.mb[m] = cache[j].mb[m]
                dst.mb[m].sta = 0
                sanitize_mb(dst.mb[m], bool(cache[j].ok))
                for t in range(3):
                    dst.mb[m].id[t] = int(base_ids[s * 5 + m][t])
            trimmed += max(0, fit_segment(dst))
            buf, nlost = native.pack(dst, bytes(out[i][off:off + 400]))
            if nlost:
                check[1] += 1
            if check is not None and not native.parse(buf).ok:
                check[0] += 1
            out[i][off:off + 400] = np.frombuffer(buf, np.uint8)
            seg_repacked += 1
        if progress and i % 50 == 0:
            progress(i, n)

    rep = dict(filled=filled, motion_reject=motion_reject,
               no_source=no_source, no_support=no_support,
               seg_copied=seg_copied, seg_repacked=seg_repacked,
               win_used=used_win.tolist(), bridged=bridged,
               rescued=rescued,
               invalid_written=check[0], overflow=check[1], trimmed=trimmed,
               diffs=np.array(diffs) if diffs else np.zeros(0))
    return out, rep
