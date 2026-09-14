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
from .fixup import sanitize_mb, fit_segment, SEG_CAPACITY_BITS as SEG_CAPACITY
from . import interp as _interp
from . import encode as _encode

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


def best_shift(a, b, prof, max_dr=2, max_dc=4, min_cells=150):
    """Desplazamiento global entre dos frames, en macrobloques enteros.

    Devuelve (dr, dc) en el convenio ORIGEN = DESTINO + (dr, dc): para rellenar
    el macrobloque que esta en (r, c) del frame destino hay que coger el que
    esta en (r + dr, c + dc) del frame origen.

    Copiar siempre de la MISMA posicion es lo que deja los macrobloques
    corridos cuando la camara panea: se pega contenido que pertenece unos
    pixeles mas alla. Medido sobre este material, compensar el desplazamiento
    baja el error un 21% de media, y hasta un 78% en los paneos rapidos, que
    son justo donde la puerta de movimiento rechazaba copiar.

    Solo se compensa en pasos de macrobloque entero: asi se sigue copiando 80
    bytes tal cual, sin recodificar nada. Los desplazamientos de menos de medio
    macrobloque salen como (0, 0), pero ahi el error ya es pequeno.
    """
    R, C = prof.mb_rows, prof.mb_cols
    best = (None, 0, 0)
    for dr in range(-max_dr, max_dr + 1):
        for dc in range(-max_dc, max_dc + 1):
            # destino[r, c] frente a origen[r + dr, c + dc]
            r0, r1 = max(0, -dr), min(R, R - dr)
            c0, c1 = max(0, -dc), min(C, C - dc)
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            A = a.grid_dc[r0:r1, c0:c1]
            OA = a.grid_ok[r0:r1, c0:c1]
            B = b.grid_dc[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            OB = b.grid_ok[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            m = OA & OB
            n = int(m.sum())
            if n < min_cells:
                continue
            e = float(np.abs(A[m] - B[m]).mean())
            if best[0] is None or e < best[0]:
                best = (e, dr, dc)
    return best[1], best[2]


class _Shifted:
    """Vista de un frame corrida (dr, dc) macrobloques, para medir el
    movimiento que QUEDA despues de compensar."""

    __slots__ = ("grid_dc", "grid_ok")

    def __init__(self, info, dr, dc, prof):
        R, C = prof.mb_rows, prof.mb_cols
        self.grid_dc = np.zeros((R, C), np.float32)
        self.grid_ok = np.zeros((R, C), bool)
        r0, r1 = max(0, -dr), min(R, R - dr)
        c0, c1 = max(0, -dc), min(C, C - dc)
        if r1 > r0 and c1 > c0:
            self.grid_dc[r0:r1, c0:c1] = info.grid_dc[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            self.grid_ok[r0:r1, c0:c1] = info.grid_ok[r0 + dr:r1 + dr, c0 + dc:c1 + dc]


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


def _budget_for(seg, m):
    """Bits que le quedan a un macrobloque sin quitarselos a los demas."""
    from .bitstream import VLC_LEN, HDR_BITS
    used = 0
    for mm in range(5):
        if mm == m:
            continue
        for j in range(6):
            b = seg.mb[mm].b[j]
            used += HDR_BITS + int(sum(int(VLC_LEN[b.tok[i]])
                                       for i in range(int(b.ntok))))
    # los dos bloques de croma del propio macrobloque no se tocan
    for j in (4, 5):
        b = seg.mb[m].b[j]
        used += HDR_BITS + int(sum(int(VLC_LEN[b.tok[i]])
                                   for i in range(int(b.ntok))))
    return max(0, SEG_CAPACITY - used)


def conceal_sequence(frames, prof, max_dist=25, thr=20.0, min_nb=6,
                     table=None, progress=None, stats=None, verify=True,
                     outlier_factor=3.0, compensate=True, interpolate=True,
                     interp_span=10):
    if table is None:
        table = shuffle.load(prof)
    pos = np.asarray(table, np.int64)
    rows, cols = pos // prof.mb_cols, pos % prof.mb_cols
    # posicion de pantalla -> bloque del flujo, para poder coger el macrobloque
    # de una posicion distinta a la del destino
    inv = np.zeros(prof.n_video, np.int64)
    inv[pos] = np.arange(prof.n_video)

    n = len(frames)
    info = [FrameInfo(np.asarray(frames[i]), prof, table) for i in range(n)]
    out = [np.array(frames[i]) for i in range(n)]
    ids = prof.video[:, None] + np.arange(3)[None, :]

    check = [0, 0]
    trimmed = 0
    interpolated = 0
    luma = _interp.LumaCache(frames, prof) if interpolate else None
    mvcache = {}

    def motion(j1, j2, r, c):
        """Movimiento entre dos frames, cacheado por region de 4x4
        macrobloques: estimarlo bloque a bloque multiplicaba por 16 el coste y
        el movimiento de camara es casi el mismo en toda la region."""
        key = (j1, j2, r // 4, c // 4)
        if key not in mvcache:
            yy = min(max((r // 4) * 4 + 2, 2), prof.mb_rows - 3) * 16
            xx = min(max((c // 4) * 4 + 2, 2), prof.mb_cols - 3) * 16
            mvcache[key] = _interp.estimate_motion(luma(j1), luma(j2), yy, xx)
        return mvcache[key]
    bridged = bridged_try = rescued = shifted = 0
    filled = motion_reject = no_source = no_support = 0
    seg_copied = seg_repacked = seg_reverted = 0
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
        sh = {}                      # desplazamiento global por frame origen
        # el frame no tiene con que medir por dentro: hara falta el puente
        blind = int((~bad).sum()) < min_nb
        src = np.full(prof.n_video, -1, np.int64)
        src_blk = np.full(prof.n_video, -1, np.int64)
        for k in np.nonzero(bad)[0]:
            r, c = int(rows[k]), int(cols[k])
            best = None
            for j in cand:
                if j not in sh:
                    dr, dc = (best_shift(info[i], info[j], prof)
                              if compensate else (0, 0))
                    sh[j] = (dr, dc)
                    pm[j] = PairMotion(info[i],
                                       _Shifted(info[j], dr, dc, prof)
                                       if (dr or dc) else info[j], prof)
                dr, dc = sh[j]
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < prof.mb_rows and 0 <= c2 < prof.mb_cols):
                    continue
                k2 = int(inv[r2 * prof.mb_cols + c2])
                if info[j].bad[k2]:
                    continue
                for w, (win, fac) in enumerate(zip(WINDOWS, WIN_FACTOR)):
                    d, cnt = pm[j].at(r, c, win)
                    if d is None or cnt < min_nb:
                        continue
                    score = d / fac
                    if best is None or score < best[0]:
                        best = (score, j, w, d, k2)
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
                        best = (est / fac, near, w, est, k)
                        break
                    bridged_try += 1
            if best is None:
                if not any((not info[j].bad[k]) for j in cand):
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
            d_keep = abs(float(info[i].dcb[k]) - float(info[best[1]].dcb[best[4]]))
            outlier = d_keep > outlier_factor * max(best[3], 4.0)
            if best[0] > thr and not outlier:
                motion_reject += 1
            else:
                src[k] = best[1]
                src_blk[k] = best[4]
                used_win[best[2]] += 1
                if best[4] != k:
                    shifted += 1
                filled += 1
                if outlier and best[0] > thr:
                    rescued += 1
                if blind:
                    bridged += 1

        # Interpolar: componer el instante intermedio entre la version buena
        # de antes y la de despues, compensando el movimiento a nivel de
        # pixel. Copiar el macrobloque tal cual solo vale si la escena esta
        # quieta; medido, interpolar baja el error mediano un 41%.
        px = {}
        if interpolate:
            for k in np.nonzero(src >= 0)[0]:
                r, c = int(rows[k]), int(cols[k])
                if not (1 <= r < prof.mb_rows - 2 and 1 <= c < prof.mb_cols - 2):
                    continue
                j1 = next((j for j in range(i - 1, max(-1, i - interp_span), -1)
                           if not info[j].bad[k]), None)
                j2 = next((j for j in range(i + 1, min(n, i + interp_span))
                           if not info[j].bad[k]), None)
                if j1 is None or j2 is None:
                    continue
                dy, dx = motion(j1, j2, r, c)
                t = (i - j1) / (j2 - j1)
                s1 = _interp.sample(luma(j1), r * 16, c * 16, -dy * t, -dx * t)
                s2 = _interp.sample(luma(j2), r * 16, c * 16,
                                    dy * (1 - t), dx * (1 - t))
                if s1 is None or s2 is None:
                    continue
                px[k] = (1 - t) * s1 + t * s2

        segs = np.nonzero((src.reshape(prof.n_seg, 5) >= 0).any(axis=1))[0]
        if len(segs) == 0:
            continue
        base_ids = out[i][ids]
        for s in segs:
            off = int(prof.seg[s, 0])
            ss = src[s * 5:(s + 1) * 5]
            sb = src_blk[s * 5:(s + 1) * 5]
            uniq = set(int(x) for x in ss if x >= 0)
            plano = all(int(sb[m]) == s * 5 + m for m in range(5) if ss[m] >= 0)
            hay_px = any((s * 5 + m) in px for m in range(5))
            if len(uniq) == 1 and (ss >= 0).all() and plano and not hay_px:
                j = uniq.pop()
                out[i][off:off + 400] = frames[j][off:off + 400]
                for m in range(5):
                    o = int(prof.seg[s, m])
                    out[i][o:o + 3] = base_ids[s * 5 + m]
                    out[i][o + 3] &= 0x0F
                seg_copied += 1
                continue
            dst = native.parse(bytes(out[i][off:off + 400]))
            # Solo se confia en el desbordamiento del segmento si ademas de
            # parsear NO habia ningun macrobloque roto en el. Un segmento con
            # un macrobloque danado parsea igual, pero su basura vive en el
            # deposito compartido y la continuacion de los vecinos sanos la
            # lee. Mientras el segmento tenia STA mezclado el decodificador
            # ignoraba ese deposito; al repararlo y marcarlo todo sano, empieza
            # a aplicarlo. Se veia como macrobloques sanos con el color roto.
            trust = bool(dst.ok) and not bool(info[i].bad[s * 5:(s + 1) * 5].any())
            cache = {}
            for m in range(5):
                j = int(ss[m])
                if j < 0:
                    # No se sustituye, pero hay que dejarlo emitible. Y si el
                    # segmento de partida NO era valido hay que truncarlo a su
                    # area aunque el macrobloque estuviera sano: lo que venga
                    # del desbordamiento puede ser basura mal repartida, y al
                    # marcar luego todo el segmento como sano el decodificador
                    # deja de ignorarlo y lo aplica. Se veia como macrobloques
                    # sanos con el color disparatado.
                    sanitize_mb(dst.mb[m], trust)
                    continue
                k2 = int(sb[m])
                s2, m2 = k2 // 5, k2 % 5
                key = (j, s2)
                if key not in cache:
                    o2 = int(prof.seg[s2, 0])
                    cache[key] = native.parse(bytes(frames[j][o2:o2 + 400]))
                dst.mb[m] = cache[key].mb[m2]
                dst.mb[m].sta = 0
                sanitize_mb(dst.mb[m], bool(cache[key].ok))
                kk = s * 5 + m
                if kk in px:
                    # el croma se queda el del macrobloque copiado; solo se
                    # sustituye la luma, que es donde se nota el desplazamiento
                    libre = _budget_for(dst, m)
                    _encode.encode_mb(dst, m, px[kk], max_bits=libre)
                    interpolated += 1
                for t in range(3):
                    dst.mb[m].id[t] = int(base_ids[s * 5 + m][t])
            # no tocar los macrobloques que no hemos sustituido: estan sanos
            keep = tuple(m for m in range(5) if int(ss[m]) < 0)
            trimmed += max(0, fit_segment(dst, protect=keep))
            buf, nlost = native.pack(dst, bytes(out[i][off:off + 400]))
            if nlost:
                check[1] += 1
            # Red de seguridad: lo que se escribe tiene que volver a leerse
            # igual que lo que queriamos escribir. Si no, se deja el segmento
            # como estaba. Mas vale no arreglar un macrobloque que estropear
            # los cuatro vecinos.
            back = native.parse(buf)
            good = bool(back.ok)
            if good:
                for mm in range(5):
                    for jj in range(6):
                        a, b_ = dst.mb[mm].b[jj], back.mb[mm].b[jj]
                        if (a.dc != b_.dc or a.cls != b_.cls or a.mode != b_.mode
                                or a.ntok != b_.ntok):
                            good = False
                            break
                    if not good:
                        break
            if not good:
                check[0] += 1
                seg_reverted += 1
                continue
            out[i][off:off + 400] = np.frombuffer(buf, np.uint8)
            seg_repacked += 1
        if progress and i % 50 == 0:
            progress(i, n)

    rep = dict(filled=filled, motion_reject=motion_reject,
               no_source=no_source, no_support=no_support,
               seg_copied=seg_copied, seg_repacked=seg_repacked,
               seg_reverted=seg_reverted,
               win_used=used_win.tolist(), bridged=bridged,
               rescued=rescued, shifted=shifted, interpolated=interpolated,
               invalid_written=check[0], overflow=check[1], trimmed=trimmed,
               diffs=np.array(diffs) if diffs else np.zeros(0))
    return out, rep
