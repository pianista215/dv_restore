"""Reparar macrobloques FALSAMENTE sanos con el video de referencia.

Esto es un problema distinto del de conceal.py, y por eso va aparte.

conceal.py rellena lo que la camara marca con error. Pero parte del dano de
esta cinta no esta marcado: son macrobloques con ECC valido que la camara da
por buenos y que llevan basura escrita desde la grabacion. Las ocho pasadas los
leen igual, asi que ni la fusion ni la votacion por mayoria los detectan, y
conceal ni los mira porque solo trabaja sobre los marcados. Medido en un frame
del arranque: 24 macrobloques marcados con error y 207 que discrepan del DVD.
Los que se VEN son los segundos.

Hace falta una verdad de campo externa para detectarlos, y el video de
referencia lo es: es una observacion independiente de la misma cinta.

El criterio tiene dos niveles, y el segundo es el que protege la cinta:

  1. Por frame: si la referencia no esta bien alineada aqui (o sencillamente
     no tiene este material), su desacuerdo MEDIANO con el frame entero se
     dispara. Medido: 4,3 y 3,8 donde esta bien, 25,9 donde no hay cobertura.
     Por encima de `frame_trust` no se toca nada del frame. Esto descarta los
     tramos sin cobertura solo, sin mas umbrales.

  2. Por macrobloque, dentro de un frame fiable: se sustituye el que discrepa
     mucho Y ADEMAS no empalma con sus vecinos. Las dos condiciones hacen
     falta: la discrepancia sola se sesga hacia los bloques texturados
     (correlacion 0,34 con el detalle; los que pasan el umbral tienen detalle
     mediano 10,6 frente a 5,0 del resto), y un bloque muy texturado discrepa
     de una copia de DVD solo por serlo. La costura es la prueba independiente
     de que esta roto de verdad: los que pasan el umbral tienen el doble de
     salto con el vecino (6,0 frente a 2,9). Un bloque basura no pega con lo
     que tiene al lado; uno texturado si.

Solo se toca la luma. El croma se queda el que habia, igual que en conceal.
"""

import numpy as np

from . import native, render, shuffle
from . import encode as _encode
from .conceal import _budget_for
from .fixup import fit_segment


def disagreement(luma_dv, luma_ref, prof, rows, cols, covers):
    """RMSE por macrobloque entre el DV y la referencia, ya nivelada.

    Devuelve (e, ref_ajustada). e vale NaN donde la referencia no cubre.
    """
    m = np.isfinite(luma_ref)
    if m.sum() < 1000:
        return None, None
    g = np.polyfit(luma_ref[m], luma_dv[m], 1)
    ref = np.polyval(g, luma_ref)
    e = np.full(prof.n_video, np.nan, np.float64)
    for k in range(prof.n_video):
        r, c = int(rows[k]), int(cols[k])
        if not covers(r, c):
            continue
        q = ref[16 * r:16 * r + 16, 16 * c:16 * c + 16]
        if not np.isfinite(q).all():
            continue
        p = luma_dv[16 * r:16 * r + 16, 16 * c:16 * c + 16]
        e[k] = np.sqrt(np.mean((p - q) ** 2))
    return e, ref


def seam(luma, r, c, prof):
    """Salto de luma en los bordes del macrobloque frente al gradiente de al
    lado. Un bloque sano continua a sus vecinos; uno con basura, no."""
    H, W = luma.shape
    vals = []
    if c > 0:
        j = 16 * c
        s = np.abs(luma[16*r:16*r+16, j-1] - luma[16*r:16*r+16, j]).mean()
        g = np.abs(luma[16*r:16*r+16, j+1] - luma[16*r:16*r+16, j+2]).mean()
        vals.append(s / max(g, 1.0))
    if 16 * (c + 1) < W - 1:
        j = 16 * (c + 1) - 1
        s = np.abs(luma[16*r:16*r+16, j] - luma[16*r:16*r+16, j+1]).mean()
        g = np.abs(luma[16*r:16*r+16, j-2] - luma[16*r:16*r+16, j-1]).mean()
        vals.append(s / max(g, 1.0))
    if r > 0:
        i = 16 * r
        s = np.abs(luma[i-1, 16*c:16*c+16] - luma[i, 16*c:16*c+16]).mean()
        g = np.abs(luma[i+1, 16*c:16*c+16] - luma[i+2, 16*c:16*c+16]).mean()
        vals.append(s / max(g, 1.0))
    return max(vals) if vals else 0.0


def clean_sequence(frames, prof, store, v_of_i, table=None, thr=15.0,
                   frame_trust=8.0, min_seam=1.6, fm=None, progress=None):
    """Sustituye la luma de los macrobloques falsamente sanos.

    frames    lista de frames (se copian, no se tocan los originales)
    store     RefStore con el video de referencia ya extraido
    v_of_i    frame de la referencia que corresponde a cada frame de la lista

    Devuelve (salida, informe).
    """
    if table is None:
        table = shuffle.load(prof)
    rows = np.asarray(table) // prof.mb_cols
    cols = np.asarray(table) % prof.mb_cols
    out = [np.array(f) for f in frames]
    rep = dict(frames_skipped=0, flagged=0, rejected_seam=0, written=0,
               seg_reverted=0, invalid_written=0, medians=[])

    for i in range(len(frames)):
        if progress and i % 25 == 0:
            progress(i, len(frames))
        v = int(v_of_i[i])
        if v < 0 or v >= store.n:
            rep["frames_skipped"] += 1
            continue
        dv = render.decode(out[i], prof, "yraw")[0].astype(np.float64)
        e, ref = disagreement(dv, store.frame_dv(v, fm), prof, rows, cols,
                              store.covers)
        if e is None:
            rep["frames_skipped"] += 1
            continue
        med = float(np.nanmedian(e))
        rep["medians"].append(med)
        if not np.isfinite(med) or med > frame_trust:
            rep["frames_skipped"] += 1
            continue

        cand = np.nonzero(np.nan_to_num(e) > thr)[0]
        if not len(cand):
            continue
        rep["flagged"] += len(cand)
        keep = []
        for k in cand:
            k = int(k)
            if seam(dv, int(rows[k]), int(cols[k]), prof) >= min_seam:
                keep.append(k)
            else:
                rep["rejected_seam"] += 1
        if not keep:
            continue

        by_seg = {}
        for k in keep:
            by_seg.setdefault(k // 5, []).append(k % 5)
        for s, ms in by_seg.items():
            off = int(prof.seg[s, 0])
            src = bytes(out[i][off:off + 400])
            seg = native.parse(src)
            if not seg.ok:
                continue
            for m in ms:
                r, c = int(rows[s * 5 + m]), int(cols[s * 5 + m])
                _encode.encode_mb(seg, m, ref[16*r:16*r+16, 16*c:16*c+16],
                                  max_bits=_budget_for(seg, m))
                seg.mb[m].sta = 0
            fit_segment(seg, protect=tuple(x for x in range(5) if x not in ms))
            buf, _ = native.pack(seg, src)
            back = native.parse(buf)
            if not back.ok:
                rep["seg_reverted"] += 1
                continue
            good = True
            for mm in range(5):
                for jj in range(6):
                    a, b = seg.mb[mm].b[jj], back.mb[mm].b[jj]
                    if (a.dc != b.dc or a.cls != b.cls or a.mode != b.mode
                            or a.ntok != b.ntok):
                        good = False
                        break
                if not good:
                    break
            if not good:
                rep["seg_reverted"] += 1
                continue
            out[i][off:off + 400] = np.frombuffer(buf, np.uint8)
            rep["written"] += len(ms)
    rep["medians"] = np.array(rep["medians"]) if rep["medians"] else np.zeros(0)
    return out, rep
