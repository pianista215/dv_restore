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

  1. Por frame: hay que saber si la referencia esta bien puesta AQUI, y eso no
     se le puede preguntar al propio frame. Un frame muy danado discrepa de la
     referencia y tiene coste de alineacion alto, pero no porque la referencia
     falle: porque lo nuestro esta roto. Medido, las dos causas se confunden
     del todo (frame destrozado pero cubierto: coste 0,43; frame sin
     cobertura: 0,38). Quien las separa es el VECINDARIO, porque un frame roto
     en medio de un tramo bien alineado sigue estando bien alineado. Ver
     refvideo.frame_trust().

  2. Por macrobloque, dentro de un frame fiable: se sustituye el que discrepa
     mucho Y ADEMAS no empalma con sus vecinos. Las dos condiciones hacen
     falta: la discrepancia sola se sesga hacia los bloques texturados
     (correlacion 0,34 con el detalle; los que pasan el umbral tienen detalle
     mediano 10,6 frente a 5,0 del resto), y un bloque muy texturado discrepa
     de una copia de DVD solo por serlo. La costura es la prueba independiente
     de que esta roto de verdad: los que pasan el umbral tienen el doble de
     salto con el vecino (6,0 frente a 2,9). Un bloque basura no pega con lo
     que tiene al lado; uno texturado si.

Se toca la luma y, si se da el croma de la referencia, tambien el color. El
croma importa mas de lo que parece: con la luma ya reparada y lisa, los bloques
de croma rotos se ven como manchas naranjas y azules encima. Y el croma del DVD
es bueno -- casa con el croma SANO del DV con error mediano 2,4 a 3,6, medido.
"""

import numpy as np

from . import native, render, shuffle
from . import encode as _encode
from .conceal import _budget_for
from .fixup import fit_segment


MIN_COVER = 0.4          # fraccion minima del macrobloque que la referencia cubre


def disagreement(luma_dv, luma_ref, prof, rows, cols, covers=None):
    """Desacuerdo por macrobloque entre el DV y la referencia, ya nivelada.

    Devuelve (e_baja, e_alta, ref_ajustada), ambas NaN donde no hay cobertura
    suficiente.

    Se separan DOS desacuerdos porque significan cosas distintas:

      e_baja  sobre medias de 4x4, o sea la estructura. Un bloque correcto pero
              mas nitido que la referencia coincide aqui; uno roto, no. Es el
              detector: medido sobre un frame limpio, su p99 es 3,9 (el del
              desacuerdo completo, 5,0), asi que discrimina con el umbral mas
              bajo y menos falsos positivos.

      e_alta  lo que queda, que es basicamente la diferencia de nitidez. Sirve
              para NO tocar un bloque que solo discrepa por ser mas fino.

    La cobertura se admite PARCIAL. La referencia es de 704 columnas y el DV de
    720, asi que la columna 0 y la 44 tienen la mitad de sus pixeles fuera y la
    fila 0 una linea. Exigir cobertura completa tiraba el 6,5% de la imagen, y
    justo el borde, que se ve. Donde falta se conserva lo nuestro.
    """
    m = np.isfinite(luma_ref)
    if m.sum() < 1000:
        return None, None, None
    g = np.polyfit(luma_ref[m], luma_dv[m], 1)
    ref = np.polyval(g, luma_ref)
    lo = np.full(prof.n_video, np.nan, np.float64)
    hi = np.full(prof.n_video, np.nan, np.float64)
    for k in range(prof.n_video):
        r, c = int(rows[k]), int(cols[k])
        if covers is not None and not covers(r, c):
            pass
        q = ref[16 * r:16 * r + 16, 16 * c:16 * c + 16]
        ok = np.isfinite(q)
        if q.shape != (16, 16) or ok.mean() < MIN_COVER:
            continue
        p = luma_dv[16 * r:16 * r + 16, 16 * c:16 * c + 16]
        d = np.where(ok, p - q, 0.0)
        n = ok.sum()
        # estructura: medias de 4x4 sobre lo cubierto
        w = ok.reshape(4, 4, 4, 4).sum(axis=(1, 3))
        dl = d.reshape(4, 4, 4, 4).sum(axis=(1, 3))
        good = w > 0
        lo[k] = np.sqrt(np.mean((dl[good] / w[good]) ** 2))
        hi[k] = np.sqrt((d ** 2).sum() / n)
    return lo, hi, ref


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


def clean_sequence(frames, prof, store, v_of_i, trust=None, table=None,
                   thr=4.0, trust_thr=0.25, min_seam=1.6, detail_floor=6.0,
                   fm=None, chroma=None, progress=None):
    """Sustituye la luma de los macrobloques falsamente sanos.

    frames    lista de frames (se copian, no se tocan los originales)
    store     RefStore con el video de referencia ya extraido
    v_of_i    frame de la referencia que corresponde a cada frame de la lista
    trust     fiabilidad de la alineacion por frame (refvideo.frame_trust);
              si falta, se trata todo
    chroma    ChromaStore opcional; si se da, se sustituye tambien el color

    Devuelve (salida, informe).
    """
    if table is None:
        table = shuffle.load(prof)
    rows = np.asarray(table) // prof.mb_cols
    cols = np.asarray(table) % prof.mb_cols
    out = [np.array(f) for f in frames]
    rep = dict(frames_skipped=0, flagged=0, rejected_seam=0, written=0,
               chroma_written=0, partial=0, seg_reverted=0, invalid_written=0,
               medians=[])

    for i in range(len(frames)):
        if progress and i % 25 == 0:
            progress(i, len(frames))
        v = int(v_of_i[i])
        if v < 0 or v >= store.n or (trust is not None and trust[i] >= trust_thr):
            rep["frames_skipped"] += 1
            continue
        dv = render.decode(out[i], prof, "yraw")[0].astype(np.float64)
        elo, ehi, ref = disagreement(dv, store.frame_dv(v, fm), prof, rows, cols)
        cro = (chroma.planes_dv(v, prof.width, prof.height)
               if chroma is not None else None)
        if elo is None:
            rep["frames_skipped"] += 1
            continue
        rep["medians"].append(float(np.nanmedian(elo)))

        cand = np.nonzero(np.nan_to_num(elo) > thr)[0]
        if not len(cand):
            continue
        rep["flagged"] += len(cand)
        # La costura protege a los bloques TEXTURADOS: uno muy texturado
        # discrepa de una copia de DVD solo por serlo, pero sigue pegando con
        # sus vecinos. Uno liso que discrepa no tiene esa excusa, y ademas en
        # un frame donde esta roto todo los vecinos tampoco pegan, asi que
        # exigir costura ahi rechazaria justo lo que hay que arreglar.
        keep = []
        for k in cand:
            k = int(k)
            r, c = int(rows[k]), int(cols[k])
            p = dv[16 * r:16 * r + 16, 16 * c:16 * c + 16]
            det = (np.abs(np.diff(p, axis=0)).mean()
                   + np.abs(np.diff(p, axis=1)).mean())
            # solo se perdona al bloque que discrepa en ALTA frecuencia (es
            # mas nitido que la referencia) y ademas empalma con sus vecinos.
            # Si la estructura ya discrepa, esta roto por texturado que sea.
            solo_nitidez = elo[k] < 0.6 * ehi[k]
            if (det <= detail_floor or not solo_nitidez
                    or seam(dv, r, c, prof) >= min_seam):
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
                cc = None
                if cro is not None:
                    q = cro[:, 8*r:8*r+8, 8*c:8*c+8]
                    if np.isfinite(q).all():
                        cc = q
                # donde la referencia no llega se conserva lo nuestro
                patch = ref[16*r:16*r+16, 16*c:16*c+16]
                if not np.isfinite(patch).all():
                    patch = np.where(np.isfinite(patch), patch,
                                     dv[16*r:16*r+16, 16*c:16*c+16])
                    rep["partial"] += 1
                _encode.encode_mb(seg, m, patch,
                                  max_bits=_budget_for(seg, m, cc is not None),
                                  chroma=cc)
                seg.mb[m].sta = 0
                if cc is not None:
                    rep["chroma_written"] += 1
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
