"""Fusion de varias lecturas del mismo frame de cinta.

La unidad de trabajo es el SEGMENTO de video (5 macrobloques, 400 bytes),
porque los 5 comparten el desbordamiento VLC. La estrategia va de lo mas seguro
a lo mas delicado:

  1. Si el segmento del lienzo esta entero y su bitstream decodifica, no se
     toca.
  2. Si alguna otra lectura lo tiene entero y valido, se copian sus 400 bytes
     TAL CUAL. Es exacto y sin perdida; solo hay que reescribir los 3 bytes de
     ID DIF, que difieren entre capturas aunque el contenido sea el mismo.
  3. Si ninguna lectura lo tiene entero, se parsean todas y se elige el mejor
     origen macrobloque a macrobloque, volviendo a serializar el segmento. Es
     la unica via que permite juntar lo bueno de dos capturas dentro de un
     mismo segmento, y sigue siendo sin perdida porque cada macrobloque lleva
     su propio QNO y no hay que recuantificar.

Los macrobloques que no tienen origen sano en ninguna lectura se dejan
marcados, para que los recoja la pasada de ocultacion.
"""

import ctypes
import numpy as np

from . import dcplane, native
from .bitstream import EOB
from .fixup import sanitize_mb, fit_segment

# calidad de un macrobloque como origen, de mejor a peor
SRC_CLEAN = 3      # sano y en un segmento cuyo bitstream decodifica
SRC_AREA = 2       # sano, pero su segmento no decodifica: solo su area fija
SRC_NONE = 0       # marcado con error


class Read:
    """Una lectura de un frame de cinta, con su diagnostico ya hecho."""

    __slots__ = ("frame", "prof", "sta", "seg_ok", "seg_bad", "n_bad",
                 "n_false", "cap", "idx")

    def __init__(self, frame, prof, cap=None, idx=None):
        self.frame = np.ascontiguousarray(frame, np.uint8)
        self.prof = prof
        self.cap = cap
        self.idx = idx
        self.sta = dcplane.sta(self.frame, prof)
        sc = native.scan_frame(self.frame, prof)
        self.seg_ok = sc.seg_ok.astype(bool)
        self.seg_bad = (self.sta != 0).reshape(prof.n_seg, 5).any(axis=1)
        self.n_bad = int((self.sta != 0).sum())
        self.n_false = int((~self.seg_ok & ~self.seg_bad).sum())

    def quality(self):
        """Calidad como origen de cada uno de los 1620 macrobloques."""
        q = np.where(self.sta != 0, SRC_NONE, SRC_AREA).astype(np.int8)
        clean = np.repeat(self.seg_ok, 5)
        q[(q == SRC_AREA) & clean] = SRC_CLEAN
        return q

    def seg_full(self):
        """Segmentos enteros y validos: se pueden copiar sin parsear."""
        return self.seg_ok & ~self.seg_bad


def merge_frame(reads, prof, order=None, prefer=None, hysteresis=0.20):
    """Funde varias lecturas del mismo frame de cinta.

    El LIENZO es la lectura que se toma de base y de la que salen el subcodigo,
    el audio y los macrobloques que no tiene sana ninguna. Elegir siempre el
    menos danado hace que el lienzo vaya saltando de una captura a otra, y como
    cada camara oculta sus errores a su manera, eso parpadea: medido, cuando el
    lienzo cambia la diferencia entre frames consecutivos sube de 7,5 a 11,3.

    Por eso, si la lectura que fue lienzo en el frame anterior sigue estando
    razonablemente sana, se mantiene. 'hysteresis' es cuanto peor se le
    consiente ser (0.20 = hasta un 20% mas de macrobloques danados).
    """
    if len(reads) == 1:
        r = reads[0]
        return r.frame.copy(), dict(kept=prof.n_seg, copied=0, repacked=0,
                                    unresolved=int((r.sta != 0).sum()),
                                    lost=0, bad_pack=0, canvas=0,
                                    canvas_cap=r.cap, final_bad=(r.sta != 0))

    if order is None:
        order = sorted(range(len(reads)),
                       key=lambda i: (reads[i].n_bad, reads[i].n_false))
    canvas = order[0]
    if prefer is not None:
        keep = [i for i in order if reads[i].cap == prefer]
        if keep:
            i = keep[0]
            best = reads[canvas].n_bad
            if reads[i].n_bad <= best * (1.0 + hysteresis) + 8:
                canvas = i
                order = [i] + [j for j in order if j != i]
    out = reads[canvas].frame.copy()
    ids = prof.video[:, None] + np.arange(3)[None, :]
    canvas_ids = reads[canvas].frame[ids]

    qual = [r.quality() for r in reads]
    full = [r.seg_full() for r in reads]
    final_bad = reads[canvas].sta != 0

    kept = copied = repacked = unresolved = lost = bad_pack = 0
    for s in range(prof.n_seg):
        if full[canvas][s]:
            kept += 1
            continue
        src = next((i for i in order if full[i][s]), None)
        off = int(prof.seg[s, 0])
        if src is not None:
            out[off:off + 400] = reads[src].frame[off:off + 400]
            # el ID DIF es de la captura, no del contenido: se repone
            for m in range(5):
                o = int(prof.seg[s, m])
                out[o:o + 3] = canvas_ids[s * 5 + m]
            final_bad[s * 5:(s + 1) * 5] = False
            copied += 1
            continue

        # hay que mezclar macrobloques: se parsea cada lectura
        best = [max(range(len(reads)), key=lambda i: (qual[i][s * 5 + m], -i))
                for m in range(5)]
        if all(qual[i][s * 5 + m] == SRC_NONE for i, m in zip(best, range(5))):
            unresolved += 5
            repacked += 0
            continue
        segs = {}
        for i in set(best):
            if qual[i][s * 5 + 0] or True:
                segs[i] = native.parse(bytes(reads[i].frame[off:off + 400]))
        base = native.DvSeg()
        ctypes.memmove(ctypes.byref(base), ctypes.byref(segs[best[0]]),
                       ctypes.sizeof(native.DvSeg))
        nres = 0
        for m in range(5):
            i = best[m]
            if qual[i][s * 5 + m] == SRC_NONE:
                nres += 1
                src_i = canvas if canvas in segs else best[0]
                base.mb[m] = segs[src_i].mb[m]
                base.mb[m].sta = 0x0E
                final_bad[s * 5 + m] = True
                sanitize_mb(base.mb[m], bool(segs[src_i].ok))
                continue
            base.mb[m] = segs[i].mb[m]
            base.mb[m].sta = 0
            final_bad[s * 5 + m] = False
            sanitize_mb(base.mb[m], bool(segs[i].ok))
        for m in range(5):
            o = int(prof.seg[s, m])
            base.mb[m].id[0] = int(canvas_ids[s * 5 + m][0])
            base.mb[m].id[1] = int(canvas_ids[s * 5 + m][1])
            base.mb[m].id[2] = int(canvas_ids[s * 5 + m][2])
        keep = tuple(m for m in range(5)
                     if qual[best[m]][s * 5 + m] != SRC_NONE
                     and best[m] == canvas)
        lost += max(0, fit_segment(base, protect=keep))
        buf, nlost = native.pack(base, bytes(reads[canvas].frame[off:off + 400]))
        out[off:off + 400] = np.frombuffer(buf, np.uint8)
        repacked += 1
        unresolved += nres
        if nlost:
            bad_pack += 1

    return out, dict(kept=kept, copied=copied, repacked=repacked,
                     unresolved=unresolved, lost=lost, bad_pack=bad_pack,
                     canvas=canvas, canvas_cap=reads[canvas].cap,
                     final_bad=final_bad)
