"""A que macrobloque de la pantalla corresponde cada bloque del flujo.

DV baraja los macrobloques: los 5 de un segmento de video estan repartidos por
sitios lejanos del cuadro, para que un arayazo de la cinta salga como puntos
sueltos en vez de como una banda. Esa tabla de barajado es fija, pero en vez de
confiarla a una tabla escrita de memoria se DEDUCE del propio decodificador:

  1. Se fabrica un frame gris uniforme (todos los DC a un valor medio y ningun
     coeficiente AC) usando el repaquetizador.
  2. Se pone en blanco el macrobloque de los indices que tienen cierto bit a 1,
     se decodifica con ffmpeg y se mira que cuadros de 16x16 han cambiado.
  3. Con 11 planos de bits queda determinado el indice de cada posicion.

Asi la tabla es verificable y no depende de que yo recuerde bien la norma.
"""

import os
import numpy as np

from .layout import PAL
from . import native, render

_DATA = os.path.join(os.path.dirname(__file__), "data")

FLAT_DC = 0          # gris medio
MARK_DC = 255        # blanco saturado


def _flat_frame(frame, prof):
    """Aplana un frame real: mismo DC en todo, sin coeficientes AC."""
    from .bitstream import EOB
    out = bytearray(bytes(frame))
    for s in range(prof.n_seg):
        off = int(prof.seg[s, 0])
        seg = native.parse(bytes(frame[off:off + 400]))
        for m in range(5):
            seg.mb[m].sta = 0
            for j in range(6):
                b = seg.mb[m].b[j]
                b.dc = FLAT_DC
                b.mode = 0
                b.cls = 1
                b.ntok = 1
                b.tok[0] = EOB
        buf, _ = native.pack(seg, bytes(frame[off:off + 400]))
        out[off:off + 400] = buf
    return bytes(out)


def _mark(flat, prof, block_idx):
    """Pone en blanco la luma de los macrobloques indicados."""
    from .bitstream import EOB
    sel = np.zeros(prof.n_video, bool)
    sel[block_idx] = True
    out = bytearray(flat)
    for s in range(prof.n_seg):
        if not sel[s * 5:(s + 1) * 5].any():
            continue
        off = int(prof.seg[s, 0])
        seg = native.parse(bytes(flat[off:off + 400]))
        for m in range(5):
            if not sel[s * 5 + m]:
                continue
            for j in range(4):                 # solo los cuatro bloques de luma
                seg.mb[m].b[j].dc = MARK_DC
        buf, _ = native.pack(seg, bytes(flat[off:off + 400]))
        out[off:off + 400] = buf
    return bytes(out)


def _changed_mbs(img, base, prof):
    """Que macrobloques de 16x16 difieren entre dos decodificaciones."""
    d = (img.astype(np.int16) - base.astype(np.int16)) != 0
    g = d.reshape(prof.mb_rows, 16, prof.mb_cols, 16).sum(axis=(1, 3))
    return g > 64          # al menos un cuarto del macrobloque ha cambiado


def calibrate(frame, prof=PAL, verbose=True):
    """Devuelve un array (n_video,) con el indice plano de macrobloque
    (fila * mb_cols + columna) de cada bloque del flujo."""
    flat = _flat_frame(frame, prof)
    base = render.decode(np.frombuffer(flat, np.uint8), prof, "y")[0]
    if base.std() > 2:
        raise RuntimeError("el frame aplanado no ha salido uniforme")

    nbits = int(np.ceil(np.log2(prof.n_video)))
    code = np.zeros((prof.mb_rows, prof.mb_cols), np.int32)
    seen = np.zeros((prof.mb_rows, prof.mb_cols), bool)
    idx = np.arange(prof.n_video)
    for b in range(nbits):
        sel = idx[(idx >> b) & 1 == 1]
        img = render.decode(np.frombuffer(_mark(flat, prof, sel), np.uint8), prof, "y")[0]
        ch = _changed_mbs(img, base, prof)
        code |= ch.astype(np.int32) << b
        seen |= ch
        if verbose:
            print(f"  plano {b:2d}: {int(ch.sum()):5d} macrobloques cambian "
                  f"(esperados {len(sel)})")

    # el indice 0 nunca enciende ningun bit: es la unica posicion sin marcar
    if verbose:
        print(f"  posiciones que nunca cambian: {int((~seen).sum())} "
              f"(debe ser 1, la del bloque 0)")
    table = np.full(prof.n_video, -1, np.int32)
    flat_code = code.ravel()
    for pos in range(prof.mb_rows * prof.mb_cols):
        table[flat_code[pos]] = pos
    return table, code


def load(prof=PAL):
    path = os.path.join(_DATA, f"shuffle_{prof.name.lower()}.npy")
    if not os.path.exists(path):
        raise RuntimeError(
            f"falta {path}: ejecuta primero la calibracion (dvr.py calib)")
    return np.load(path)


def save(table, prof=PAL):
    os.makedirs(_DATA, exist_ok=True)
    np.save(os.path.join(_DATA, f"shuffle_{prof.name.lower()}.npy"), table)
