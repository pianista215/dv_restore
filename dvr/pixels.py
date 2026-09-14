"""Decodificar un macrobloque DV a pixeles, y volver.

Hace falta para poder INTERPOLAR: copiar bloques enteros solo permite pegar
contenido que ya existe en otro frame, y si la escena se mueve eso nunca cae
donde debe. Interpolar entre la version buena de antes y la de despues si da el
contenido del instante intermedio, pero crea pixeles nuevos, y para meterlos en
el flujo hay que codificarlos.

La puerta de entrada a todo esto es reproducir EXACTAMENTE lo que hace el
decodificador: si nuestros pixeles no coinciden con los de ffmpeg, cualquier
cosa que codifiquemos saldra mal. Por eso este modulo se verifica contra ffmpeg
antes de usarse para nada.
"""

import json
import os

import numpy as np

from .bitstream import VLC_LEVEL, VLC_RUN
from .layout import DCT_BLOCKS

_Q = json.load(open(os.path.join(os.path.dirname(__file__), "data", "quant.json")))
QUANT_SHIFTS = np.array(_Q["shifts"], np.int32)      # (22, 4)
QUANT_OFFSET = np.array(_Q["offset"], np.int32)      # (4,)
IWEIGHT_88 = np.array(_Q["iw88"], np.int64)
IWEIGHT_248 = np.array(_Q["iw248"], np.int64)
IWEIGHT_BITS = 14
QUANT_AREAS = (6, 21, 43, 64)

# zigzag estandar: posicion en el recorrido -> indice en el bloque 8x8
ZIGZAG = np.array([
    0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63], np.int32)

# El recorrido 2-4-8, para el modo DCT de campo
ZIGZAG248 = np.array([
    0,  8,  1,  9, 16, 24,  2, 10, 17, 25, 32, 40, 48, 56, 33, 41,
    18, 26,  3, 11, 19, 27, 34, 42, 49, 57, 50, 58, 35, 43,  4, 12,
    20, 28,  5, 13, 21, 29, 36, 44, 51, 59, 52, 60, 37, 45,  6, 14,
    22, 30,  7, 15, 23, 31, 38, 46, 53, 61, 54, 62, 39, 47, 55, 63], np.int32)

# area de cuantificacion de cada posicion del recorrido
_AREA = np.zeros(64, np.int32)
_i = 0
for _c, _end in enumerate(QUANT_AREAS):
    _AREA[_i:_end] = _c
    _i = _end
del _i, _c, _end


_FT = {}


def factor_table(qno, cls, mode):
    """Factor de desescalado de cada posicion del recorrido.

    Cacheado: solo hay 16x4x2 combinaciones y se pedia una por bloque y clase,
    que era una parte notable del coste de codificar.
    """
    key = (qno, cls, mode)
    f = _FT.get(key)
    if f is None:
        f = _FT[key] = _factor_table(qno, cls, mode)
    return f


def _factor_table(qno, cls, mode):
    s = qno + int(QUANT_OFFSET[cls])
    s = min(max(s, 0), 21)
    iw = IWEIGHT_248 if mode else IWEIGHT_88
    f = iw << (QUANT_SHIFTS[s, _AREA] + 1)
    if cls == 3:
        f = f << 1
    return f


def _idct_matrix(n):
    k = np.arange(n)
    m = np.cos((2 * k[:, None] + 1) * k[None, :] * np.pi / (2 * n))
    m[:, 0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2.0 / n)


_M = _idct_matrix(8)
_M4 = _idct_matrix(4)


def block_pixels(dc, mode, cls, qno, tokens):
    """Un bloque DCT -> 8x8 de pixeles (float, sin recortar)."""
    coef = np.zeros(64, np.int64)
    f = factor_table(qno, cls, mode)
    pos = 0
    for t in tokens:
        pos += int(VLC_RUN[t]) + 1
        if pos >= 64:
            break
        lv = int(VLC_LEVEL[t])
        if lv:
            coef[pos] = (lv * int(f[pos]) + (1 << (IWEIGHT_BITS - 1))) >> IWEIGHT_BITS
    scan = ZIGZAG248 if mode else ZIGZAG
    blk = np.zeros(64, np.float64)
    blk[scan] = coef
    # el DC no pasa por la tabla: se escala por 4 y se le suma el offset
    blk[0] = dc * 4 + 1024
    b = blk.reshape(8, 8)
    if mode:
        # Modo 2-4-8 (DCT de campo). No esta implementado: lleva una mariposa
        # de 2 puntos entre filas, IDCT de 8 en horizontal y de 4 en vertical
        # por campo, y reproducirlo exacto se resistio (error medio 12,2 frente
        # a 0,4 del modo normal). Es el 13,7% de los bloques.
        #
        # No hace falta: para INTERPOLAR codificamos siempre en modo 8x8, que
        # lo elegimos nosotros, y como origen basta con saltarse estos bloques.
        # Devolver None obliga a quien llame a tratarlo, en vez de colar
        # pixeles equivocados sin avisar.
        return None
    return _M @ b @ _M.T


QUAD = ((0, 0), (0, 8), (8, 0), (8, 8))    # Y0 Y1 Y2 Y3 dentro del 16x16


def mb_luma(seg, m):
    """Los 16x16 de luma de un macrobloque. None si algun bloque va en 2-4-8."""
    out = np.zeros((16, 16), np.float64)
    for j in range(4):
        b = seg.mb[m].b[j]
        px = block_pixels(b.dc, b.mode, b.cls, seg.mb[m].qno,
                          [int(b.tok[i]) for i in range(int(b.ntok))])
        if px is None:
            return None
        r, c = QUAD[j]
        out[r:r + 8, c:c + 8] = px
    return out
