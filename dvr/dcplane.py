"""Extraccion del plano DC sin decodificar el bitstream.

El coeficiente DC de cada bloque DCT esta en una posicion FIJA: los 9 primeros
bits del area del bloque DCT dentro del macrobloque. No depende de las tablas
VLC ni del desbordamiento, asi que se puede leer de forma vectorizada para todo
el frame de golpe.

Eso da, por cada macrobloque, 4 valores de luminancia y 2 de croma: una
miniatura de 90x72 (luma) del frame, exacta y baratisima. Es la base de la
deteccion de bloques falsamente sanos y de la puerta de movimiento, y no
necesita nada del parser.
"""

import numpy as np

from .layout import AREA_OFF, DCT_BLOCKS


def dc_raw(frames, prof):
    """DC de los 6 bloques DCT de cada macrobloque.

    frames: (n, frame_size) uint8   ->   (n, n_video, 6) int16 con signo
    """
    frames = np.asarray(frames)
    single = frames.ndim == 1
    if single:
        frames = frames[None, :]
    off = prof.video[:, None] + np.array(AREA_OFF)[None, :]      # (n_video, 6)
    hi = frames[:, off].astype(np.int16)                         # (n, mb, 6)
    lo = frames[:, off + 1].astype(np.int16)
    dc = (hi << 1) | (lo >> 7)                                   # 9 bits
    dc = np.where(dc >= 256, dc - 512, dc)                       # signo
    return dc[0] if single else dc


def dct_mode(frames, prof):
    """Bit de modo DCT (8x8 frente a 2-4-8) de cada bloque DCT."""
    frames = np.asarray(frames)
    single = frames.ndim == 1
    if single:
        frames = frames[None, :]
    off = prof.video[:, None] + np.array(AREA_OFF)[None, :]
    m = (frames[:, off + 1] >> 6) & 1
    return m[0] if single else m


def dct_class(frames, prof):
    """Numero de clase (2 bits) de cada bloque DCT."""
    frames = np.asarray(frames)
    single = frames.ndim == 1
    if single:
        frames = frames[None, :]
    off = prof.video[:, None] + np.array(AREA_OFF)[None, :]
    c = (frames[:, off + 1] >> 4) & 3
    return c[0] if single else c


def sta(frames, prof):
    """Estado de error de cada macrobloque: 0 = la camara lo da por sano."""
    frames = np.asarray(frames)
    if frames.ndim == 1:
        return frames[prof.sta] >> 4
    return frames[:, prof.sta] >> 4


def qno(frames, prof):
    """Numero de cuantificacion de cada macrobloque."""
    frames = np.asarray(frames)
    if frames.ndim == 1:
        return frames[prof.sta] & 0x0F
    return frames[:, prof.sta] & 0x0F
