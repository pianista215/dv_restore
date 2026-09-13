"""Decodificar DV a imagen con ffmpeg, y volcar PNG."""

import subprocess
import numpy as np

from .layout import profile_for


def decode(frame_bytes, prof, planes="yuv"):
    """Decodifica uno o varios frames DV a numpy.

    Devuelve (n, h, w) uint8 con la luma si planes='y', o
    (n, h, w, 3) RGB si planes='rgb'.
    """
    data = np.ascontiguousarray(frame_bytes, dtype=np.uint8).tobytes()
    n = len(data) // prof.frame_size
    if planes == "rgb":
        pix, depth = "rgb24", 3
    else:
        pix, depth = "gray", 1
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "dv", "-i", "pipe:0",
         "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    need = n * prof.height * prof.width * depth
    out = p.stdout
    if len(out) < need:
        raise RuntimeError(
            f"ffmpeg devolvio {len(out)} bytes, esperaba {need}: "
            f"{p.stderr.decode(errors='replace')[:300]}")
    a = np.frombuffer(out[:need], dtype=np.uint8)
    if depth == 1:
        return a.reshape(n, prof.height, prof.width)
    return a.reshape(n, prof.height, prof.width, 3)


def save_png(img, path):
    from PIL import Image
    Image.fromarray(img).save(path)
