"""Decodificar DV a imagen con ffmpeg, y volcar PNG.

Con ec=0 se desactiva la ocultacion de errores del propio decodificador. Sirve
para ver el estado REAL del flujo: con la ocultacion puesta, ffmpeg disimula
gran parte del dano y las comparativas parecen todas iguales.
"""

import subprocess
import numpy as np

from .layout import profile_for


def decode(frame_bytes, prof, planes="yuv", ec=None):
    """Decodifica uno o varios frames DV a numpy.

    Devuelve (n, h, w) uint8 con la luma si planes='y', o
    (n, h, w, 3) RGB si planes='rgb'.
    """
    data = np.ascontiguousarray(frame_bytes, dtype=np.uint8).tobytes()
    n = len(data) // prof.frame_size
    if planes == "rgb":
        pix, depth = "rgb24", 3
    elif planes == "y":
        # OJO: pedir 'gray' hace que ffmpeg expanda el rango de estudio
        # (16-235) a completo (0-255). Para comparar con nuestros propios
        # pixeles hay que pedir el plano Y tal cual, con 'yuv420p'.
        pix, depth = "gray", 1
    else:
        pix, depth = "yuv420p", 0
    pre = [] if ec is None else ["-ec", str(ec)]
    p = subprocess.run(
        ["ffmpeg", "-v", "error"] + pre + ["-f", "dv", "-i", "pipe:0",
         "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if depth == 0:      # yuv420p: Y entero + dos cromas a la mitad
        need = n * prof.height * prof.width * 3 // 2
    else:
        need = n * prof.height * prof.width * depth
    out = p.stdout
    if len(out) < need:
        raise RuntimeError(
            f"ffmpeg devolvio {len(out)} bytes, esperaba {need}: "
            f"{p.stderr.decode(errors='replace')[:300]}")
    a = np.frombuffer(out[:need], dtype=np.uint8)
    if depth == 0:
        fs = prof.height * prof.width * 3 // 2
        return np.stack([a[i * fs:i * fs + prof.height * prof.width]
                         .reshape(prof.height, prof.width) for i in range(n)])
    if depth == 1:
        return a.reshape(n, prof.height, prof.width)
    return a.reshape(n, prof.height, prof.width, 3)


def save_png(img, path):
    from PIL import Image
    Image.fromarray(img).save(path)
