"""Acceso a ficheros DV crudos."""

import os
import numpy as np

from .layout import profile_for


class Capture:
    """Un fichero DV crudo, visto como (n_frames, frame_size)."""

    def __init__(self, path, prof=None):
        self.path = path
        self.name = os.path.basename(path)
        self.prof = prof or profile_for(path)
        size = os.path.getsize(path)
        self.n = size // self.prof.frame_size
        if self.n == 0:
            raise ValueError(f"{path}: no llega a un frame completo")
        self.data = np.memmap(path, dtype=np.uint8, mode="r",
                              shape=(self.n, self.prof.frame_size))

    def frame(self, i):
        return np.array(self.data[i])

    def sta(self, i):
        return self.data[i][self.prof.sta] >> 4

    def n_bad(self, i):
        return int((self.sta(i) != 0).sum())

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"<Capture {self.name} {self.n} frames {self.prof.name}>"


def write_frames(path, chunks):
    """Escribe una secuencia de frames (bytes o arrays) a un fichero DV."""
    with open(path, "wb") as fh:
        for c in chunks:
            fh.write(c.tobytes() if isinstance(c, np.ndarray) else c)


def clip(src, dst, start, count):
    """Corta un tramo por bytes, sin recodificar: el resultado es DV identico."""
    cap = src if isinstance(src, Capture) else Capture(src)
    start = max(0, min(start, cap.n))
    count = max(0, min(count, cap.n - start))
    with open(dst, "wb") as fh:
        for i in range(start, start + count):
            fh.write(cap.data[i].tobytes())
    return count
