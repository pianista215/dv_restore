"""Mapas visuales del estado de un frame."""

import numpy as np

from . import dcplane, native, shuffle

# Estados de un macrobloque.
#
# La distincion entre FALSE_OK y SUSPECT importa: un segmento cuyo bitstream no
# decodifica y que ademas tiene algun macrobloque ya marcado con error no
# demuestra nada sobre los otros cuatro, porque la basura del marcado se cuela
# por el desbordamiento VLC. Solo cuando los CINCO se declaran sanos y aun asi
# el bitstream es invalido queda probado que la camara miente.
OK = 0          # sano y su segmento decodifica
BAD = 1         # la camara lo marca con error
FALSE_OK = 2    # los 5 del segmento se dan por sanos y el bitstream es invalido
SUSPECT = 3     # sano, pero comparte segmento con uno marcado malo
MISSING = 4     # no hay dato utilizable en ninguna captura

COLORS = {
    OK: None,                     # sin pintar
    BAD: (220, 40, 40),           # rojo
    FALSE_OK: (255, 160, 0),      # naranja
    SUSPECT: (60, 130, 230),      # azul
    MISSING: (140, 0, 180),       # morado
}


def block_state(frame, prof, scan=None):
    """Estado de cada uno de los 1620 bloques del flujo."""
    st = dcplane.sta(frame, prof)
    state = np.where(st != 0, BAD, OK).astype(np.uint8)
    if scan is None:
        scan = native.scan_frame(frame, prof)
    seg_bad = (st != 0).reshape(prof.n_seg, 5).any(axis=1)
    invalid = np.repeat(scan.seg_ok == 0, 5)
    state[(state == OK) & np.repeat(seg_bad, 5)] = SUSPECT
    state[(state == OK) & invalid] = FALSE_OK
    return state


def to_grid(state, prof, table=None):
    """Coloca el estado de cada bloque en su sitio de la pantalla."""
    if table is None:
        table = shuffle.load(prof)
    g = np.zeros(prof.mb_rows * prof.mb_cols, np.uint8)
    g[table] = state
    return g.reshape(prof.mb_rows, prof.mb_cols)


def overlay(img, grid, prof, alpha=0.55, edge=True):
    """Pinta la rejilla de estados encima de la imagen decodificada."""
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    out = img.astype(np.float32).copy()
    for st, col in COLORS.items():
        if col is None:
            continue
        m = np.repeat(np.repeat(grid == st, 16, axis=0), 16, axis=1)
        if not m.any():
            continue
        for c in range(3):
            out[:, :, c][m] = out[:, :, c][m] * (1 - alpha) + col[c] * alpha
        if edge:
            b = m & ~_shrink(m)
            for c in range(3):
                out[:, :, c][b] = col[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def _shrink(m):
    s = m.copy()
    s[1:, :] &= m[:-1, :]
    s[:-1, :] &= m[1:, :]
    s[:, 1:] &= m[:, :-1]
    s[:, :-1] &= m[:, 1:]
    return s


def legend_text(state):
    n = len(state)
    return (f"sanos {int((state == OK).sum())}  "
            f"error {int((state == BAD).sum())}  "
            f"falsos sanos {int((state == FALSE_OK).sum())}  "
            f"sospechosos {int((state == SUSPECT).sum())}  de {n}")
