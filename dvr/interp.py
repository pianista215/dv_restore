"""Interpolacion temporal compensada en movimiento, en el dominio de pixeles.

Copiar un macrobloque de un frame vecino solo sirve si la escena esta quieta:
si la camara panea, lo copiado pertenece a otro sitio. Aqui se hace lo que pide
la fisica del problema: se cogen la version BUENA de antes y la de DESPUES, se
mide cuanto se ha movido el contenido entre las dos, y se compone el instante
intermedio. El resultado son pixeles nuevos, asi que hay que codificarlos.

El movimiento se estima con los pixeles que rodean al bloque, no con el bloque
en si: en el frame de destino esta roto, que es justo el motivo de estar aqui.
"""

import numpy as np

from . import render
from .layout import PAL


class LumaCache:
    """Planos de luma decodificados, con memoria acotada."""

    def __init__(self, frames, prof, limit=64):
        self.frames = frames
        self.prof = prof
        self.limit = limit
        self._c = {}
        self._order = []

    def __call__(self, i):
        if i in self._c:
            return self._c[i]
        y = render.decode(np.asarray(self.frames[i]), self.prof, "yraw")[0]
        self._c[i] = y.astype(np.float32)
        self._order.append(i)
        if len(self._order) > self.limit:
            del self._c[self._order.pop(0)]
        return self._c[i]


def estimate_motion(a, b, y0, x0, size=16, ctx=16, rng=12):
    """Cuanto se ha movido el contenido de (y0,x0) entre los frames a y b.

    Convenio: a(y, x) ~= b(y + dy, x + dx). Es decir, lo que en 'a' esta en
    (y, x) aparece en 'b' desplazado (dy, dx).

    OJO AL SIGNO. Para rellenar la posicion (y0, x0) de un frame intermedio hay
    que RESTAR la fraccion del movimiento, no sumarla: el contenido que ahora
    esta ahi venia de mas atras. Equivocarse de signo no da un resultado
    mediocre, da uno peor que no compensar (error mediano 5,68 frente a 3,97 de
    copiar sin mas; con el signo bueno, 2,41).

    Se compara un recuadro con contexto alrededor del bloque, porque el bloque
    en si puede no ser representativo. Devuelve (dy, dx) en pixeles enteros.
    """
    h, w = a.shape
    Y0, X0 = y0 - ctx, x0 - ctx
    S = size + 2 * ctx
    if Y0 - rng < 0 or X0 - rng < 0 or Y0 + S + rng > h or X0 + S + rng > w:
        return 0, 0
    # Barrido grueso submuestreando de dos en dos (cuatro veces menos datos) y
    # afinado a paso 1 sobre el mejor, ya con todos los pixeles. Hacerlo de una
    # vez con sliding_window_view sale PEOR: materializa un array enorme.
    ref2 = a[Y0:Y0 + S:2, X0:X0 + S:2]
    best = (None, 0, 0)
    for dy in range(-rng, rng + 1, 2):
        for dx in range(-rng, rng + 1, 2):
            e = np.abs(ref2 - b[Y0 + dy:Y0 + dy + S:2, X0 + dx:X0 + dx + S:2]).sum()
            if best[0] is None or e < best[0]:
                best = (e, dy, dx)
    ref = a[Y0:Y0 + S, X0:X0 + S]
    by, bx = best[1], best[2]
    best = (None, by, bx)
    # el grueso va de dos en dos, asi que el afinado cubre +-2 para alcanzar
    # los impares aunque haya caido en el par de al lado
    for dy in range(by - 2, by + 3):
        for dx in range(bx - 2, bx + 3):
            if abs(dy) > rng or abs(dx) > rng:
                continue
            e = np.abs(ref - b[Y0 + dy:Y0 + dy + S, X0 + dx:X0 + dx + S]).sum()
            if best[0] is None or e < best[0]:
                best = (e, dy, dx)
    return best[1], best[2]


def sample(img, y0, x0, oy, ox, size=16):
    """Recorta size x size en (y0+oy, x0+ox) con muestreo bilineal."""
    fy, fx = int(np.floor(oy)), int(np.floor(ox))
    ty, tx = oy - fy, ox - fx
    Y, X = y0 + fy, x0 + fx
    if Y < 0 or X < 0 or Y + size + 1 > img.shape[0] or X + size + 1 > img.shape[1]:
        return None
    a = img[Y:Y + size, X:X + size]
    b = img[Y:Y + size, X + 1:X + 1 + size]
    c = img[Y + 1:Y + 1 + size, X:X + size]
    d = img[Y + 1:Y + 1 + size, X + 1:X + 1 + size]
    return (a * (1 - ty) * (1 - tx) + b * (1 - ty) * tx
            + c * ty * (1 - tx) + d * ty * tx)


def interpolate_block(luma, i, j1, j2, y0, x0, size=16, rng=12, blend=False):
    """El bloque de (y0,x0) en el instante i, reconstruido desde j1 y j2.

    Por defecto se compensa el movimiento y se copia del mas cercano, sin
    mezclar: mezclar los dos emborrona. Devuelve (pixeles, discrepancia entre
    las dos fuentes) o (None, None) si no se puede.
    """
    a, b = luma(j1), luma(j2)
    dy, dx = estimate_motion(a, b, y0, x0, size, rng=rng)
    span = j2 - j1
    if span <= 0:
        return None, None
    t = (i - j1) / span
    s1 = sample(a, y0, x0, -dy * t, -dx * t, size)
    s2 = sample(b, y0, x0, dy * (1 - t), dx * (1 - t), size)
    if s1 is None or s2 is None:
        return None, None
    conf = float(np.abs(s1 - s2).mean())
    if blend:
        return (1 - t) * s1 + t * s2, conf
    return (s1 if (i - j1) <= (j2 - i) else s2), conf
