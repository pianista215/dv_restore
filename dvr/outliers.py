"""Descartar frames que no encajan donde han quedado colocados.

En los tramos muy dañados aparecen frames que el deck suelta al patinar: no
corresponden a una posicion de cinta distinta, pero llevan datos suficientes
para no ser relleno puro y no casan con nada, asi que quedan como grupo de una
sola lectura y se cuelan entre frames buenos. El efecto es un fogonazo: dos
frames casi iguales con uno completamente distinto en medio.

La prueba es local y no necesita saber de donde salio el frame: si quitarlo
ACERCA a sus vecinos a menos de la mitad de lo que distaban de el, es que no va
ahi. Un frame intermedio de verdad, aunque la camara este barriendo rapido,
deja a sus vecinos al DOBLE de distancia, no a la mitad.

Solo se aplica a frames debilmente sostenidos y con bastante imagen inventada:
nunca se tira algo que traiga datos reales de varias pasadas.
"""

import numpy as np

from . import dcplane


def find_misplaced(frames, reads, prof, max_reads=2, ratio=0.5, min_move=20.0,
                   min_bad=0.10):
    """Devuelve el indice de los frames que sobran.

    frames: lista de frames ya fundidos.
    reads:  cuantas lecturas sostiene cada uno.
    """
    n = len(frames)
    if n < 3:
        return []
    dc = np.stack([dcplane.dc_raw(np.asarray(f), prof)[:, :4].mean(axis=1)
                   for f in frames])
    bad = np.array([int((dcplane.sta(np.asarray(f), prof) != 0).sum())
                    for f in frames])
    lim = min_bad * prof.n_video

    def dist(a, b):
        return float(np.abs(dc[a] - dc[b]).mean())

    out = []
    for k in range(1, n - 1):
        if reads[k] > max_reads or bad[k] < lim:
            continue
        dp, dn = dist(k - 1, k), dist(k, k + 1)
        if min(dp, dn) <= min_move:
            continue
        if dist(k - 1, k + 1) < ratio * min(dp, dn):
            out.append(k)
    return out
