"""Codificar pixeles a un macrobloque DV.

Es lo que permite INTERPOLAR: mezclar la version buena de antes con la de
despues y meter el resultado en el flujo. Copiar bloques enteros solo permite
pegar contenido que ya existe tal cual en otro frame.

Se codifica siempre en modo DCT 8x8: el modo 2-4-8 lo elige el codificador, y
nosotros elegimos el normal.

La tabla VLC no cubre todos los pares (recorrido, nivel) -- a partir de
recorrido 11 ya no hay ningun nivel representable -- asi que los recorridos
largos se parten: primero una palabra de nivel 0 que solo avanza, y luego el
valor. Es lo mismo que hace el codificador de la camara.
"""

import numpy as np

from .bitstream import EOB, VLC_LEN, VLC_LEVEL, VLC_RUN, NB_VLC
from .pixels import ZIGZAG, factor_table, _M, QUAD

# (recorrido, nivel) -> palabra mas corta que lo codifica
_CODE = {}
# avanzar n posiciones sin escribir nada -> palabra de nivel 0
_SKIP = {}
for _i in range(NB_VLC):
    _r, _l, _n = int(VLC_RUN[_i]), int(VLC_LEVEL[_i]), int(VLC_LEN[_i])
    if _l:
        _k = (_r, _l)
        if _k not in _CODE or _n < int(VLC_LEN[_CODE[_k]]):
            _CODE[_k] = _i
    elif _r < 64:
        _adv = _r + 1
        if _adv not in _SKIP or _n < int(VLC_LEN[_SKIP[_adv]]):
            _SKIP[_adv] = _i
del _i, _r, _l, _n

# nivel maximo representable para cada recorrido
_MAXLEV = {}
for (_r, _l) in _CODE:
    _MAXLEV[_r] = max(_MAXLEV.get(_r, 0), _l)
del _r, _l

MAX_SKIP = max(_SKIP) if _SKIP else 1


def forward_dct(px):
    """8x8 de pixeles -> coeficientes en el dominio que espera el decodificador."""
    return _M.T @ px @ _M


def _tokens_for(levels):
    """Niveles en orden de recorrido (1..63) -> palabras VLC."""
    out = []
    run = 0
    for pos in range(1, 64):
        lv = int(levels[pos])
        if lv == 0:
            run += 1
            continue
        sign = -1 if lv < 0 else 1
        mag = abs(lv)
        # partir el recorrido si es demasiado largo para la tabla
        while run > 0 and (run, min(mag, _MAXLEV.get(run, 0))) not in _CODE:
            adv = min(run, MAX_SKIP)
            while adv > 0 and adv not in _SKIP:
                adv -= 1
            if adv <= 0:
                break
            out.append(_SKIP[adv])
            run -= adv
        cap = _MAXLEV.get(run, 0)
        if cap == 0:
            run += 1
            continue
        mag = min(mag, cap)
        while (run, mag) not in _CODE and mag > 1:
            mag -= 1
        if (run, mag) not in _CODE:
            run += 1
            continue
        i = _CODE[(run, mag)]
        # la entrada positiva y la negativa van seguidas en la tabla
        out.append(i if sign > 0 else i + 1)
        run = 0
    out.append(EOB)
    return out


def _levels(z, qno, cls):
    f = factor_table(qno, cls, 0)
    out = np.zeros(64, np.int64)
    out[1:] = np.rint(z[1:] * (1 << 14) / f[1:]).astype(np.int64)
    return out


def _saturated(levels):
    """Cuantos coeficientes se salen de lo que la tabla puede representar.

    El unico limite de verdad es 255: con recorrido 0 la tabla cubre 1..255
    entera, y cualquier recorrido largo se parte con una palabra de nivel 0
    hasta llegar a recorrido 0. Contar como "no representable" todo lo que
    excedia el maximo de SU recorrido era un error, y encima hacia elegir
    clases demasiado finas, que son las que saturan de verdad.
    """
    return int((np.abs(levels) > 255).sum())


def _class_order(qno, cls):
    """La clase de partida, y detras las que dan niveles mas pequenos.

    Lo medido: conservar la clase del bloque original da 87,3% de bloques
    exactos en el round-trip, y forzar cualquier clase fija baja al 78% o
    menos. La camara eligio la que encaja con ese contenido, y reusarla
    reproduce su misma rejilla de cuantificacion. Solo se cambia si satura.
    """
    scale = []
    for c in range(4):
        f = factor_table(qno, c, 0)
        scale.append((-float(f[20]), c))     # factor grande = niveles pequenos
    rest = [c for _, c in sorted(scale) if c != cls]
    return (cls,) + tuple(rest)


def encode_block(px, qno, cls=0):
    """8x8 de pixeles -> (dc, clase, palabras VLC), en modo DCT 8x8.

    Se conserva la clase que se pasa (la del bloque de origen) y solo se cambia
    si satura: la clase fija la escala de cuantificacion, y una demasiado fina
    manda los coeficientes por encima de 255. Ver _class_order.
    """
    F = forward_dct(px)
    dc = int(np.clip(round((F[0, 0] - 1024) / 4.0), -256, 255))
    z = F.ravel()[ZIGZAG]
    order = _class_order(qno, cls)
    best = None
    for c in order:
        raw = _levels(z, qno, c)
        n = _saturated(raw)
        if best is None or n < best[0]:
            best = (n, c, np.clip(raw, -255, 255))
        if n == 0:
            break
    _, c, levels = best
    return dc, c, _tokens_for(levels)


def encode_mb(seg, m, luma, qno=None, cls=None):
    """Mete un 16x16 de luma en el macrobloque m de un segmento ya parseado.

    Solo toca los cuatro bloques de luma; el croma se deja como estaba, que es
    lo correcto cuando el macrobloque de origen ya trae su croma.
    """
    if qno is None:
        qno = int(seg.mb[m].qno)
    for j in range(4):
        b = seg.mb[m].b[j]
        c = int(b.cls) if cls is None else cls
        r, cc = QUAD[j]
        dc, c, toks = encode_block(luma[r:r + 8, cc:cc + 8], qno, c)
        b.dc = dc
        b.mode = 0
        b.cls = c
        for i, t in enumerate(toks[:70]):
            b.tok[i] = t
        b.ntok = min(len(toks), 70)
        b.ntok_area = b.ntok
        b.pos = 128
        b.done = 1
    seg.mb[m].qno = qno
