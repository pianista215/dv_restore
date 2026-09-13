"""Parser y repaquetizador de un segmento de video DV.

Un segmento son 5 bloques DIF (400 bytes) que llevan 5 macrobloques de 6
bloques DCT cada uno. Cada bloque DCT tiene un area FIJA dentro de su
macrobloque (112 bits los cuatro de luma, 80 bits los dos de croma) que empieza
con DC (9 bits), modo DCT (1 bit) y clase (2 bits); despues van los
coeficientes AC en VLC.

Lo que hace complicado tocar un solo bloque es el DESBORDAMIENTO: si los
coeficientes de un bloque DCT no caben en su area, siguen en el hueco que dejan
libre los demas bloques del mismo macrobloque, y si tampoco caben, en el hueco
que dejan los otros cuatro macrobloques del segmento. Por eso sustituir 80
bytes sueltos puede romper los otros cuatro macrobloques, y por eso hace falta
parsear y volver a serializar el segmento entero para mezclar macrobloques de
capturas distintas.

El reparto en tres pasadas (area -> deposito del macrobloque -> deposito del
segmento) esta implementado aqui igual que lo define la norma, y se comprueba
exigiendo que parse + pack devuelva los 400 bytes EXACTOS del original.
"""

import os
import numpy as np

from .layout import AREA_BITS, AREA_OFF, BLOCK, DCT_BLOCKS, HDR_BITS

_DATA = os.path.join(os.path.dirname(__file__), "data", "dvvlc.npz")

_t = np.load(_DATA)
VLC_CODE = _t["code"].astype(np.uint32)
VLC_LEN = _t["length"].astype(np.uint8)
VLC_RUN = _t["run"].astype(np.int16)
VLC_LEVEL = _t["level"].astype(np.int16)
NB_VLC = len(VLC_LEN)
MAX_LEN = int(VLC_LEN.max())            # 16 bits contando el de signo

# Indice de la palabra EOB: run 127, nivel 0 -> la posicion salta por encima
# de 63 y el bloque DCT termina.
EOB = int(np.nonzero((VLC_RUN == 127) & (VLC_LEVEL == 0))[0][0])

# Tabla de decodificacion directa: 16 bits -> indice de entrada.
# El codigo es prefijo completo (suma de Kraft = 1), asi que toda combinacion
# de 16 bits empieza por exactamente una palabra valida y no hay huecos.
_LUT = np.zeros(1 << MAX_LEN, dtype=np.uint16)
for _i in range(NB_VLC):
    _l = int(VLC_LEN[_i])
    _c = int(VLC_CODE[_i]) << (MAX_LEN - _l)
    _LUT[_c:_c + (1 << (MAX_LEN - _l))] = _i
del _i, _l, _c

SEG_BYTES = 5 * BLOCK
SEG_MB = 5


def _bits(buf):
    return np.unpackbits(np.frombuffer(buf, dtype=np.uint8))


def _peek_table(bits):
    """Valor de los 16 bits que empiezan en cada posicion, de una vez."""
    n = len(bits)
    pad = np.concatenate([bits, np.zeros(MAX_LEN, np.uint8)])
    w = np.lib.stride_tricks.sliding_window_view(pad, MAX_LEN)[:n]
    return w.astype(np.uint32) @ (1 << np.arange(MAX_LEN - 1, -1, -1)).astype(np.uint32)


class _Stream:
    """Lectura de bits sobre una secuencia logica que puede venir troceada."""

    __slots__ = ("bits", "peek", "i", "n")

    def __init__(self, bits):
        self.bits = bits
        self.n = len(bits)
        self.peek = _peek_table(bits) if self.n else np.zeros(0, np.uint32)
        self.i = 0

    def read(self, k):
        v = 0
        for b in self.bits[self.i:self.i + k]:
            v = (v << 1) | int(b)
        self.i += k
        return v

    def rest(self):
        return self.bits[self.i:]


class Block:
    """Un bloque DCT: cabecera, coeficientes y las palabras VLC que los dieron.

    Se guardan los indices de las palabras (no solo run/nivel) porque la
    correspondencia (run, nivel) -> palabra NO es unica en DV: para poder
    reconstruir el segmento byte a byte hay que reemitir exactamente la misma.
    """

    __slots__ = ("dc", "mode", "cls", "tokens", "pos", "done")

    def __init__(self, dc=0, mode=0, cls=0):
        self.dc = dc
        self.mode = mode
        self.cls = cls
        self.tokens = []        # indices en la tabla VLC, EOB incluido
        self.pos = 0
        self.done = False

    def coeffs(self):
        """Coeficientes cuantificados en orden zigzag, 64 valores."""
        out = np.zeros(64, dtype=np.int16)
        out[0] = self.dc
        pos = 0
        for t in self.tokens:
            pos += int(VLC_RUN[t]) + 1
            if pos >= 64:
                break
            out[pos] = VLC_LEVEL[t]
        return out

    def nnz(self):
        """Coeficientes AC no nulos."""
        return sum(1 for t in self.tokens if VLC_LEVEL[t] != 0)

    def bit_len(self):
        return HDR_BITS + int(VLC_LEN[self.tokens].sum()) if self.tokens else HDR_BITS


class Segment:
    __slots__ = ("mb", "qno", "sta", "dif_id", "raw", "damaged", "reason")

    def __init__(self):
        self.mb = [[Block() for _ in range(DCT_BLOCKS)] for _ in range(SEG_MB)]
        self.qno = [0] * SEG_MB
        self.sta = [0] * SEG_MB
        # los 3 bytes de ID DIF de cada bloque: no son contenido, pero hay que
        # devolverlos tal cual para que el segmento siga siendo valido
        self.dif_id = [b"\x00\x00\x00"] * SEG_MB
        # bytes originales: el hueco que sobra tras el ultimo
        # coeficiente lleva relleno del codificador de la camara,
        # que no es cero y hay que devolver tal cual
        self.raw = None
        self.damaged = False
        self.reason = ""


class ParseError(Exception):
    pass


def _decode_into(st, blk):
    """Consume palabras VLC de st hasta que el bloque termine o se acaben."""
    while True:
        if st.i >= st.n:
            return False
        t = int(_LUT[st.peek[st.i]])
        l = int(VLC_LEN[t])
        if st.i + l > st.n:
            # la palabra queda a caballo: se continua en el siguiente deposito
            return False
        st.i += l
        blk.tokens.append(t)
        blk.pos += int(VLC_RUN[t]) + 1
        if blk.pos >= 64:
            blk.done = True
            return True


def parse_segment(buf):
    """400 bytes -> Segment. Lanza ParseError si el bitstream no es valido."""
    if len(buf) != SEG_BYTES:
        raise ValueError("un segmento son 400 bytes")
    seg = Segment()
    seg.raw = bytes(buf)
    bits = _bits(buf)

    # --- pasada 1: el area fija de cada bloque DCT ---
    mb_free = [[] for _ in range(SEG_MB)]      # bits sobrantes -> deposito MB
    pend = [[None] * DCT_BLOCKS for _ in range(SEG_MB)]
    for m in range(SEG_MB):
        base = m * BLOCK * 8
        seg.dif_id[m] = bytes(buf[m * BLOCK:m * BLOCK + 3])
        seg.sta[m] = int(buf[m * BLOCK + 3]) >> 4
        seg.qno[m] = int(buf[m * BLOCK + 3]) & 0x0F
        for j in range(DCT_BLOCKS):
            a = base + AREA_OFF[j] * 8
            st = _Stream(bits[a:a + AREA_BITS[j]])
            blk = seg.mb[m][j]
            blk.dc = st.read(9)
            if blk.dc >= 256:
                blk.dc -= 512
            blk.mode = st.read(1)
            blk.cls = st.read(2)
            _decode_into(st, blk)
            if blk.done:
                mb_free[m].append(st.rest())    # sobra area: al deposito
            else:
                pend[m][j] = st.rest()          # bits a medias de una palabra

    # --- pasada 2: deposito del macrobloque ---
    vs_free = []
    for m in range(SEG_MB):
        pool = np.concatenate(mb_free[m]) if mb_free[m] else np.zeros(0, np.uint8)
        all_done = True
        for j in range(DCT_BLOCKS):
            blk = seg.mb[m][j]
            if blk.done:
                continue
            head = pend[m][j] if pend[m][j] is not None else np.zeros(0, np.uint8)
            st = _Stream(np.concatenate([head, pool]))
            _decode_into(st, blk)
            if not blk.done:
                pend[m][j] = st.rest()
                all_done = False
                break
            pend[m][j] = None
            pool = st.rest()
        if all_done:
            vs_free.append(pool)

    # --- pasada 3: deposito del segmento ---
    pool = np.concatenate(vs_free) if vs_free else np.zeros(0, np.uint8)
    for m in range(SEG_MB):
        for j in range(DCT_BLOCKS):
            blk = seg.mb[m][j]
            if blk.done:
                continue
            head = pend[m][j] if pend[m][j] is not None else np.zeros(0, np.uint8)
            st = _Stream(np.concatenate([head, pool]))
            _decode_into(st, blk)
            if not blk.done:
                seg.damaged = True
                seg.reason = f"mb{m} blq{j} no termina: pos={blk.pos}"
                return seg
            pend[m][j] = None
            pool = st.rest()
    return seg


class _Writer:
    """Escritura de bits con capacidad limitada."""

    __slots__ = ("bits", "i", "n")

    def __init__(self, n):
        self.bits = np.zeros(n, dtype=np.uint8)
        self.i = 0
        self.n = n

    def put(self, value, k):
        room = min(k, self.n - self.i)
        for b in range(room):
            self.bits[self.i + b] = (value >> (k - 1 - b)) & 1
        self.i += room
        return k - room        # bits que no han cabido

    def free(self):
        return self.n - self.i


def pack_segment(seg, pad_from=None):
    """Segment -> 400 bytes, repitiendo el reparto en tres pasadas.

    Devuelve (buf, perdidos): perdidos son las palabras VLC que no han cabido y
    ha habido que descartar (las de frecuencia mas alta). Con un segmento
    recien salido de parse_segment siempre es 0; solo puede pasar al mezclar
    macrobloques de origenes distintos cuyos coeficientes no caben juntos.
    """
    streams = []
    for m in range(SEG_MB):
        row = []
        for j in range(DCT_BLOCKS):
            blk = seg.mb[m][j]
            head = ((blk.dc & 0x1FF) << 3) | ((blk.mode & 1) << 2) | (blk.cls & 3)
            row.append([(head, HDR_BITS)] + [(int(VLC_CODE[t]), int(VLC_LEN[t]))
                                             for t in blk.tokens])
        streams.append(row)

    out = np.zeros(SEG_BYTES * 8, dtype=np.uint8)
    touched = np.zeros(SEG_BYTES * 8, dtype=np.uint8)
    lost = 0

    # --- pasada 1: area fija de cada bloque DCT ---
    # Si todo cabe, el hueco que sobra pasa al deposito del macrobloque.
    # Si no cabe, la cola del area es la PRIMERA continuacion del propio
    # bloque (ahi va partida la palabra que desborda), no parte del deposito.
    holes = [[] for _ in range(SEG_MB)]
    left = [[None] * DCT_BLOCKS for _ in range(SEG_MB)]
    own = [[None] * DCT_BLOCKS for _ in range(SEG_MB)]
    for m in range(SEG_MB):
        base = m * BLOCK * 8
        for j in range(DCT_BLOCKS):
            a = base + AREA_OFF[j] * 8
            cap = AREA_BITS[j]
            toks = streams[m][j]
            used = k = 0
            while k < len(toks) and used + toks[k][1] <= cap:
                _put(out, a + used, toks[k][0], toks[k][1], touched)
                used += toks[k][1]
                k += 1
            if k == len(toks):
                if cap - used:
                    holes[m].append((a + used, cap - used))
            else:
                left[m][j] = toks[k:]
                if cap - used:
                    own[m][j] = (a + used, cap - used)

    # --- pasada 2: deposito del macrobloque ---
    vs_holes = []
    for m in range(SEG_MB):
        pool = list(holes[m])
        all_done = True
        for j in range(DCT_BLOCKS):
            if left[m][j] is None:
                continue
            mine = ([own[m][j]] if own[m][j] else []) + pool
            own[m][j] = None
            pool, rest = _drain(out, mine, left[m][j], touched)
            if rest:
                left[m][j] = rest
                all_done = False
                break
            left[m][j] = None
        if all_done:
            vs_holes += pool

    # --- pasada 3: deposito del segmento ---
    pool = vs_holes
    for m in range(SEG_MB):
        for j in range(DCT_BLOCKS):
            if left[m][j] is None:
                continue
            mine = ([own[m][j]] if own[m][j] else []) + pool
            own[m][j] = None
            pool, rest = _drain(out, mine, left[m][j], touched)
            if rest:
                lost += len(rest)
            left[m][j] = None

    src = pad_from if pad_from is not None else seg.raw
    if src is not None:
        keep = np.unpackbits(np.frombuffer(src, dtype=np.uint8))
        out = np.where(touched.astype(bool), out, keep)
    raw = bytearray(np.packbits(out).tobytes())
    for m in range(SEG_MB):
        raw[m * BLOCK:m * BLOCK + 3] = seg.dif_id[m]
        raw[m * BLOCK + 3] = ((seg.sta[m] & 0x0F) << 4) | (seg.qno[m] & 0x0F)
    return bytes(raw), lost


def _put(out, at, value, k, touched=None):
    for b in range(k):
        out[at + b] = (value >> (k - 1 - b)) & 1
    if touched is not None:
        touched[at:at + k] = 1


def _drain(out, holes, toks, touched=None):
    """Escribe toks en la lista de huecos; devuelve (huecos restantes, sobras).

    Una palabra puede quedar partida entre dos huecos, e incluso entre dos
    depositos distintos: es justo lo que hace el formato. Si se agotan los
    huecos a mitad de una palabra, lo que se devuelve como sobra es el TROZO
    QUE FALTA, no la palabra entera, porque los bits ya escritos se quedan
    donde estan y el siguiente deposito continua desde ahi.
    """
    hi = 0
    ti = 0
    while ti < len(toks):
        v, l = toks[ti]
        need = l
        while need and hi < len(holes):
            at, n = holes[hi]
            take = min(need, n)
            _put(out, at, (v >> (need - take)) & ((1 << take) - 1), take, touched)
            need -= take
            if take == n:
                hi += 1
            else:
                holes[hi] = (at + take, n - take)
        if need:
            return [], [(v & ((1 << need) - 1), need)] + list(toks[ti + 1:])
        ti += 1
    return holes[hi:], []
