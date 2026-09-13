"""Dejar un segmento en estado emitible antes de volver a empaquetarlo.

Al mezclar macrobloques de origenes distintos hay que asegurarse de que TODOS
los bloques DCT del segmento son emitibles, no solo los que se sustituyen: si
uno se quedo a medias (porque venia de un segmento roto) y se empaqueta tal
cual, sale sin EOB y el decodificador se come el resto del segmento. Eso deja
franjas negras en la imagen.

La regla es sencilla:

  - Si el segmento de origen decodificaba entero, sus bloques valen completos,
    desbordamiento incluido.
  - Si no, solo valen las palabras que se leyeron dentro del area fija del
    propio bloque: lo que viene del desbordamiento puede ser basura de otro
    macrobloque. Se corta ahi y se cierra con EOB.
"""

from .bitstream import EOB, VLC_RUN


def sanitize_block(blk, trust_overflow):
    """Deja un bloque DCT listo para empaquetar. Devuelve True si lo ha tocado."""
    if trust_overflow and blk.done:
        return False
    n = min(int(blk.ntok_area), int(blk.ntok))
    keep = []
    pos = 0
    for i in range(n):
        t = int(blk.tok[i])
        pos += int(VLC_RUN[t]) + 1
        keep.append(t)
        if pos >= 64:
            break
    if pos < 64:
        keep.append(EOB)
    for i, t in enumerate(keep):
        blk.tok[i] = t
    blk.ntok = len(keep)
    blk.ntok_area = min(int(blk.ntok_area), len(keep))
    blk.pos = 128
    blk.done = 1
    return True


def sanitize_mb(mb, trust_overflow):
    n = 0
    for j in range(6):
        n += sanitize_block(mb.b[j], trust_overflow)
    return n


# Capacidad de datos de un segmento: 5 bloques de 76 bytes utiles.
SEG_CAPACITY_BITS = 5 * 76 * 8


def segment_bits(seg):
    from .bitstream import VLC_LEN, HDR_BITS
    total = 0
    for m in range(5):
        for j in range(6):
            b = seg.mb[m].b[j]
            total += HDR_BITS
            for i in range(int(b.ntok)):
                total += int(VLC_LEN[b.tok[i]])
    return total


def fit_segment(seg, capacity=SEG_CAPACITY_BITS):
    """Recorta coeficientes hasta que el segmento quepa.

    Al juntar macrobloques de origenes distintos sus coeficientes pueden no
    caber en los 380 bytes del segmento. Si se empaqueta sin mas, lo que se
    pierde es el final del ultimo bloque, EOB incluido, y el decodificador se
    come el resto del segmento: salen franjas negras.

    Aqui se quitan a proposito las frecuencias MAS ALTAS, que es lo que menos
    se nota, y siempre se deja el EOB. Quitar la ultima palabra de un bloque es
    seguro porque los recorridos son relativos: las que quedan no se mueven.
    Devuelve cuantos coeficientes se han tenido que soltar.
    """
    from .bitstream import VLC_LEN, EOB
    total = segment_bits(seg)
    if total <= capacity:
        return 0
    eob_len = int(VLC_LEN[EOB])
    dropped = 0
    while total > capacity:
        best = None
        for m in range(5):
            for j in range(6):
                b = seg.mb[m].b[j]
                if int(b.ntok) >= 2 and (best is None or b.ntok > best.ntok):
                    best = b
        if best is None:
            return -1
        n = int(best.ntok)
        # la ultima palabra es el terminador; se quita la anterior y se cierra
        # siempre con EOB, que puede ser mas corto que el terminador original
        total -= int(VLC_LEN[best.tok[n - 2]]) + int(VLC_LEN[best.tok[n - 1]])
        best.tok[n - 2] = EOB
        best.ntok = n - 1
        total += eob_len
        best.ntok_area = min(int(best.ntok_area), n - 1)
        dropped += 1
    return dropped
