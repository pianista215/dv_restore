"""Geometria de un frame DV: donde esta cada bloque y que significa cada byte.

Todo lo de este modulo es estructura fija del formato (IEC 61834 / SMPTE 314M),
verificada contra el material real. Nada de heuristica.

Un frame PAL son 12 secuencias DIF de 150 bloques de 80 bytes. Dentro de cada
secuencia:

    bloque 0        cabecera
    bloques 1-2     subcodigo
    bloques 3-5     VAUX
    bloques 6+16k   audio (k = 0..8)
    el resto        video (135 por secuencia, 1620 por frame)

Los 5 bloques de video consecutivos forman un SEGMENTO DE VIDEO: cinco
macrobloques que comparten el desbordamiento VLC. Por eso el segmento, y no el
bloque, es la unidad minima que se puede sustituir sin parsear el bitstream.
"""

import numpy as np

BLOCK = 80
BLOCKS_PER_SEQ = 150
AUDIO_IN_SEQ = [6 + 16 * k for k in range(9)]

# Los 3 primeros bytes de cada bloque son el ID DIF. Difieren entre capturas
# aunque el contenido sea identico, asi que toda comparacion los salta.
ID_BYTES = 3


class Profile:
    """Parametros que cambian entre PAL y NTSC."""

    def __init__(self, name, sequences, width, height):
        self.name = name
        self.sequences = sequences
        self.frame_size = sequences * BLOCKS_PER_SEQ * BLOCK
        self.width = width
        self.height = height
        self.mb_cols = width // 16
        self.mb_rows = height // 16

        video, audio = [], []
        for seq in range(sequences):
            for blk in range(BLOCKS_PER_SEQ):
                if blk < ID_BYTES + 3:          # cabecera, subcodigo y VAUX
                    continue
                off = (seq * BLOCKS_PER_SEQ + blk) * BLOCK
                (audio if blk in AUDIO_IN_SEQ else video).append(off)

        self.video = np.array(video, dtype=np.int64)
        self.audio = np.array(audio, dtype=np.int64)
        self.n_video = len(self.video)          # 1620 en PAL
        self.n_audio = len(self.audio)          # 108 en PAL
        self.n_seg = self.n_video // 5          # 324 segmentos de video

        # byte 3 de cada bloque de video: STA en el nibble alto, QNO en el bajo
        self.sta = self.video + 3
        # bytes de datos de un bloque de video, saltando el ID DIF
        self.vdata = self.video[:, None] + np.arange(ID_BYTES, BLOCK)[None, :]
        # muestras de audio: se salta tambien el pack AAUX de 5 bytes
        self.adata = self.audio[:, None] + np.arange(8, BLOCK)[None, :]
        # segmentos: (n_seg, 5) con el offset de cada uno de sus 5 macrobloques
        self.seg = self.video.reshape(self.n_seg, 5)
        self.seg_sta = self.seg + 3

    def __repr__(self):
        return (f"<Profile {self.name} {self.width}x{self.height} "
                f"{self.frame_size}B {self.n_video}mb {self.n_seg}seg>")


PAL = Profile("PAL", 12, 720, 576)
NTSC = Profile("NTSC", 10, 720, 480)


def profile_for(path_or_size):
    """Deduce el perfil por el tamano del fichero."""
    size = path_or_size
    if isinstance(path_or_size, str):
        import os
        size = os.path.getsize(path_or_size)
    if size % PAL.frame_size == 0:
        return PAL
    if size % NTSC.frame_size == 0:
        return NTSC
    # ninguno divide exacto: elegimos el que deje menos resto
    return PAL if size % PAL.frame_size <= size % NTSC.frame_size else NTSC


# --- areas fijas de los 6 bloques DCT dentro de un macrobloque -------------
#
# El payload de un bloque de video son 76 bytes (608 bits) repartidos en areas
# fijas: 14 bytes para cada uno de los 4 bloques Y y 10 para cada croma.
# Cada bloque DCT empieza con DC (9 bits), modo DCT (1 bit) y clase (2 bits);
# despues vienen los coeficientes AC en VLC, que pueden desbordar al hueco
# libre de los demas bloques del macrobloque y luego del segmento.
DCT_BLOCKS = 6
AREA_BITS = (112, 112, 112, 112, 80, 80)        # Y0 Y1 Y2 Y3 Cr Cb
AREA_BYTES = tuple(b // 8 for b in AREA_BITS)   # 14 14 14 14 10 10
assert sum(AREA_BYTES) == BLOCK - 4
# offset en bytes de cada bloque DCT desde el principio del bloque DIF
AREA_OFF = (4, 18, 32, 46, 60, 70)
HDR_BITS = 12                                   # DC(9) + modo DCT(1) + clase(2)
