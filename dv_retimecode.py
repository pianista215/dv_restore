#!/usr/bin/env python3
"""
dv_retimecode.py - Reescribe timecodes secuenciales limpios en un stream DV crudo.

Uso:
    python3 dv_retimecode.py entrada.dv salida.dv [--start N] [--ntsc] [--force]

--start N   El primer frame del archivo recibe el timecode correspondiente al
            frame N. Sirve para alinear varias capturas de la misma cinta:
            si el frame 15 de la captura A es el mismo trozo de cinta que el
            frame 0 de la captura B, entonces A va con --start 0 y B con
            --start 15.
--ntsc      El archivo es NTSC (120000 bytes/frame). Por defecto PAL (144000).
--force     Escribe el pack de timecode en TODOS los slots de subcodigo,
            incluso donde ahora hay otra cosa. Usalo si el informe final dice
            que muchos frames no tenian ningun pack de timecode.

No toca el video ni el audio: solo los 5 bytes del pack 0x13 dentro de los
bloques de subcodigo. Trabaja sobre una copia, nunca sobre el original.
"""

import sys
import os

PAL = dict(frame_size=144000, sequences=12)
NTSC = dict(frame_size=120000, sequences=10)

BLOCK = 80              # bytes por bloque DIF
BLOCKS_PER_SEQ = 150    # bloques DIF por secuencia
SUBCODE_BLOCKS = (1, 2)  # indices de los bloques de subcodigo dentro de la secuencia
SSYB_COUNT = 6          # sync blocks por bloque de subcodigo
SSYB_OFFSET = 3         # bytes de cabecera antes del primer SSYB
SSYB_SIZE = 8           # 3 bytes de id + 5 bytes de pack
PACK_TIMECODE = 0x13


def bcd(value):
    """Convierte 0-99 a BCD de un byte."""
    return ((value // 10) << 4) | (value % 10)


def timecode_pack(frame_index, fps):
    """Devuelve los 5 bytes del pack 0x13 para un indice de frame absoluto."""
    frames = frame_index % fps
    total_seconds = frame_index // fps
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = (total_seconds // 3600) % 24

    pc1 = bcd(frames)                # bits 7-6: color frame / drop frame (0 = non-drop)
    pc2 = 0x80 | bcd(seconds)
    pc3 = 0x80 | bcd(minutes)
    pc4 = 0xC0 | bcd(hours)
    return bytes([PACK_TIMECODE, pc1, pc2, pc3, pc4])


def process_frame(buf, pack, sequences, force):
    """Reescribe el timecode dentro de un frame. Devuelve nº de packs escritos."""
    written = 0
    for seq in range(sequences):
        for blk in SUBCODE_BLOCKS:
            base = (seq * BLOCKS_PER_SEQ + blk) * BLOCK
            for i in range(SSYB_COUNT):
                off = base + SSYB_OFFSET + i * SSYB_SIZE
                header = buf[off + 3]
                if force or header == PACK_TIMECODE:
                    buf[off + 3:off + 8] = pack
                    written += 1
    return written


def main():
    args = [a for a in sys.argv[1:]]
    if len(args) < 2 or args[0] in ('-h', '--help'):
        print(__doc__)
        return 1

    ntsc = '--ntsc' in args
    force = '--force' in args
    start = 0
    if '--start' in args:
        idx = args.index('--start')
        start = int(args[idx + 1])
        del args[idx:idx + 2]
    args = [a for a in args if not a.startswith('--')]

    if len(args) < 2:
        print("Faltan argumentos: entrada.dv salida.dv")
        return 1

    src, dst = args[0], args[1]
    profile = NTSC if ntsc else PAL
    fps = 30 if ntsc else 25
    frame_size = profile['frame_size']
    sequences = profile['sequences']

    size = os.path.getsize(src)
    total = size // frame_size
    remainder = size % frame_size

    print(f"Entrada : {src}")
    print(f"Formato : {'NTSC' if ntsc else 'PAL'} ({frame_size} bytes/frame, {fps} fps)")
    print(f"Frames  : {total}" + (f"  (+{remainder} bytes sobrantes al final)" if remainder else ""))
    print(f"Timecode: empieza en el frame {start}\n")

    if total == 0:
        print("ERROR: el archivo no llega ni a un frame completo. "
              "Comprueba que es DV crudo y que el formato PAL/NTSC es correcto.")
        return 1

    no_tc = 0
    with open(src, 'rb') as fin, open(dst, 'wb') as fout:
        for n in range(total):
            buf = bytearray(fin.read(frame_size))
            pack = timecode_pack(start + n, fps)
            if process_frame(buf, pack, sequences, force) == 0:
                no_tc += 1
            fout.write(buf)
            if n % 5000 == 0 and n:
                print(f"  {n}/{total} frames...")
        if remainder:
            fout.write(fin.read(remainder))

    print(f"\nSalida  : {dst}")
    first = timecode_pack(start, fps)
    last = timecode_pack(start + total - 1, fps)
    def show(p):
        return f"{p[4] & 0x3F:02x}:{p[3] & 0x7F:02x}:{p[2] & 0x7F:02x}:{p[1] & 0x3F:02x}"
    print(f"Rango   : {show(first)} -> {show(last)}")
    if no_tc:
        print(f"\nAVISO: {no_tc} de {total} frames no tenian ningun pack de timecode "
              f"y se han quedado sin el.\n       Vuelve a lanzarlo con --force para "
              f"escribirlo igualmente en esos frames.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
