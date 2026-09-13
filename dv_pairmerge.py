#!/usr/bin/env python3
"""
dv_pairmerge.py - Fusiona bloques de dos capturas DV con anclaje manual.

    python3 dv_pairmerge.py base.dv donante.dv salida.dv \\
            --base-start A --don-start B --count X

Tu garantizas que el frame A del base y el frame B del donante son el mismo
frame de cinta, y que a partir de ahi van en paralelo. El script no analiza
nada: procesa X parejas consecutivas y, para cada bloque de video danado en el
base, coge el del donante si ese esta sano. Si ninguno de los dos lo tiene
sano, se queda el del base.

No toca el subcodigo: timecode, fechas y todo lo demas siguen siendo los del
base. Solo cambia imagen (y audio si lo pides).

Por defecto la salida es el archivo base completo con ese tramo parcheado, asi
que puedes encadenar pasadas:

    dv_pairmerge.py base.dv p1.dv paso1.dv --base-start 43 --don-start 29 --count 500
    dv_pairmerge.py paso1.dv p2.dv paso2.dv --base-start 900 --don-start 12 --count 800

Opciones:
  --base-start A   Frame del base donde empieza el tramo. Por defecto 0.
  --don-start B    Frame del donante equivalente al A. Por defecto 0.
  --count X        Cuantas parejas procesar. Por defecto, hasta agotar uno.
  --only-range     Escribe solo el tramo procesado en vez del base entero.
  --audio          Copia tambien los bloques de audio del donante en los frames
                   donde el base tenga bloques de video danados y el donante
                   ninguno. Ojo: no va guiado por errores de audio, es una
                   sustitucion en bloque.
  --ntsc           Material NTSC (120000 bytes/frame). Por defecto PAL.
  --dry-run        Solo estadisticas, no escribe nada.
  --verbose        Una linea por frame.
"""

import sys
import os

try:
    import numpy as np
except ImportError:
    sys.exit("Falta numpy. Instalalo con:  pip3 install numpy")

BLOCK = 80
BLOCKS_PER_SEQ = 150
AUDIO_IN_SEQ = [6 + 16 * k for k in range(9)]


def layout(sequences):
    video, audio = [], []
    for seq in range(sequences):
        for blk in range(BLOCKS_PER_SEQ):
            off = (seq * BLOCKS_PER_SEQ + blk) * BLOCK
            if blk < 6:
                continue
            (audio if blk in AUDIO_IN_SEQ else video).append(off)
    return np.array(video, dtype=np.int64), np.array(audio, dtype=np.int64)


def load(path, frame_size):
    n = os.path.getsize(path) // frame_size
    if n == 0:
        sys.exit(f"ERROR: {path} no llega a un frame completo de {frame_size} bytes.")
    return np.memmap(path, dtype=np.uint8, mode='r', shape=(n, frame_size)), n


def main():
    args = sys.argv[1:]
    if len(args) < 3 or args[0] in ('-h', '--help'):
        print(__doc__)
        return 1

    ntsc = '--ntsc' in args
    dry = '--dry-run' in args
    verbose = '--verbose' in args
    do_audio = '--audio' in args
    only_range = '--only-range' in args
    vals = {'--base-start': 0, '--don-start': 0, '--count': None}
    for flag in list(vals):
        if flag in args:
            i = args.index(flag)
            vals[flag] = int(args[i + 1])
            del args[i:i + 2]
    args = [a for a in args if not a.startswith('--')]
    if len(args) < 3:
        print("Faltan argumentos: base.dv donante.dv salida.dv")
        return 1

    base_path, don_path, out_path = args[:3]
    frame_size = 120000 if ntsc else 144000
    sequences = 10 if ntsc else 12
    a0, b0 = vals['--base-start'], vals['--don-start']

    vid_off, aud_off = layout(sequences)
    sta_off = vid_off + 3
    data_idx = vid_off[:, None] + np.arange(3, BLOCK)[None, :]
    nvid = len(vid_off)

    A, na = load(base_path, frame_size)
    B, nb = load(don_path, frame_size)

    count = vals['--count']
    room = min(na - a0, nb - b0)
    if room <= 0:
        sys.exit(f"ERROR: fuera de rango. base tiene {na} frames (pediste desde {a0}), "
                 f"donante tiene {nb} (pediste desde {b0}).")
    if count is None or count > room:
        if count is not None:
            print(f"AVISO: pediste {count} frames pero solo caben {room}. Uso {room}.\n")
        count = room

    print(f"Base    : {base_path}  ({na} frames)")
    print(f"Donante : {don_path}  ({nb} frames)")
    print(f"Tramo   : base {a0}..{a0+count-1}  <->  donante {b0}..{b0+count-1}"
          f"   ({count} frames)\n")

    out = None if dry else open(out_path, 'wb')

    # frames del base anteriores al tramo, tal cual
    if out and not only_range:
        for i in range(a0):
            out.write(np.array(A[i]).tobytes())

    bad_before = bad_after = ported = both_bad = 0
    common = common_same = 0
    frames_touched = 0
    audio_swaps = 0
    perfect = untouched = 0

    for k in range(count):
        i, j = a0 + k, b0 + k
        frame = np.array(A[i])
        fb = np.array(B[j])
        sa = frame[sta_off] >> 4
        sb = fb[sta_off] >> 4
        bad_a = sa != 0
        bad_b = sb != 0
        nba = int(bad_a.sum())
        bad_before += nba

        # sanidad: de los bloques sanos en ambos, cuantos coinciden
        both_ok = (~bad_a) & (~bad_b)
        nboth = int(both_ok.sum())
        if nboth:
            idx = data_idx[both_ok]
            nsame = int((frame[idx] == fb[idx]).all(axis=1).sum())
            common += nboth
            common_same += nsame

        take = bad_a & (~bad_b)
        ntake = int(take.sum())
        both_bad += int((bad_a & bad_b).sum())
        if ntake:
            for off in vid_off[take]:
                frame[off:off + BLOCK] = fb[off:off + BLOCK]
            ported += ntake
            frames_touched += 1
        if do_audio and nba and not bad_b.any():
            for off in aud_off:
                frame[off:off + BLOCK] = fb[off:off + BLOCK]
            audio_swaps += 1

        rest = int((frame[sta_off] >> 4 != 0).sum())
        bad_after += rest
        if rest == 0 and nba:
            perfect += 1
        if nba and rest == nba:
            untouched += 1

        if verbose:
            if nba == 0:
                mark = "ya estaba limpio"
            elif rest == 0:
                mark = "REPARADO DEL TODO"
            elif ntake == 0:
                mark = "sin tocar (el donante no aporta nada)"
            else:
                mark = f"mejorado {100.0*ntake/nba:.0f}%"
            print(f"  base {i:>6} / don {j:>6}: danados {nba:>4} -> {rest:>4}"
                  f"   portados {ntake:>4}   {mark}")
        if out:
            out.write(frame.tobytes())
        if not verbose and k and k % 2000 == 0:
            print(f"  {k}/{count} frames...")

    # resto del base
    if out and not only_range:
        for i in range(a0 + count, na):
            out.write(np.array(A[i]).tobytes())
    if out:
        out.close()

    total = count * nvid
    pb = 100.0 * bad_before / total
    pa = 100.0 * bad_after / total
    print(f"\n--- Estadisticas del tramo ---")
    print(f"Bloques de video por frame     : {nvid}")
    print(f"Bloques danados en el base     : {bad_before}  ({pb:.3f}%)")
    print(f"Bloques portados del donante   : {ported}")
    print(f"Bloques danados en los dos     : {both_bad}  (irrecuperables aqui)")
    print(f"Bloques danados al terminar    : {bad_after}  ({pa:.3f}%)")
    print(f"Mejora                         : -{pb - pa:.3f} puntos")
    if bad_before:
        print(f"Recuperado del dano inicial    : {100.0*ported/bad_before:.1f}%")
    print(f"Frames tocados                 : {frames_touched} de {count}")
    print(f"Frames danados que no mejoran  : {untouched}")
    print(f"Frames que quedan impecables   : {perfect}")
    if do_audio:
        print(f"Frames con audio sustituido    : {audio_swaps}")

    print(f"\n--- Control del anclaje ---")
    if common:
        pct = 100.0 * common_same / common
        print(f"Bloques sanos en ambos         : {common}")
        print(f"De esos, identicos             : {common_same}  ({pct:.2f}%)")
        if pct > 99.0:
            print("El anclaje cuadra: son los mismos frames de cinta.")
        elif pct < 50.0:
            print("OJO: casi ningun bloque comun coincide. Si esperabas que\n"
                  "     fueran los mismos frames, revisa el anclaje o busca un\n"
                  "     hueco de frames dentro del tramo.")
        else:
            print("Coincidencia parcial: probablemente el tramo empieza bien y\n"
                  "     se desincroniza a mitad (algun frame perdido). Prueba a\n"
                  "     acortar --count.")
    else:
        print("No habia ningun bloque sano en ambos a la vez: sin control posible.")

    if not dry:
        print(f"\nSalida: {out_path}"
              + ("  (solo el tramo)" if only_range else "  (base completo, tramo parcheado)"))
    return 0


if __name__ == '__main__':
    sys.exit(main())
