#!/usr/bin/env python3
"""
dv_blockmerge.py - Fusiona dos capturas DV de la misma cinta a nivel de bloque.

    python3 dv_blockmerge.py base.dv donante.dv salida.dv [opciones]

Toma "base.dv" como referencia y, para cada frame, sustituye solo los bloques
de video marcados como danados por el equivalente de "donante.dv" cuando ese
si esta sano. La salida tiene exactamente los mismos frames que el base, asi
que puedes encadenar pasadas:

    dv_blockmerge.py base.dv pasada1.dv paso1.dv
    dv_blockmerge.py paso1.dv pasada2.dv paso2.dv

El alineado no usa timecode ni abst. Dos lecturas correctas del mismo punto de
cinta dan bloques byte a byte identicos: el programa calcula una huella de cada
bloque sano, vota que desfase explica mas coincidencias y empareja frame a
frame. Tolera huecos de captura en cualquiera de los dos archivos y encuentra
donantes cortos aunque correspondan a cualquier punto del base.

Opciones:
  --ntsc          El material es NTSC (120000 bytes/frame). Por defecto PAL.
  --min-votes N   Coincidencias de bloque para aceptar un emparejamiento (4).
  --drift N       Desviacion maxima admitida frente al desfase global (300).
  --probes N      Bloques por frame usados para la huella (64).
  --audio         Sustituye tambien el audio cuando el base tiene mas del 30%
                  de bloques danados y el donante ninguno.
  --dry-run       Analiza y muestra el informe sin escribir la salida.
"""

import sys
import os
from collections import defaultdict, Counter

try:
    import numpy as np
except ImportError:
    sys.exit("Falta numpy. Instalalo con:  pip3 install numpy")

BLOCK = 80
BLOCKS_PER_SEQ = 150
AUDIO_IN_SEQ = [6 + 16 * k for k in range(9)]
PRIME = np.uint64(1099511628211)
SEED = np.uint64(1469598103934665603)
MAX_FRAMES_PER_KEY = 200      # huellas mas repetidas que esto no discriminan
CHUNK = 2000                  # frames por lote al calcular huellas


def layout(sequences):
    video, audio = [], []
    for seq in range(sequences):
        for blk in range(BLOCKS_PER_SEQ):
            off = (seq * BLOCKS_PER_SEQ + blk) * BLOCK
            if blk < 6:
                continue
            (audio if blk in AUDIO_IN_SEQ else video).append(off)
    return np.array(video, dtype=np.int64), np.array(audio, dtype=np.int64)


def fingerprints(arr, probe_off):
    """Huella de 64 bits de cada bloque sonda, y si estaba sano."""
    n = arr.shape[0]
    idx = (probe_off[:, None] + np.arange(BLOCK)[None, :]).ravel()
    hs = np.empty((n, len(probe_off)), dtype=np.uint64)
    ok = np.empty((n, len(probe_off)), dtype=bool)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        data = np.ascontiguousarray(arr[a:b][:, idx]).reshape(b - a, len(probe_off), BLOCK)
        u = data.view(np.uint64).reshape(b - a, len(probe_off), BLOCK // 8)
        with np.errstate(over='ignore'):
            h = np.full((b - a, len(probe_off)), SEED, dtype=np.uint64)
            for k in range(u.shape[2]):
                h = (h ^ u[:, :, k]) * PRIME
        hs[a:b] = h
        ok[a:b] = (arr[a:b][:, probe_off + 3] >> 4) == 0
    return hs, ok


def verify(fa, fb, sta_off, data_idx, min_blocks=6, min_score=1.0):
    """Confirma que dos frames son el mismo punto de cinta comparando bytes."""
    both = ((fa[sta_off] >> 4) == 0) & ((fb[sta_off] >> 4) == 0)
    n = int(both.sum())
    if n < min_blocks:
        return False
    idx = data_idx[both]
    return float((fa[idx] == fb[idx]).all(axis=1).mean()) >= min_score


def load(path, frame_size):
    size = os.path.getsize(path)
    n = size // frame_size
    if n == 0:
        sys.exit(f"ERROR: {path} no contiene ni un frame completo de {frame_size} bytes.\n"
                 f"       Comprueba que es DV crudo y que PAL/NTSC es correcto.")
    return np.memmap(path, dtype=np.uint8, mode='r', shape=(n, frame_size)), n


def main():
    args = sys.argv[1:]
    if len(args) < 3 or args[0] in ('-h', '--help'):
        print(__doc__)
        return 1

    ntsc = '--ntsc' in args
    dry = '--dry-run' in args
    do_audio = '--audio' in args
    opts = {'--min-votes': 2, '--drift': 300, '--probes': 192}
    for flag in list(opts):
        if flag in args:
            i = args.index(flag)
            opts[flag] = int(args[i + 1])
            del args[i:i + 2]
    args = [a for a in args if not a.startswith('--')]
    if len(args) < 3:
        print("Faltan argumentos: base.dv donante.dv salida.dv")
        return 1

    base_path, don_path, out_path = args[:3]
    frame_size = 120000 if ntsc else 144000
    sequences = 10 if ntsc else 12
    min_votes, drift, nprobe = opts['--min-votes'], opts['--drift'], opts['--probes']

    vid_off, aud_off = layout(sequences)
    sta_off = vid_off + 3
    data_idx = vid_off[:, None] + np.arange(3, BLOCK)[None, :]
    step = max(1, len(vid_off) // nprobe)
    probe_off = vid_off[::step][:nprobe]

    A, na = load(base_path, frame_size)
    B, nb = load(don_path, frame_size)
    print(f"Base    : {base_path}  ({na} frames)")
    print(f"Donante : {don_path}  ({nb} frames)")
    print(f"Sondas por frame: {len(probe_off)} de {len(vid_off)} bloques de video\n")

    print("Calculando huellas...")
    hb, okb = fingerprints(A, probe_off)
    hd, okd = fingerprints(B, probe_off)

    print("Indexando el base...")
    index = defaultdict(list)
    for f in range(na):
        row, good = hb[f], okb[f]
        for p in np.nonzero(good)[0]:
            index[int(row[p])].append(f)
    dropped = 0
    for k in list(index):
        if len(index[k]) > MAX_FRAMES_PER_KEY:
            del index[k]
            dropped += 1

    print("Votando el desfase global...")
    cand = [None] * nb
    globalvotes = Counter()
    for d in range(nb):
        row, good = hd[d], okd[d]
        v = Counter()
        for p in np.nonzero(good)[0]:
            for f in index.get(int(row[p]), ()):
                v[f] += 1
        cand[d] = v
        for f, c in v.items():
            if c >= min_votes:
                globalvotes[f - d] += c
    if not globalvotes:
        sys.exit("No hay ninguna coincidencia de bloque entre los dos archivos.\n"
                 "Comprueba que son capturas de la misma cinta y del mismo formato.")
    top = globalvotes.most_common(3)
    offset = top[0][0]
    print(f"Desfase global: {offset} frames  ({top[0][1]} votos)")
    if len(top) > 1:
        alt = ", ".join(f"{o} ({v})" for o, v in top[1:])
        print(f"  otros candidatos: {alt}")
    print("  (el frame 0 del donante corresponde al frame "
          f"{offset} del base)")
    print()

    # emparejamiento donante -> base:
    # primero el frame que predice el desfase global; si no verifica,
    # se prueban los mejores candidatos por votos, mas cercanos primero.
    print("Emparejando y verificando byte a byte...")
    match, rejected, used_votes = {}, 0, 0
    for d in range(nb):
        tries = []
        f0 = d + offset
        if 0 <= f0 < na:
            tries.append(f0)
        ranked = sorted(((c, -abs((f - d) - offset), f) for f, c in cand[d].items()
                         if c >= min_votes and abs((f - d) - offset) <= drift),
                        reverse=True)
        for _, _, f in ranked[:5]:
            if f != f0:
                tries.append(f)
        got = None
        for f in tries:
            if f in match:
                continue
            if verify(np.array(A[f]), np.array(B[d]), sta_off, data_idx):
                got = f
                break
        if got is None:
            if tries:
                rejected += 1
        else:
            match[got] = d
            if got != f0:
                used_votes += 1
        if d % 500 == 0 and d:
            print(f"  {d}/{nb} frames del donante...")

    withvotes = sum(1 for d in range(nb) if cand[d])
    print(f"Frames del donante emparejados: {len(match)} de {nb}"
          + (f"  ({used_votes} reenganchados tras un hueco)" if used_votes else ""))
    print(f"  (con alguna coincidencia de bloque: {withvotes})")
    print()

    out = None if dry else open(out_path, 'wb')
    repaired_frames = repaired_blocks = audio_swaps = 0
    bad_before = bad_after = 0
    total = na * len(vid_off)

    for i in range(na):
        frame = np.array(A[i])
        sta = frame[sta_off] >> 4
        bad = sta != 0
        nbad = int(bad.sum())
        bad_before += nbad

        j = match.get(i)
        if j is not None and nbad:
            sta_b = B[j][sta_off] >> 4
            take = bad & (sta_b == 0)
            ntake = int(take.sum())
            if ntake:
                for off in vid_off[take]:
                    frame[off:off + BLOCK] = B[j][off:off + BLOCK]
                repaired_blocks += ntake
                repaired_frames += 1
            if do_audio and nbad > 0.30 * len(vid_off) and not sta_b.any():
                for off in aud_off:
                    frame[off:off + BLOCK] = B[j][off:off + BLOCK]
                audio_swaps += 1

        bad_after += int((frame[sta_off] >> 4 != 0).sum())
        if out:
            out.write(frame.tobytes())
        if i % 2000 == 0 and i:
            print(f"  {i}/{na} frames...")

    if out:
        out.close()

    print(f"\n--- Resultado ---")
    print(f"Frames reparados          : {repaired_frames}")
    print(f"Bloques de video portados : {repaired_blocks}")
    if do_audio:
        print(f"Frames con audio sustituido: {audio_swaps}")
    pb = 100.0 * bad_before / total
    pa = 100.0 * bad_after / total
    print(f"\nBloques danados: {pb:.3f}%  ->  {pa:.3f}%   (-{pb - pa:.3f} puntos)")
    if not dry:
        print(f"\nSalida: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
