#!/usr/bin/env python3
"""
dv_autofix.py - Repara un DV con otra pasada, emparejando frame a frame.

    python3 dv_autofix.py base.dv donante.dv salida.dv

Pensado para capturas con la linea temporal rota: no usa anclas, ni desfases,
ni tramos. Para CADA frame del base busca su pareja real entre todos los del
donante, la verifica byte a byte, y solo entonces porta los bloques danados.
Si un frame no tiene pareja fiable, se queda como esta.

La salida tiene exactamente los mismos frames que el base, asi que puedes
encadenar pasadas:

    dv_autofix.py parte1_mala.dv pasada1.dv paso1.dv
    dv_autofix.py paso1.dv pasada2.dv paso2.dv

Opciones:
  --ntsc         Material NTSC (120000 bytes/frame). Por defecto PAL.
  --probes N     Bloques usados como huella por frame (por defecto 128).
  --window N     Al seguir la pista, cuantos frames mirar alrededor antes de
                 buscar por todo el donante (por defecto 400). Solo afecta a
                 la velocidad, no al resultado.
  --candidates N Cuantos candidatos verificar por frame (por defecto 60). Subelo
                 si quedan muchos "sin pareja" en escenas estaticas; baja si va
                 muy lento.
  --audio        Sustituye tambien el audio en frames donde el base tenga mas
                 del 30% de bloques danados y el donante ninguno.
  --report FICH  Guarda el detalle por frame en un fichero.
  --dry-run      Solo estadisticas, no escribe la salida.
  --verbose      Una linea por frame reparado en pantalla.
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
MAX_PER_KEY = 400
CHUNK = 2000


def layout(sequences):
    video, audio = [], []
    for seq in range(sequences):
        for blk in range(BLOCKS_PER_SEQ):
            off = (seq * BLOCKS_PER_SEQ + blk) * BLOCK
            if blk < 6:
                continue
            (audio if blk in AUDIO_IN_SEQ else video).append(off)
    return np.array(video, dtype=np.int64), np.array(audio, dtype=np.int64)


def fingerprints(arr, probe):
    n = arr.shape[0]
    # solo bytes de datos: saltamos la cabecera DIF, que puede diferir
    # entre capturas aunque el contenido sea identico
    idx = (probe[:, None] + np.arange(8, BLOCK)[None, :]).ravel()
    hs = np.empty((n, len(probe)), dtype=np.uint64)
    ok = np.empty((n, len(probe)), dtype=bool)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        data = np.ascontiguousarray(arr[a:b][:, idx]).reshape(b - a, len(probe), BLOCK - 8)
        u = data.view(np.uint64).reshape(b - a, len(probe), (BLOCK - 8) // 8)
        with np.errstate(over='ignore'):
            h = np.full((b - a, len(probe)), SEED, dtype=np.uint64)
            for k in range(u.shape[2]):
                h = (h ^ u[:, :, k]) * PRIME
        hs[a:b] = h
        ok[a:b] = (arr[a:b][:, probe + 3] >> 4) == 0
    return hs, ok


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
    report_path = None
    if '--report' in args:
        i = args.index('--report')
        report_path = args[i + 1]
        del args[i:i + 2]
    opts = {'--probes': 128, '--window': 400, '--candidates': 60}
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
    nprobe, window = opts['--probes'], opts['--window']
    ncand = opts['--candidates']

    vid_off, aud_off = layout(sequences)
    sta_off = vid_off + 3
    data_idx = vid_off[:, None] + np.arange(3, BLOCK)[None, :]
    step = max(1, len(vid_off) // nprobe)
    probe = vid_off[::step][:nprobe]

    A, na = load(base_path, frame_size)
    B, nb = load(don_path, frame_size)
    print(f"Base    : {base_path}  ({na} frames)")
    print(f"Donante : {don_path}  ({nb} frames)")
    print(f"Emparejando frame a frame, sin suponer ningun desfase.\n")

    print("Calculando huellas del donante...")
    hd, okd = fingerprints(B, probe)
    index = defaultdict(list)
    for d in range(nb):
        row, good = hd[d], okd[d]
        for p in np.nonzero(good)[0]:
            index[int(row[p])].append(d)
    dropped = 0
    for k in list(index):
        if len(index[k]) > MAX_PER_KEY:
            del index[k]
            dropped += 1

    print("Calculando huellas del base...")
    ha, oka = fingerprints(A, probe)

    def verify(fa, fb):
        both = ((fa[sta_off] >> 4) == 0) & ((fb[sta_off] >> 4) == 0)
        n = int(both.sum())
        if n < 6:
            return False, 0
        idx = data_idx[both]
        return bool((fa[idx] == fb[idx]).all()), n

    if '--debug' in sys.argv:
        di = int(sys.argv[sys.argv.index('--debug') + 1])
        print(f"\n=== DEPURACION del frame {di} del base ===")
        good = oka[di]
        print(f"Sondas totales           : {len(probe)}")
        print(f"Sondas sanas en este frame: {int(good.sum())}")
        hits = miss = 0
        v = Counter()
        for p in np.nonzero(good)[0]:
            lst = index.get(int(ha[di, p]))
            if lst:
                hits += 1
                for d in lst:
                    v[d] += 1
            else:
                miss += 1
        print(f"Sondas que encuentran algo: {hits}")
        print(f"Sondas sin coincidencia   : {miss}")
        if v:
            print(f"Candidatos distintos      : {len(v)}")
            print(f"Mejores: {v.most_common(6)}")
            for cand, c in v.most_common(3):
                ok_, n = verify(np.array(A[di]), np.array(B[cand]))
                print(f"  donante {cand}: {c} votos, {n} bloques comunes, verifica={ok_}")
        else:
            print("NINGUN voto: las huellas del base no aparecen en el indice.")
        print(f"\nClaves en el indice       : {len(index)}")
        print(f"Claves descartadas por comunes: {dropped}")
        return 0

    print("Reparando...\n")
    out = None if dry else open(out_path, 'wb')
    rep = open(report_path, 'w') if report_path else None

    last = None
    matched = repaired = unmatched = 0
    ported = bad_before = bad_after = both_bad = 0
    perfect = untouched = audio_swaps = 0

    for i in range(na):
        frame = np.array(A[i])
        sta = frame[sta_off] >> 4
        bad = sta != 0
        nba = int(bad.sum())
        bad_before += nba

        # candidatos: primero el que continua la pista, luego por huella
        tries = []
        if last is not None and 0 <= last + 1 < nb:
            tries.append(last + 1)
        votes = Counter()
        for p in np.nonzero(oka[i])[0]:
            for d in index.get(int(ha[i, p]), ()):
                votes[d] += 1
        ranked = [d for d, _ in votes.most_common(ncand * 2)]
        if last is not None:
            ranked.sort(key=lambda d: abs(d - last) if abs(d - last) <= window else 10**9)
        tries += [d for d in ranked if d not in tries]

        j = None
        for cand in tries[:ncand]:
            ok_, _ = verify(frame, np.array(B[cand]))
            if ok_:
                j = cand
                break

        if j is None:
            unmatched += 1
            if nba:
                untouched += 1
            bad_after += nba
            if rep:
                rep.write(f"{i}\t-\t{nba}\t{nba}\t0\tsin pareja\n")
            if out:
                out.write(frame.tobytes())
            continue

        matched += 1
        last = j
        if nba:
            sta_b = B[j][sta_off] >> 4
            take = bad & (sta_b == 0)
            ntake = int(take.sum())
            both_bad += int((bad & (sta_b != 0)).sum())
            if ntake:
                for off in vid_off[take]:
                    frame[off:off + BLOCK] = B[j][off:off + BLOCK]
                ported += ntake
                repaired += 1
            if do_audio and nba > 0.30 * len(vid_off) and not sta_b.any():
                for off in aud_off:
                    frame[off:off + BLOCK] = B[j][off:off + BLOCK]
                audio_swaps += 1

        rest = int((frame[sta_off] >> 4 != 0).sum())
        bad_after += rest
        if nba and rest == 0:
            perfect += 1
        if nba and rest == nba:
            untouched += 1
        if rep:
            rep.write(f"{i}\t{j}\t{nba}\t{rest}\t{nba-rest}\tok\n")
        if verbose and nba:
            print(f"  base {i:>6} / don {j:>6}: {nba:>4} -> {rest:>4}")
        if out:
            out.write(frame.tobytes())
        if not verbose and i and i % 500 == 0:
            print(f"  {i}/{na} frames...  emparejados {matched}, reparados {repaired}")

    if out:
        out.close()
    if rep:
        rep.close()

    total = na * len(vid_off)
    pb = 100.0 * bad_before / total
    pa = 100.0 * bad_after / total
    print(f"\n--- Resultado ---")
    print(f"Frames con pareja verificada  : {matched} de {na}")
    print(f"Frames sin pareja en el donante: {unmatched}")
    print(f"Frames reparados              : {repaired}")
    print(f"Frames que quedan impecables  : {perfect}")
    print(f"Frames danados que no mejoran : {untouched}")
    print(f"Bloques portados              : {ported}")
    print(f"Bloques danados en los dos    : {both_bad}")
    if do_audio:
        print(f"Frames con audio sustituido   : {audio_swaps}")
    print(f"\nBloques danados: {pb:.3f}%  ->  {pa:.3f}%   (-{pb-pa:.3f} puntos)")
    if bad_before:
        print(f"Recuperado del dano inicial   : {100.0*ported/bad_before:.1f}%")
    if report_path:
        print(f"\nDetalle por frame en: {report_path}")
    if not dry:
        print(f"Salida: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
