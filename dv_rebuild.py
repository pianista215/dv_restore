#!/usr/bin/env python3
"""
dv_rebuild.py - Reconstruye la linea temporal usando otra pasada.

    python3 dv_rebuild.py base.dv donante.dv salida.dv

A diferencia de dv_autofix, este SI anade frames. Recorre el base buscando
anclas (frames que casan byte a byte con uno del donante) y, entre dos anclas
consecutivas, mira cuantos frames tiene cada uno de ese mismo trozo de cinta:
si el donante tiene mas, es que al base le faltan y se meten los del donante.

Lo suyo es pasar antes dv_autofix para que el base tenga ya los bloques
reparados, y despues este para tapar los huecos.

La salida es mas larga que el base: crece justo lo que se ha insertado.

Opciones:
  --ntsc          Material NTSC (120000 bytes/frame). Por defecto PAL.
  --probes N      Bloques usados como huella por frame (por defecto 128).
  --candidates N  Candidatos a verificar por frame (por defecto 60).
  --min-gain N    Solo rellena un hueco si el donante aporta al menos N frames
                  mas que el base (por defecto 1).
  --bulk-fill     MODO DE AYER: rellena el hueco metiendo los frames del
                  donante enteros, sin emparejar uno a uno ni comprobar si
                  duplican. Recupera mas continuidad pero puede duplicar.
                  Combinalo con --min-audio 0 --max-gap 0 para reproducir
                  exactamente el comportamiento anterior.
  --drop-orphans  Descarta los frames del base que queden entre dos anclas que
                  el donante da como consecutivas: no caben en la cinta, son
                  frames que solto la camara al patinar. Quita ruido y tirones.
  --orphan-factor N  Con --drop-orphans, un frame se descarta si tiene mas de
                  N veces el dano de las anclas que lo rodean (por defecto 2).
                  Subelo para ser mas conservador.
  --orphan-damage N  Con --drop-orphans, umbral ABSOLUTO: descarta los frames
                  con mas de N bloques danados, sin mirar sus anclas. Mas
                  predecible que --orphan-factor. Prueba con 150-250.
  --only-empty    MODO PRUDENTE: rellena solo los huecos donde el base no tiene
                  NINGUN frame. Evita duplicar material cuando encadenas muchos
                  donantes, a cambio de recuperar algo menos.
  --max-gap N     No rellenar huecos de mas de N frames del donante (por
                  defecto 400). Un hueco de captura real no pasa de unos
                  cientos; huecos enormes indican anclas falsas. 0 = sin limite.
  --smooth N      Descarta anclas cuyo desfase se separe mas de N frames de
                  sus vecinas. DESACTIVADO por defecto (0): un hueco real
                  cambia el desfase justo en su tamano, asi que este filtro
                  elimina las anclas que delimitan los huecos buenos y hace
                  que se pierdan frames. Usalo solo si sospechas de anclas
                  falsas y con un valor alto.
  --min-audio N   Bloques de audio identicos exigidos para aceptar un ancla
                  (por defecto 20). Bajalo si el material tiene mucho silencio
                  y salen pocas anclas.
  --report FICH   Guarda el mapa de anclas y huecos en un fichero.
  --dry-run       Solo informa, no escribe la salida.
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
            if blk < 6:
                continue
            (audio if blk in AUDIO_IN_SEQ else video).append(
                (seq * BLOCKS_PER_SEQ + blk) * BLOCK)
    return np.array(video, dtype=np.int64), np.array(audio, dtype=np.int64)


def fingerprints(arr, probe):
    n = arr.shape[0]
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
    report_path = None
    if '--report' in args:
        i = args.index('--report')
        report_path = args[i + 1]
        del args[i:i + 2]
    opts = {'--probes': 128, '--candidates': 60, '--min-gain': 1,
            '--max-gap': 400, '--min-audio': 20, '--smooth': 0, '--orphan-factor': 2, '--orphan-damage': 0}
    for flag in list(opts):
        if flag in args:
            i = args.index(flag)
            v = args[i + 1]
            opts[flag] = float(v) if flag == '--orphan-factor' else int(v)
            del args[i:i + 2]
    args = [a for a in args if not a.startswith('--')]
    if len(args) < 3:
        print("Faltan argumentos: base.dv donante.dv salida.dv")
        return 1

    base_path, don_path, out_path = args[:3]
    frame_size = 120000 if ntsc else 144000
    sequences = 10 if ntsc else 12
    nprobe, ncand, min_gain = opts['--probes'], opts['--candidates'], opts['--min-gain']
    max_gap = opts['--max-gap']
    min_audio = opts['--min-audio']
    smooth = opts['--smooth']
    orphan_factor = opts['--orphan-factor']
    orphan_damage = opts['--orphan-damage']
    only_empty = '--only-empty' in sys.argv
    bulk = '--bulk-fill' in sys.argv
    drop_orphans = '--drop-orphans' in sys.argv

    vid_off, aud_off = layout(sequences)
    # solo las muestras de audio: saltamos cabecera DIF y pack AAUX, que
    # difieren entre capturas aunque el audio sea el mismo
    aud_idx = aud_off[:, None] + np.arange(8, BLOCK)[None, :]
    sta_off = vid_off + 3
    data_idx = vid_off[:, None] + np.arange(3, BLOCK)[None, :]
    step = max(1, len(vid_off) // nprobe)
    probe = vid_off[::step][:nprobe]

    A, na = load(base_path, frame_size)
    B, nb = load(don_path, frame_size)
    print(f"Base    : {base_path}  ({na} frames)")
    print(f"Donante : {don_path}  ({nb} frames)\n")

    print("Calculando huellas...")
    hd, okd = fingerprints(B, probe)
    index = defaultdict(list)
    for d in range(nb):
        for p in np.nonzero(okd[d])[0]:
            index[int(hd[d, p])].append(d)
    for k in list(index):
        if len(index[k]) > MAX_PER_KEY:
            del index[k]
    ha, oka = fingerprints(A, probe)

    def verify(fa, fb):
        """Ancla: todos los bloques de video sanos comunes deben coincidir, y
        ademas el audio. Sin la comprobacion de audio, dos frames distintos de
        una escena estatica pueden compartir bloques de video identicos y
        emparejarse mal."""
        both = ((fa[sta_off] >> 4) == 0) & ((fb[sta_off] >> 4) == 0)
        n = int(both.sum())
        if n < 6:
            return False
        idx = data_idx[both]
        if not bool((fa[idx] == fb[idx]).all()):
            return False
        aud = int((fa[aud_idx] == fb[aud_idx]).all(axis=1).sum())
        return aud >= min_audio

    def same_frame(fa, fb):
        """Criterio relajado, para decidir si dos frames de un hueco son el
        mismo instante de cinta. Usa tambien el audio, que no lleva marca de
        error y sirve de prueba cuando el video esta muy danado."""
        both = ((fa[sta_off] >> 4) == 0) & ((fb[sta_off] >> 4) == 0)
        n = int(both.sum())
        if n:
            idx = data_idx[both]
            if not bool((fa[idx] == fb[idx]).all()):
                return False
        aud = int((fa[aud_idx] == fb[aud_idx]).all(axis=1).sum())
        if n == 0:
            return aud >= 20
        return n + aud >= 3

    print("Buscando anclas...")
    anchors = []          # (frame del base, frame del donante)
    last_j = -1
    for i in range(na):
        votes = Counter()
        for p in np.nonzero(oka[i])[0]:
            for d in index.get(int(ha[i, p]), ()):
                if d > last_j:
                    votes[d] += 1
        if not votes:
            continue
        fa = np.array(A[i])
        for cand, _ in votes.most_common(ncand):
            if verify(fa, np.array(B[cand])):
                anchors.append((i, cand))
                last_j = cand
                break
        if i and i % 2000 == 0:
            print(f"  {i}/{na} frames...  {len(anchors)} anclas")

    if smooth and len(anchors) >= 7:
        # Un ancla falsa aparece como un pico en el desfase: se separa de sus
        # vecinas mientras estas concuerdan entre si. Las descartamos.
        offs = [j - i for i, j in anchors]
        limpio = []
        for k in range(len(anchors)):
            a, b = max(0, k - 3), min(len(offs), k + 4)
            ventana = sorted(offs[a:k] + offs[k + 1:b])
            if not ventana:
                limpio.append(anchors[k])
                continue
            med = ventana[len(ventana) // 2]
            if abs(offs[k] - med) <= smooth:
                limpio.append(anchors[k])
        quitadas = len(anchors) - len(limpio)
        if quitadas:
            print(f"Anclas descartadas por desfase incoherente: {quitadas}")
        anchors = limpio

    print(f"\nAnclas encontradas: {len(anchors)}")
    if len(anchors) < 3:
        print(f"Solo {len(anchors)} anclas: estas dos capturas apenas comparten\n"
              f"nada, o el emparejamiento no es fiable. Copio el base sin tocar.")
        if not dry:
            with open(out_path, 'wb') as out:
                for k in range(na):
                    out.write(np.array(A[k]).tobytes())
            print(f"Salida: {out_path}")
        return 0

    # decidir que hacer con cada hueco entre anclas consecutivas
    rep = open(report_path, 'w') if report_path else None
    inserted = gaps_filled = gaps_kept = dropped_orphans = 0

    def emit_base(Ga, pi, ni):
        """Emite los frames del base de un hueco, descartando los que estan
        claramente peor que las anclas que los rodean."""
        nonlocal dropped_orphans
        if not (drop_orphans and Ga):
            seq_out.extend(("base", k) for k in Ga)
            return
        d_prev = int(((np.array(A[pi])[sta_off] >> 4) != 0).sum())
        d_next = int(((np.array(A[ni])[sta_off] >> 4) != 0).sum())
        if orphan_damage:
            umbral = orphan_damage
        else:
            umbral = orphan_factor * max(d_prev, d_next) + 50
        for k in Ga:
            dk = int(((np.array(A[k])[sta_off] >> 4) != 0).sum())
            if dk > umbral:
                dropped_orphans += 1
                if rep:
                    rep.write(f"base {k}: espurio, {dk} bloques danados frente "
                              f"a {d_prev}/{d_next} de sus anclas\n")
            else:
                seq_out.append(("base", k))

    prev_i, prev_j = anchors[0]
    seq_out = [("base", k) for k in range(0, prev_i + 1)]
    for (i, j) in anchors[1:]:
        Ga = list(range(prev_i + 1, i))
        Gb = list(range(prev_j + 1, j))
        # guarda contra anclas falsas: si los primeros frames del hueco del
        # donante son los mismos que el base ya tiene tras el hueco, no hay
        # hueco de verdad y rellenar duplicaria material
        spurious = False
        if Gb and not bulk:
            fa_next = np.array(A[i])
            for b in Gb[:6]:
                if same_frame(fa_next, np.array(B[b])):
                    spurious = True
                    break
        if spurious:
            emit_base(Ga, prev_i, i)
            gaps_kept += 1
        elif not Gb:
            emit_base(Ga, prev_i, i)
        elif not Ga:
            seq_out += [("don", k) for k in Gb]
            inserted += len(Gb)
            gaps_filled += 1
            if rep:
                rep.write(f"hueco vacio tras base {prev_i}: +{len(Gb)} del donante\n")
        elif only_empty or (max_gap and len(Gb) > max_gap):
            emit_base(Ga, prev_i, i)
            gaps_kept += 1
        elif bulk:
            # modo de ayer: si el donante tiene mas frames en el hueco, se
            # meten los suyos enteros y se descartan los del base
            if len(Gb) - len(Ga) >= min_gain:
                seq_out += [("don", k) for k in Gb]
                inserted += len(Gb) - len(Ga)
                gaps_filled += 1
                if rep:
                    rep.write(f"hueco tras base {prev_i}: base {len(Ga)}, "
                              f"donante {len(Gb)} -> relleno en bloque\n")
            else:
                emit_base(Ga, prev_i, i)
                gaps_kept += 1
        else:
            # emparejar dentro del hueco: solo se anaden los frames del
            # donante que no tengan equivalente en el base
            pa, added = 0, 0
            for b in Gb:
                fb = np.array(B[b])
                hit = None
                for k in range(pa, min(len(Ga), pa + 60)):
                    if same_frame(np.array(A[Ga[k]]), fb):
                        hit = k
                        break
                if hit is None:
                    seq_out.append(("don", b))
                    added += 1
                else:
                    seq_out += [("base", Ga[k]) for k in range(pa, hit + 1)]
                    pa = hit + 1
            seq_out += [("base", Ga[k]) for k in range(pa, len(Ga))]
            if added:
                inserted += added
                gaps_filled += 1
                if rep:
                    rep.write(f"hueco tras base {prev_i}: base {len(Ga)}, "
                              f"donante {len(Gb)} -> +{added} nuevos\n")
            else:
                gaps_kept += 1
        seq_out.append(("base", i))
        prev_i, prev_j = i, j
    seq_out += [("base", k) for k in range(prev_i + 1, na)]
    if rep:
        rep.close()

    total = len(seq_out)
    print(f"Huecos rellenados con el donante : {gaps_filled}")
    print(f"Huecos en los que gana el base   : {gaps_kept}")
    print(f"Frames insertados                : {inserted}")
    if drop_orphans:
        print(f"Frames espurios descartados      : {dropped_orphans}")
    print(f"\nLongitud: {na} -> {total} frames  "
          f"({(total-na)/25.0:+.1f} s de material recuperado)")

    if dry:
        print("\n(--dry-run: no se ha escrito nada)")
        return 0

    print(f"\nEscribiendo {out_path}...")
    with open(out_path, 'wb') as out:
        for src, k in seq_out:
            arr = A if src == "base" else B
            out.write(np.array(arr[k]).tobytes())
    print(f"Salida: {out_path}")
    if report_path:
        print(f"Detalle de los huecos en: {report_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
