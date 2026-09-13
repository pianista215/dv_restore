#!/usr/bin/env python3
"""
dv_diag.py - Por que dos capturas de la misma cinta no casan.

    python3 dv_diag.py base.dv donante.dv [desfase]

Para una muestra de frames compara el frame d del donante con el frame
d+desfase del base y cuenta, bloque a bloque de video:
  sanos A   : bloques sanos en el base
  sanos B   : bloques sanos en el donante
  ambos     : posiciones donde los dos estan sanos
  iguales   : de esas, cuantas coinciden byte a byte
Con --find busca cada frame del donante por todo el base (no supone ningun
desfase) y dice a que frame del base corresponde realmente.
"""
import sys, os
import numpy as np

BLOCK = 80

def layout(sequences=12):
    v = []
    for seq in range(sequences):
        for blk in range(150):
            if blk < 6 or blk in [6 + 16 * k for k in range(9)]:
                continue
            v.append((seq * 150 + blk) * BLOCK)
    return np.array(v)

PRIME = np.uint64(1099511628211)
SEED = np.uint64(1469598103934665603)


def fingerprints(arr, probe):
    n = arr.shape[0]
    idx = (probe[:, None] + np.arange(BLOCK)[None, :]).ravel()
    hs = np.empty((n, len(probe)), dtype=np.uint64)
    ok = np.empty((n, len(probe)), dtype=bool)
    for a in range(0, n, 2000):
        b = min(a + 2000, n)
        data = np.ascontiguousarray(arr[a:b][:, idx]).reshape(b - a, len(probe), BLOCK)
        u = data.view(np.uint64).reshape(b - a, len(probe), BLOCK // 8)
        with np.errstate(over='ignore'):
            h = np.full((b - a, len(probe)), SEED, dtype=np.uint64)
            for k in range(u.shape[2]):
                h = (h ^ u[:, :, k]) * PRIME
        hs[a:b] = h
        ok[a:b] = (arr[a:b][:, probe + 3] >> 4) == 0
    return hs, ok


def find_mode(A, B, V):
    from collections import defaultdict, Counter
    probe = V[::8]
    print(f"Buscando cada frame del donante por todo el base "
          f"({len(probe)} sondas)...\n")
    hb, okb = fingerprints(A, probe)
    hd, okd = fingerprints(B, probe)
    index = defaultdict(list)
    for f in range(A.shape[0]):
        for p in np.nonzero(okb[f])[0]:
            index[int(hb[f, p])].append(f)
    print(f"{'donante':>8} {'base':>8} {'votos':>7}   {'sanos B':>8}")
    found = 0
    n = B.shape[0]
    sample = sorted(set(list(range(0, min(n, 120), 5))
                        + list(range(0, n, max(1, n // 25)))))
    for d in sample:
        v = Counter()
        for p in np.nonzero(okd[d])[0]:
            for f in index.get(int(hd[d, p]), ()):
                v[f] += 1
        if v:
            f, c = v.most_common(1)[0]
            print(f"{d:>8} {f:>8} {c:>7}   {int(okd[d].sum()*8):>8}")
            found += 1
        else:
            print(f"{d:>8} {'(nada)':>8} {0:>7}   {int(okd[d].sum()*8):>8}")
    print(f"\n{found} de {len(sample)} frames muestreados aparecen en el base.")
    if 0 < found < len(sample):
        last = None
        for d in sample:
            pass
        print("Las filas con (nada) son tramos que el base no tiene.")
    if found < len(sample) * 0.5:
        print("\nEl base NO contiene la mayor parte de este material: no es un\n"
              "parche, es contenido que falta. Habria que insertarlo, no fusionarlo.")


def pair_mode(A, B, V, f, d):
    S = V + 3
    D = V[:, None] + np.arange(3, BLOCK)[None, :]
    print(f"Comparando base {f} con donante {d}, y desfases cercanos:\n")
    print(f"{'base':>6} {'donante':>8} {'desf':>5} {'sanos A':>8} {'sanos B':>8} "
          f"{'ambos':>7} {'iguales':>8}")
    fb = np.array(B[d])
    gb = (fb[S] >> 4) == 0
    for delta in range(-4, 5):
        ff = f + delta
        if not (0 <= ff < A.shape[0]):
            continue
        fa = np.array(A[ff])
        ga = (fa[S] >> 4) == 0
        both = ga & gb
        n = int(both.sum())
        same = int((fa[D[both]] == fb[D[both]]).all(axis=1).sum()) if n else 0
        mark = "  <-- MISMO FRAME" if n and same == n else ""
        print(f"{ff:>6} {d:>8} {ff-d:>5} {int(ga.sum()):>8} {int(gb.sum()):>8} "
              f"{n:>7} {same:>8}{mark}")
    print("\nSi alguna fila tiene iguales == ambos, ese par es el mismo frame\n"
          "de cinta y sus bloques SI se pueden combinar.")


def scan_mode(A, B, V, f):
    """Busca un frame concreto del base por todo el donante."""
    S = V + 3
    D = V[:, None] + np.arange(3, BLOCK)[None, :]
    fa = np.array(A[f])
    ga = (fa[S] >> 4) == 0
    print(f"Buscando el frame {f} del base entre los {B.shape[0]} del donante.")
    print(f"Bloques sanos en el base: {int(ga.sum())} de {len(V)}\n")
    best = []
    for d in range(B.shape[0]):
        fb = np.array(B[d])
        both = ga & ((fb[S] >> 4) == 0)
        n = int(both.sum())
        if n < 20:
            continue
        idx = D[both]
        same = int((fa[idx] == fb[idx]).all(axis=1).sum())
        if same:
            best.append((same / n, same, n, d))
    best.sort(reverse=True)
    if not best:
        print("Ni un solo bloque de este frame aparece en NINGUN frame del")
        print("donante. Este trozo de cinta no esta en esa pasada.")
        return
    print(f"{'donante':>8} {'ambos':>7} {'iguales':>8} {'%':>7}")
    for r, same, n, d in best[:10]:
        mark = "  <-- MISMO FRAME" if same == n else ""
        print(f"{d:>8} {n:>7} {same:>8} {100*r:>6.1f}%{mark}")
    if best[0][1] != best[0][2]:
        print("\nNingun frame del donante coincide del todo: este frame del")
        print("base no esta en esa pasada.")


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    base_p, don_p = args[0], args[1]
    off = int(args[2]) if len(args) > 2 else 14
    FS = 144000
    V = layout(); S = V + 3
    D = V[:, None] + np.arange(3, BLOCK)[None, :]
    A = np.memmap(base_p, dtype=np.uint8, mode='r',
                  shape=(os.path.getsize(base_p)//FS, FS))
    B = np.memmap(don_p, dtype=np.uint8, mode='r',
                  shape=(os.path.getsize(don_p)//FS, FS))
    if '--find' in sys.argv:
        return find_mode(A, B, V)
    if '--scan' in sys.argv:
        i = sys.argv.index('--scan')
        return scan_mode(A, B, V, int(sys.argv[i+1]))
    if '--pair' in sys.argv:
        i = sys.argv.index('--pair')
        return pair_mode(A, B, V, int(sys.argv[i+1]), int(sys.argv[i+2]))
    print(f"base {A.shape[0]} frames, donante {B.shape[0]} frames, desfase {off}\n")
    print(f"{'donante':>8} {'base':>7} {'sanos A':>8} {'sanos B':>8} {'ambos':>7} {'iguales':>8}")
    tot_both = tot_same = 0
    for d in range(0, min(B.shape[0], 1400), 100):
        f = d + off
        if not (0 <= f < A.shape[0]):
            continue
        fa, fb = np.array(A[f]), np.array(B[d])
        ga = (fa[S] >> 4) == 0
        gb = (fb[S] >> 4) == 0
        both = ga & gb
        n = int(both.sum())
        same = int((fa[D[both]] == fb[D[both]]).all(axis=1).sum()) if n else 0
        tot_both += n; tot_same += same
        print(f"{d:>8} {f:>7} {int(ga.sum()):>8} {int(gb.sum()):>8} {n:>7} {same:>8}")
    print(f"\ntotal: {tot_both} posiciones con ambos sanos, {tot_same} iguales", end="")
    if tot_both:
        print(f"  ({100.0*tot_same/tot_both:.1f}%)")
    else:
        print()

if __name__ == '__main__':
    sys.exit(main())
