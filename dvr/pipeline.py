"""Proceso completo por ventanas, con costuras y resumible.

Una sola ventana no cabe: con 29 donantes serian 84 000 frames en un mismo
indice N-way, horas y varios GB. Se trocea la base en ventanas solapadas y cada
una se procesa entera (recorte, indice, fusion, ocultacion), pero solo se EMITE
su tramo central.

La costura sale exacta porque la base hace de columna vertebral: la ventana w
emite desde el frame de cinta que contiene el frame `s` de la base hasta el que
contiene `s + paso`, sin incluirlo; y la ventana siguiente arranca justo en ese.
Las dos contienen ese frame de la base, asi que las dos saben exactamente donde
esta y no se duplica ni se pierde nada.

Cada ventana escribe su tramo en un fichero aparte, asi que el proceso se puede
parar y retomar: lo ya hecho no se repite.
"""

import os
import time

import numpy as np

from . import conceal as cc
from . import merge as mg
from . import outliers
from . import shuffle
from .dvfile import Capture
from .match import build_index, window_clips


def _core_range(idx, cap_index, off_lo, off_hi, first, last):
    """Que posiciones del indice emite esta ventana.

    Ojo: en el indice la base es el RECORTE, asi que sus frames van de 0 a
    ancho-1, no con los numeros del fichero original. Los limites llegan ya en
    esa escala. Buscarlos con los numeros originales devolvia rangos absurdos y
    rompia las costuras.
    """
    n = len(idx)
    pos = {}
    for k in range(n):
        for c, f in idx.reads(k):
            if c == cap_index:
                pos[f] = k
    def at(target, default):
        if target in pos:
            return int(pos[target])
        later = [f for f in pos if f >= target]
        if later:
            return int(pos[min(later)])
        return default
    k0 = 0 if first else at(off_lo, 0)
    k1 = n if last else at(off_hi, n)
    if k1 <= k0:
        k1 = n
    return int(k0), int(k1)


def run_windows(base_path, donor_paths, out_dir, width=750, step=600,
                margin=30, thr=20.0, max_dist=25, hysteresis=0.20,
                budget=None, verbose=True, ref=None):
    base = Capture(base_path)
    prof = base.prof
    clips_dir = os.path.join(out_dir, "clips")
    cores_dir = os.path.join(out_dir, "cores")
    work_dir = os.path.join(out_dir, "work")
    for d in (clips_dir, cores_dir, work_dir):
        os.makedirs(d, exist_ok=True)

    starts = list(range(0, base.n, step))
    t0 = time.time()
    done = []
    for w, s in enumerate(starts):
        core = os.path.join(cores_dir, f"core_{w:03d}.dv")
        if os.path.exists(core):
            done.append(core)
            continue
        if budget and time.time() - t0 > budget:
            if verbose:
                print(f"\n[se agota el tiempo asignado; quedan "
                      f"{len(starts)-w} ventanas]")
            break
        lo, hi = s, min(s + width - 1, base.n - 1)
        first, last = (w == 0), (s + step >= base.n)
        if verbose:
            print(f"\n=== ventana {w+1}/{len(starts)}: base {lo}..{hi} ===")

        tag = f"w{w:03d}"
        for old in os.listdir(clips_dir):
            if old.startswith(tag + "_"):
                os.remove(os.path.join(clips_dir, old))
        made = window_clips(base, donor_paths, lo, hi, clips_dir, tag,
                            margin=margin, verbose=verbose)
        caps = [Capture(p) for p in made]
        idx = build_index(caps, verbose=False)
        if verbose:
            print(f"  {idx.stats['total_reads']} lecturas -> "
                  f"{idx.stats['clusters']} frames de cinta "
                  f"({idx.stats['shared']} con mas de una)")

        bi = next(i for i, c in enumerate(caps) if c.path == made[0])
        # limites en la escala del recorte: su frame 0 es el frame `lo` de la
        # base, asi que el nucleo va de 0 a `step`
        k0, k1 = _core_range(idx, bi, 0, step, first, last)

        frames = []
        nreads = []
        prev_canvas = None
        for k in range(len(idx)):
            reads = [mg.Read(idx.caps[c].frame(f), prof, c, f)
                     for c, f in idx.reads(k)]
            out, info = mg.merge_frame(reads, prof, prefer=prev_canvas,
                                       hysteresis=hysteresis)
            prev_canvas = info["canvas_cap"]
            frames.append(out)
            nreads.append(len(reads))

        # fuera los frames que no encajan donde han quedado, ANTES de ocultar:
        # asi la ocultacion ve vecinos buenos
        drop = outliers.find_misplaced(frames, nreads, prof)
        if drop:
            dset = set(drop)
            shift = np.cumsum([1 if i in dset else 0 for i in range(len(frames))])
            frames = [f for i, f in enumerate(frames) if i not in dset]
            k0 = int(k0 - (shift[k0 - 1] if k0 > 0 else 0))
            k1 = int(k1 - (shift[k1 - 1] if k1 > 0 else 0))
            if verbose:
                print(f"  descartados {len(drop)} frames fuera de sitio")
        # la referencia se reparte DESPUES del descarte: el mapa es un array
        # paralelo a `frames` y hay que filtrarlo igual, o la ventana entera
        # queda desplazada (invisible en lo quieto, demoledor en un barrido)
        refp = None
        if ref is not None:
            table = shuffle.load(prof)
            refp = ref.for_window(idx, bi, lo, prof, table, drop=drop)
            if refp is not None and len(refp.v) != len(frames):
                raise RuntimeError(
                    f"la referencia trae {len(refp.v)} frames y la ventana "
                    f"tiene {len(frames)}: el descarte no se ha aplicado igual")
        cleaned, rep = cc.conceal_sequence(frames, prof, max_dist=max_dist,
                                           thr=thr, ref=refp)
        if verbose and refp is not None:
            print(f"  DVD: escribe {rep['ref_written']} macrobloques, "
                  f"rescata {rep['ref_rescued_motion']} de los rechazados por "
                  f"movimiento, descarta {rep['ref_gated']} por desconfianza")
        tmp = core + ".part"
        with open(tmp, "wb") as fh:
            for k in range(k0, k1):
                fh.write(cleaned[k].tobytes())
        os.replace(tmp, core)
        done.append(core)
        if verbose:
            print(f"  emite posiciones {k0}..{k1-1} -> {os.path.basename(core)} "
                  f"({k1-k0} frames)   [{time.time()-t0:.0f}s]")
        for p in made[1:]:
            os.remove(p)
        del caps, idx, frames, cleaned
    return done, len(starts)


def concat(cores, out_path, prof):
    n = 0
    with open(out_path, "wb") as fh:
        for c in sorted(cores):
            with open(c, "rb") as g:
                while True:
                    b = g.read(prof.frame_size * 200)
                    if not b:
                        break
                    fh.write(b)
                    n += len(b) // prof.frame_size
    return n
