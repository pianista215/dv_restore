#!/usr/bin/env python3
"""dvr - restauracion de MiniDV combinando varias capturas de la misma cinta.

  dvr.py calib FICHERO            calibra el barajado y verifica el parser
  dvr.py scan FICHERO [...]       daño y bloques falsamente sanos
  dvr.py clip --ref F --from N --count M FICHERO...   recortes alineados
  dvr.py png FICHERO --frames A-B curso de frames a PNG
  dvr.py map FICHERO --frames A-B mapa de daño sobre la imagen
  dvr.py compare A.dv B.dv --frames A-B   comparativa lado a lado
  dvr.py index FICHERO...         construye el indice de frames de cinta
  dvr.py merge ...                fusiona (ver dvr.py merge -h)

Todas las rutas de salida van a --out (por defecto work/).
"""

import argparse
import os
import sys

import numpy as np


def _prof(caps):
    return caps[0].prof


def _range(s, n):
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), min(int(b) + 1, n)))
    return [int(s)]


def cmd_calib(args):
    from dvr.dvfile import Capture
    from dvr import shuffle, native, render, dcplane
    from dvr.bitstream import parse_segment, pack_segment
    cap = Capture(args.file)
    prof = cap.prof

    # 1) el parser tiene que devolver los bytes exactos
    print("round-trip del parser de bitstream...")
    tot = ok = fail = 0
    for fi in np.linspace(0, cap.n - 1, args.frames).astype(int):
        f = cap.frame(fi)
        st = dcplane.sta(f, prof).reshape(prof.n_seg, 5)
        for s in range(prof.n_seg):
            if st[s].any():
                continue
            off = int(prof.seg[s, 0])
            buf = f[off:off + 400].tobytes()
            tot += 1
            seg = native.parse(buf)
            if not seg.ok:
                fail += 1
                continue
            out, lost = native.pack(seg, buf)
            ok += out == buf and lost == 0
    print(f"  {ok}/{tot} segmentos exactos ({100*ok/max(tot,1):.3f}%), "
          f"{fail} con bitstream invalido pese a estar marcados sanos")
    if ok + fail != tot:
        print("  AVISO: hay segmentos que no vuelven a salir iguales")

    # 2) barajado
    if args.shuffle:
        print("\ncalibrando el barajado de macrobloques...")
        base = cap.frame(args.ref_frame)
        table, _ = shuffle.calibrate(base, prof)
        assert len(np.unique(table)) == prof.n_video
        shuffle.save(table, prof)
        print("  tabla guardada en dvr/data/")

    # 3) comprobacion independiente del barajado
    table = shuffle.load(prof)
    print("\nverificando el barajado con un bloque al azar...")
    rng = np.random.default_rng(7)
    frame = cap.frame(args.ref_frame)
    flat = shuffle._flat_frame(frame, prof)
    b0 = render.decode(np.frombuffer(flat, np.uint8), prof, "y")[0]
    bad = 0
    for _ in range(args.probes):
        i = int(rng.integers(prof.n_video))
        img = render.decode(np.frombuffer(shuffle._mark(flat, prof, [i]), np.uint8),
                            prof, "y")[0]
        ch = shuffle._changed_mbs(img, b0, prof)
        pos = int(table[i])
        want = np.zeros_like(ch)
        want.ravel()[pos] = True
        if not np.array_equal(ch, want):
            bad += 1
            print(f"  bloque {i}: esperaba el macrobloque {pos//prof.mb_cols},"
                  f"{pos%prof.mb_cols}; cambian {int(ch.sum())}")
    print(f"  {args.probes - bad}/{args.probes} bloques caen exactamente donde dice la tabla")
    return 0 if bad == 0 else 1


def cmd_scan(args):
    from dvr.dvfile import Capture
    from dvr import native, dcplane, maps
    for path in args.files:
        cap = Capture(path)
        prof = cap.prof
        idx = np.linspace(0, cap.n - 1, min(args.frames, cap.n)).astype(int)
        nbad = nfalse = nseg_bad = nsus = 0
        per_frame = []
        for fi in idx:
            f = cap.frame(fi)
            sc = native.scan_frame(f, prof)
            state = maps.block_state(f, prof, sc)
            b = int((state == maps.BAD).sum())
            fa = int((state == maps.FALSE_OK).sum())
            nsus += int((state == maps.SUSPECT).sum())
            nbad += b
            nfalse += fa
            nseg_bad += int((sc.seg_ok == 0).sum())
            per_frame.append((fa, b, int(fi)))
        tot = len(idx) * prof.n_video
        print(f"\n{cap.name}: {cap.n} frames ({cap.n/25:.1f}s), "
              f"{len(idx)} muestreados")
        print(f"  macrobloques marcados con error : {nbad:8d}  ({100*nbad/tot:6.3f}%)")
        print(f"  macrobloques FALSAMENTE sanos   : {nfalse:8d}  ({100*nfalse/tot:6.3f}%)")
        print(f"  sanos en segmento contaminado   : {nsus:8d}  ({100*nsus/tot:6.3f}%)")
        per_frame.sort(reverse=True)
        worst = [p for p in per_frame if p[0]]
        if worst:
            print(f"  frames con falsos sanos: {len(worst)}/{len(idx)}   "
                  f"peores (frame:cuantos): "
                  + " ".join(f"{p[2]}:{p[0]}" for p in worst[:10]))
        else:
            print("  ningun falso sano detectado por validez de bitstream")
    return 0


def cmd_clip(args):
    from dvr.dvfile import Capture, clip
    from dvr.match import Matcher
    caps = [Capture(p) for p in args.files]
    names = [c.name for c in caps]
    ri = names.index(os.path.basename(args.ref))
    print("calculando huellas...")
    m = Matcher(caps, probes=args.probes)
    lo = [None] * len(caps)
    hi = [None] * len(caps)
    rng = range(args.start, min(args.start + args.count, caps[ri].n))
    print(f"buscando el tramo equivalente de {caps[ri].name} "
          f"[{rng.start}..{rng.stop-1}] en las demas capturas...")
    for f in rng:
        v = m.votes(ri, f)
        done = set()
        for (cj, g), _ in v.most_common(args.candidates):
            if cj in done:
                continue
            ok, _, _ = m.same_tape_frame(ri, f, cj, g)
            if ok:
                done.add(cj)
                lo[cj] = g if lo[cj] is None else min(lo[cj], g)
                hi[cj] = g if hi[cj] is None else max(hi[cj], g)
    os.makedirs(args.out, exist_ok=True)
    for ci, c in enumerate(caps):
        if ci == ri:
            a, n = rng.start, len(rng)
        elif lo[ci] is None:
            print(f"  {c.name}: sin solape con ese tramo, no se recorta")
            continue
        else:
            a = max(0, lo[ci] - args.margin)
            n = min(c.n - a, hi[ci] - a + 1 + args.margin)
        dst = os.path.join(args.out, f"{args.tag}_{os.path.splitext(c.name)[0]}.dv")
        got = clip(c, dst, a, n)
        print(f"  {c.name}: frames {a}..{a+got-1} ({got}, {got/25:.1f}s) -> {dst}")
    return 0


def cmd_png(args):
    from dvr.dvfile import Capture
    from dvr import render, sheet
    cap = Capture(args.file)
    os.makedirs(args.out, exist_ok=True)
    for fi in _range(args.frames, cap.n):
        img = render.decode(cap.frame(fi), cap.prof, "rgb")[0]
        p = os.path.join(args.out, f"{args.tag}{fi:06d}.png")
        sheet.save(img, p)
        print(p)
    return 0


def cmd_map(args):
    from dvr.dvfile import Capture
    from dvr import render, maps, sheet, native
    cap = Capture(args.file)
    prof = cap.prof
    table = None
    os.makedirs(args.out, exist_ok=True)
    for fi in _range(args.frames, cap.n):
        f = cap.frame(fi)
        sc = native.scan_frame(f, prof)
        state = maps.block_state(f, prof, sc)
        grid = maps.to_grid(state, prof, table)
        img = render.decode(f, prof, "rgb")[0]
        over = maps.overlay(img, grid, prof)
        p = os.path.join(args.out, f"{args.tag}map{fi:06d}.png")
        sheet.save(sheet.strip([img, over],
                               [f"{cap.name} frame {fi}",
                                maps.legend_text(state)],
                               scale=args.scale), p)
        print(f"{p}   {maps.legend_text(state)}")
    return 0


def cmd_index(args):
    from dvr.dvfile import Capture
    from dvr.match import build_index, save_index
    caps = [Capture(p) for p in args.files]
    for c in caps:
        print(f"  {c}")
    idx = build_index(caps, probes=args.probes, min_audio=args.min_audio,
                      candidates=args.candidates)
    st = idx.stats
    print(f"\nlecturas totales      : {st['total_reads']}")
    print(f"frames de cinta       : {st['clusters']}  ({st['clusters']/25:.1f}s)")
    print(f"  con mas de una lectura: {st['shared']}")
    print(f"  con lecturas repetidas de la misma captura: {st['conflicts']}")
    for ci, c in enumerate(caps):
        solo = sum(1 for g in idx.clusters if len(g) == 1 and g[0][0] == ci)
        print(f"  solo en {c.name}: {solo} frames")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_index(idx, args.out)
    print(f"\nindice guardado en {args.out}")
    return 0


def cmd_merge(args):
    from dvr.dvfile import Capture, write_frames
    from dvr.match import load_index
    from dvr import merge as mg
    import time
    idx = load_index(args.index)
    prof = idx.caps[0].prof
    lo, hi = 0, len(idx)
    if args.range:
        a, b = args.range.split("-")
        lo, hi = int(a), min(int(b) + 1, len(idx))
    print(f"fundiendo frames de cinta {lo}..{hi-1} de {len(idx)}")
    tot_before = tot_after = added = 0
    multi_before = multi_after = multi_n = 0
    agg = {}
    t0 = time.time()
    with open(args.out, "wb") as fh:
        for k in range(lo, hi):
            reads = [mg.Read(idx.caps[c].frame(f), prof, c, f)
                     for c, f in idx.reads(k)]
            out, info = mg.merge_frame(reads, prof)
            best = min(r.n_bad for r in reads)
            tot_before += best
            tot_after += info["unresolved"]
            if len(reads) == 1:
                added += 1
            else:
                multi_n += 1
                multi_before += best
                multi_after += info["unresolved"]
            for key in ("kept", "copied", "repacked", "lost", "bad_pack"):
                agg[key] = agg.get(key, 0) + info[key]
            fh.write(out.tobytes())
            if args.verbose and (k - lo) % 100 == 0:
                print(f"  {k-lo}/{hi-lo}...")
    n = hi - lo
    tot = n * prof.n_video
    print(f"\nframes escritos                : {n}  ({n/25:.1f}s)")
    print(f"  con una sola lectura         : {added}")
    print(f"segmentos sin tocar            : {agg['kept']}")
    print(f"segmentos copiados enteros     : {agg['copied']}")
    print(f"segmentos vueltos a empaquetar : {agg['repacked']}")
    print(f"coeficientes recortados por falta de sitio: {agg['lost']}")
    print(f"segmentos que no cupieron: {agg['bad_pack']}  "
          f"{'(debe ser 0)' if agg['bad_pack'] else 'OK'}")
    mt = multi_n * prof.n_video
    print(f"\n--- frames con MAS DE UNA lectura ({multi_n}) ---")
    if mt:
        print(f"  malos en la mejor lectura : {multi_before:8d}  ({100*multi_before/mt:6.3f}%)")
        print(f"  sin resolver tras fundir  : {multi_after:8d}  ({100*multi_after/mt:6.3f}%)")
        if multi_before:
            print(f"  recuperado                : {100*(multi_before-multi_after)/multi_before:.1f}%")
    st = (added) * prof.n_video
    print(f"--- frames con UNA sola lectura ({added}) ---")
    if st:
        print(f"  malos, sin nada que aportar: {tot_before-multi_before:8d}  "
              f"({100*(tot_before-multi_before)/st:6.3f}%)")
    print(f"\nTOTAL macrobloques malos   : {tot_before:8d}  ({100*tot_before/tot:6.3f}%)")
    print(f"TOTAL sin resolver         : {tot_after:8d}  ({100*tot_after/tot:6.3f}%)")
    if tot_before:
        print(f"recuperado                 : {100*(tot_before-tot_after)/tot_before:.1f}%")
    print(f"\n{time.time()-t0:.0f}s   salida: {args.out}")
    return 0


def cmd_conceal(args):
    from dvr.dvfile import Capture
    from dvr import conceal
    import time
    cap = Capture(args.file)
    prof = cap.prof
    before = sum(cap.n_bad(i) for i in range(cap.n))
    t0 = time.time()
    out, rep = conceal.conceal_sequence(
        cap.data, prof, max_dist=args.max_dist,
        thr=args.threshold, min_nb=args.min_neighbours,
        progress=lambda i, n: print(f"  {i}/{n}...") if args.verbose else None)
    with open(args.out, "wb") as fh:
        for f in out:
            fh.write(f.tobytes())
    after = sum(int((f[prof.sta] >> 4 != 0).sum()) for f in out)
    tot = cap.n * prof.n_video
    print(f"macrobloques rotos antes  : {before:8d}  ({100*before/tot:6.3f}%)")
    print(f"  tapados por copia temporal : {rep['filled']:8d}")
    print(f"  dejados: la zona se mueve  : {rep['motion_reject']:8d}")
    print(f"  dejados: ningun frame vecino lo tiene sano : {rep['no_source']:8d}")
    print(f"  dejados: sin vecinos sanos con que medir   : {rep['no_support']:8d}")
    w = rep["win_used"]
    print(f"  ventana usada (radio 3/6/12/frame): {w[0]}/{w[1]}/{w[2]}/{w[3]}")
    d = rep["diffs"]
    if len(d):
        print(f"  diferencia de DC medida: mediana {np.median(d):.1f}  "
              f"p25 {np.percentile(d,25):.1f}  p75 {np.percentile(d,75):.1f}")
    print(f"segmentos copiados enteros : {rep['seg_copied']}, "
          f"repaquetizados: {rep['seg_repacked']}")
    print(f"coeficientes recortados por falta de sitio: {rep['trimmed']}")
    print(f"segmentos escritos que NO vuelven a parsear: {rep['invalid_written']}"
          f"  {'(debe ser 0)' if rep['invalid_written'] else 'OK'}")
    print(f"macrobloques rotos despues: {after:8d}  ({100*after/tot:6.3f}%)")
    if before:
        print(f"reduccion                 : {100*(before-after)/before:.1f}%")
    print(f"\n{time.time()-t0:.0f}s   salida: {args.out}")
    return 0


def cmd_calconceal(args):
    """Calibra la puerta de movimiento contra verdad de campo.

    Toma macrobloques que estan SANOS, hace como si estuvieran rotos, aplica la
    misma regla de copia y compara con el original. Asi el umbral deja de ser
    una corazonada y pasa a ser una eleccion medida.
    """
    from dvr.dvfile import Capture
    from dvr import dcplane, shuffle
    from dvr.conceal import FrameInfo, PairMotion, WINDOWS, WIN_FACTOR
    cap = Capture(args.file)
    prof = cap.prof
    n = cap.n
    table = shuffle.load(prof)
    pos = np.asarray(table, np.int64)
    rows, cols = pos // prof.mb_cols, pos % prof.mb_cols
    info = [FrameInfo(cap.frame(i), prof, table) for i in range(n)]
    dcs = [dcplane.dc_raw(cap.frame(i), prof)[:, :4].mean(axis=1) for i in range(n)]
    rng = np.random.default_rng(3)
    rec = []
    for i in range(2, n - 2):
        ok = np.nonzero(~info[i].bad)[0]
        if not len(ok):
            continue
        sel = rng.choice(ok, size=min(args.per_frame, len(ok)), replace=False)
        pm = {}
        for k in sel:
            r, c = int(rows[k]), int(cols[k])
            best = None
            for d in range(1, args.max_dist + 1):
                for j in (i - d, i + d):
                    if not (0 <= j < n) or info[j].bad[k]:
                        continue
                    if j not in pm:
                        pm[j] = PairMotion(info[i], info[j], prof)
                    for w, (win, fac) in enumerate(zip(WINDOWS, WIN_FACTOR)):
                        dd, cnt = pm[j].at(r, c, win)
                        if dd is None or cnt < 6:
                            continue
                        sc = dd / fac
                        if best is None or sc < best[0]:
                            best = (sc, j)
                        break
                if best is not None:
                    break
            if best is None:
                continue
            rec.append((best[0], abs(float(dcs[i][k]) - float(dcs[best[1]][k]))))
    rec = np.array(rec)
    sc, err = rec[:, 0], rec[:, 1]
    print(f"muestras con verdad de campo: {len(rec)}")
    print("\numbral  se tapa   err mediana   err p90   copias con err>20")
    for t in (2, 3, 4, 5, 6, 8, 10, 12, 16):
        m = sc <= t
        if m.sum() < 20:
            continue
        print(f"{t:6d}  {100*m.mean():6.1f}%  {np.median(err[m]):11.1f}  "
              f"{np.percentile(err[m],90):8.1f}  {100*(err[m]>20).mean():14.1f}%")
    print("\n(unidades de DC; 1 DC equivale a ~0,58 niveles de luma)")
    return 0


def cmd_triage(args):
    from dvr.dvfile import Capture
    from dvr.match import triage
    base = Capture(args.base)
    recs = triage(base, args.donors, probes=args.probes,
                  min_audio=args.min_audio, candidates=args.candidates)
    useful = [r for r in recs if r["matched"] >= args.min_matched]
    print(f"\n--- resumen ---")
    print(f"donantes con solape util (>= {args.min_matched} frames): "
          f"{len(useful)} de {len(recs)}")
    tot = sum(r["matched"] for r in useful)
    print(f"frames de donante que casan con la base: {tot}")
    if useful:
        lo = min(r["base_lo"] for r in useful)
        hi = max(r["base_hi"] for r in useful)
        print(f"tramo de la base cubierto: {lo}..{hi} "
              f"({(hi-lo+1)/25:.0f}s de {base.n/25:.0f}s)")
    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(recs, fh, indent=1)
        print(f"\ndetalle en {args.out}")
    return 0


def cmd_window(args):
    from dvr.dvfile import Capture
    from dvr.match import window_clips
    base = Capture(args.base)
    made = window_clips(base, args.donors, args.start,
                        args.start + args.count - 1, args.out, args.tag,
                        margin=args.margin, probes=args.probes,
                        min_audio=args.min_audio, candidates=args.candidates,
                        min_matched=args.min_matched)
    tot = sum(os.path.getsize(p) for p in made)
    print(f"\n{len(made)} recortes, {tot/1e9:.2f} GB en {args.out}")
    return 0


def cmd_preview(args):
    """Comparativa en video: la captura original estirada a la linea temporal
    real de la cinta, al lado del resultado."""
    import subprocess
    from dvr.match import load_index, align_to_tape
    idx = load_index(args.index)
    names = [c.name for c in idx.caps]
    ci = next((i for i, n in enumerate(names) if args.capture in n), None)
    if ci is None:
        print(f"no encuentro '{args.capture}' entre: {', '.join(names)}")
        return 1
    tmp = os.path.splitext(args.out)[0] + "_alineado.dv"
    missing = align_to_tape(idx, ci, tmp)
    print(f"{names[ci]} estirado a la cinta: {len(idx)} frames, "
          f"{missing} congelados porque esa captura no los tiene")
    lab = ["drawtext=text='%s':x=12:y=h-34:fontsize=22:fontcolor=white:"
           "box=1:boxcolor=black@0.6:boxborderw=6" % t
           for t in ("ORIGINAL  (congela donde perdio frames)", "RESTAURADO")]
    fc = (f"[0:v]setpts=N/25/TB,{lab[0]}[a];"
          f"[1:v]setpts=N/25/TB,{lab[1]}[b];[a][b]hstack=inputs=2[v]")
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "dv", "-i", tmp, "-f", "dv", "-i", args.result,
           "-filter_complex", fc, "-map", "[v]", "-map", "1:a:0",
           "-c:v", "libx264", "-crf", str(args.crf), "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-shortest", args.out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        print(r.stderr.decode(errors="replace")[-1500:])
        return 1
    if not args.keep:
        os.remove(tmp)
    print(f"{args.out}  ({os.path.getsize(args.out)/1e6:.0f} MB)")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="dvr.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calib", help="calibra el barajado y verifica el parser")
    p.add_argument("file")
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--ref-frame", type=int, default=0)
    p.add_argument("--probes", type=int, default=8)
    p.add_argument("--shuffle", action="store_true", help="recalibra el barajado")
    p.set_defaults(func=cmd_calib)

    p = sub.add_parser("scan", help="daño y bloques falsamente sanos")
    p.add_argument("files", nargs="+")
    p.add_argument("--frames", type=int, default=400)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("triage",
                       help="que donantes solapan con la base, y en que tramo")
    p.add_argument("base")
    p.add_argument("donors", nargs="+")
    p.add_argument("--probes", type=int, default=162)
    p.add_argument("--min-audio", type=int, default=20)
    p.add_argument("--candidates", type=int, default=40)
    p.add_argument("--min-matched", type=int, default=25)
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_triage)

    p = sub.add_parser("window",
                       help="recorta un tramo de la base y el equivalente de cada donante")
    p.add_argument("base")
    p.add_argument("donors", nargs="+")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=750)
    p.add_argument("--margin", type=int, default=30)
    p.add_argument("--probes", type=int, default=162)
    p.add_argument("--min-audio", type=int, default=20)
    p.add_argument("--candidates", type=int, default=40)
    p.add_argument("--min-matched", type=int, default=15)
    p.add_argument("--tag", default="w")
    p.add_argument("--out", default="clips")
    p.set_defaults(func=cmd_window)

    p = sub.add_parser("clip", help="recortes alineados de varias capturas")
    p.add_argument("files", nargs="+")
    p.add_argument("--ref", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=250)
    p.add_argument("--margin", type=int, default=25)
    p.add_argument("--probes", type=int, default=162)
    p.add_argument("--candidates", type=int, default=40)
    p.add_argument("--tag", default="clip")
    p.add_argument("--out", default="clips")
    p.set_defaults(func=cmd_clip)

    p = sub.add_parser("index", help="construye el indice de frames de cinta")
    p.add_argument("files", nargs="+")
    p.add_argument("--probes", type=int, default=162)
    p.add_argument("--min-audio", type=int, default=20)
    p.add_argument("--candidates", type=int, default=40)
    p.add_argument("--out", default="work/index.json")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("merge", help="funde las capturas usando el indice")
    p.add_argument("--index", default="work/index.json")
    p.add_argument("--out", required=True)
    p.add_argument("--range")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("conceal", help="tapa lo que quedo roto con frames vecinos")
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.add_argument("--max-dist", type=int, default=6)
    p.add_argument("--threshold", type=float, default=8.0)
    p.add_argument("--min-neighbours", type=int, default=6)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_conceal)

    p = sub.add_parser("calconceal",
                       help="calibra la puerta de movimiento con verdad de campo")
    p.add_argument("file")
    p.add_argument("--per-frame", type=int, default=120)
    p.add_argument("--max-dist", type=int, default=6)
    p.set_defaults(func=cmd_calconceal)

    p = sub.add_parser("preview",
                       help="video comparativo original / restaurado")
    p.add_argument("result")
    p.add_argument("--index", required=True)
    p.add_argument("--capture", default="parte1",
                   help="que captura se pone a la izquierda")
    p.add_argument("--out", required=True)
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("png", help="exporta frames a PNG")
    p.add_argument("file")
    p.add_argument("--frames", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="work")
    p.set_defaults(func=cmd_png)

    p = sub.add_parser("map", help="mapa de daño sobre la imagen")
    p.add_argument("file")
    p.add_argument("--frames", required=True)
    p.add_argument("--scale", type=float, default=0.6)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="work")
    p.set_defaults(func=cmd_map)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
