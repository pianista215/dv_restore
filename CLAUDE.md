# CLAUDE.md

Guidance for working in this repository.

## What this is

Tools to restore a degraded MiniDV tape by combining several capture passes of
the same material. It is not a player and not a transcoder: it operates on the
raw DV stream, at macroblock level.

## Rules that are not bent

- **The source `.dv` files are sacred.** Always opened with
  `np.memmap(..., mode="r")`. Never written, moved or deleted.
- **No video material in the repository.** `.gitignore` excludes `*.dv`,
  `*.mp4`, `*.png`, and the `clips/`, `work/` and `out/` directories. If you add
  a new output, add it to `.gitignore` too.
- **Every segment that gets written must parse again.** That is the invariant
  that keeps black stripes out of the picture. If the coefficients do not fit
  when merging, the highest frequencies are trimmed while the EOB is preserved
  (`dvr/fixup.py`); never let the packer lose the end of a block.
- **Measure before you claim.** Every decision in this repo came out of a
  measurement, and several reasonable-sounding hypotheses turned out to be false
  once measured. If you change a threshold, show the number that justifies it.

## Architecture, one line each

- `dvr/layout.py` — DV frame geometry. The unit of work is the **video
  segment** (5 macroblocks, 400 bytes), not the 80-byte DIF block: the five
  share the VLC overflow, so touching one breaks the other four.
- `dvr/bitstream.py` — segment parser and repacker. Reference implementation in
  Python, verified byte-for-byte by round-trip.
- `dvr/_native/dvbits.c` + `dvr/native.py` — the same thing in C, 85x faster.
  Compiles itself with gcc on first use and caches the `.so` in `work/`.
- `dvr/dcplane.py` — DC, DCT mode, class, STA and QNO, vectorized, without
  decoding. The DC sits at a fixed bit position, so it comes for free.
- `dvr/shuffle.py` — stream block -> screen macroblock, deduced from the
  decoder itself.
- `dvr/match.py` — N-way matching and the tape-frame index.
- `dvr/merge.py` — per-macroblock merging. `dvr/conceal.py` — concealment.
- `dvr/refvideo.py` — an optional external reference (a DVD of the same tape).
  Everything that knows about MPEG-2 lives here; the rest of the repo only ever
  sees `ref(i, k) -> 16x16 luma | None`.
- `dvr/pipeline.py` — the full windowed run, resumable.

## Things worth knowing about this material

- **The camera already conceals errors** before it outputs video, and marks what
  it concealed with `STA = 0xE`. A marked block is not garbage: it is the
  camera's estimate. What you measure is not "how many blocks are broken" but
  **how many carry real data and how many carry an estimate**.
- **The damage has period 5.** Match probes are drawn from a fixed random seed,
  never at a constant stride: a stride that is a multiple of 5 lands every
  single probe on a position that is always broken.
- **DIF IDs differ between captures** even when the content is identical. When
  you copy a block you have to put them back.
- **Asking ffmpeg for `gray` expands the studio range.** Measured on both
  paths: `gray = 1.1348*Y - 14.95`, i.e. 16-235 stretched to 0-255. On DV,
  decode `yuv420p` and slice the Y plane (`render.decode(..., "yraw")`); on
  MPEG-2, use `-vf extractplanes=y`, which is byte-identical to the Y plane.
  `refvideo.check_range_safe()` asserts this rather than trusting it, because
  the same trap has already cost an hour once.
- **One capture cannot read the same tape frame twice.** The union-find refuses
  those unions: accepting them creates cycles in the ordering graph, and one
  early cycle scrambles everything behind it.

## Things that seemed like a good idea and do NOT work

Measured and discarded. Do not retry them without reading this.

- **Choosing the copy source by ECC structure.** The damage on these tapes
  always hits the same position within the video segment, and the shuffle turns
  that into a 9-column band crossing the screen: m=3 is columns 0-8, m=1 is
  9-17, m=0 is 18-26, m=2 is 27-35 and m=4 is 36-44. When the failing position
  drifts, the band is born in the centre and walks right until it leaves the
  frame. It explains the artifact perfectly, but preferring sources whose band
  is elsewhere makes things **worse**: median error 2.2 -> 3.0 where it changes
  the choice, worse in 52% of cases. Structurally clean sources are further away
  in time, and that costs more than it gains. Using it only to break ties
  between candidates at equal distance is worse still (mean 4.6 -> 9.3).
  **Temporal proximity dominates.**

- **Stabilizing the canvas between consecutive frames.** The canvas jumped from
  one capture to another in 31% of transitions, and the mean difference was
  larger when it changed (11.3 against 7.5). It looked causal and it was not:
  bringing it down to 8% did not move a single jump. The canvas changes where
  there is more damage, and there is more damage where the picture moves. It was
  kept because it keeps subcode and audio coherent, not because it fixes
  anything.

- **Majority voting between healthy reads that disagree.** Across eight passes
  only 391 blocks out of 792,087 disagree (0.049%), all with a clear majority:
  0.3 blocks per frame. Invisible. The garbage that remains is identical in all
  eight passes, which means it is written on the tape.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

They are fast and they cover the invariants: the VLC table, the parser
round-trip, the C agreeing with the Python, trimming when there is no room, and
the ordering (gaps, cycles, unlinked chains, same-capture unions).

Before calling a change to the ordering or the merging good, on top of the
tests: pull a strip of consecutive frames and **look at them**. Global numbers
hide local disorder; both times this repo was wrong, it was the eye that caught
it, not the metric.
