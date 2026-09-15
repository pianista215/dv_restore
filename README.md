# dv_restore — restoring MiniDV by combining several captures

Tools to combine several capture passes of the same degraded MiniDV tape and
keep the best of each one.

The starting point: **two correct reads of the same tape frame produce
byte-identical video blocks**. That makes it possible to match frames across
captures without relying on timecode (invalid on these tapes) or on `abst`.

## What we found out about this material

Everything below is measured on the real files, not assumed.

**The camera conceals errors before it outputs video.** A block marked
`STA = 0xE` does not contain garbage: it contains the camera's estimate, usually
the same block from the previous frame. That is why a frame with 1606 of its
1620 macroblocks marked still looks fine. What has to be measured is not "how
many blocks are broken" but **how many carry real data and how many carry an
estimate**.

Measured against another capture used as ground truth, the error of the
camera's concealment is: median 14 DC units, p90 106, and **23% of blocks
deviate by more than 50** (about 29 luma levels). That is plainly visible
damage: it keeps a stale frame and the motion is lost.

**The damage on this tape has period 5.** Within each video segment,
macroblocks 1 and 3 almost always survive while 0, 2 and 4 die. Practical
consequence: choosing match probes at a constant stride that is a multiple of 5
puts every probe on a position that is always broken, and you get **not one
vote** even when the two frames share hundreds of identical blocks. Probes are
drawn from a fixed random seed.

**VLC overflow forces you to work segment by segment.** The 5 macroblocks of a
segment share the spare space: replacing a single 80-byte DIF block can break
the other four. That is why there is a full segment parser and repacker.

**DIF IDs differ between captures** (all 1620 of them, byte 0: 147 against 148)
even when the content is identical. When you copy, you have to put them back.

## What each piece contributes, measured

Full run over `parte1`, all passes:

| | |
|---|---|
| Frames | 7913 (316.5 s) -> **11846 (473.8 s)**, +3933 recovered |
| Macroblocks that are camera invention | **14.55% -> 0.22%** |
| Macroblocks we corrupt while repacking | **0** |

On a 10 s clip of the worst-damaged stretch (296 tape frames):

| | |
|---|---|
| Merging: macroblocks going from estimate to real data | **18.9%** |
| Timeline reconstruction: frames the base never captured | **103** (+4.1 s of 12) |
| Merging: macroblocks left unresolved | 8.15%, which is **exactly the theoretical minimum** |

Merging is **optimal**: it leaves unresolved exactly the macroblocks that are
concealed in every capture, not one more.

Concealment carries the rest. Plain temporal copy is barely better than what the
camera already did, because where the camera is right, copying agrees with it,
and where the camera is wrong there is motion. What actually moved the needle
was, in order: **global pan compensation** when copying a macroblock from
another frame (median error 5.68 with the motion vector sign inverted, 3.97 with
no compensation at all, **2.41** with the sign right — hence the regression
test), and **motion-compensated temporal interpolation** between the nearest
healthy frames on each side.

The honest limit: our luma block-seam metric is 1.217 against 1.030 for the
original and 1.156 for a clean capture. Our content is accurate but distinct
where the camera's is smooth but wrong. `dvr/interp.py` has a `feather()` that
takes the edge jump from 3.35 down to 1.94 at the cost of raising the error from
2.35 to 2.90; it is implemented and deliberately not wired in, because that is a
taste decision rather than a correctness one.

## How to run it

Requirements: Python 3 with numpy and Pillow, ffmpeg and gcc. Nothing else; the
C accelerator builds itself on first use.

### The normal case: restore one capture using all the others

```bash
# 1. which donors really overlap, and over what stretch (read-only, writes nothing)
python3 dvr.py triage BAD.dv OTHERS*.dv --out work/triage.json

# 2. the full run. It is resumable: if it stops, relaunch the same command
python3 dvr.py runall BAD.dv OTHERS*.dv \
        --out /path/with/room/work \
        --final /path/with/room/restored.dv

# 3. video comparison: the original stretched onto the real tape timeline
#    (freezing where it lost frames) next to the result
python3 dvr.py preview restored.dv --index work/w.json --out comparison.mp4
```

`runall` splits the base into 750-frame windows with a step of 600 and stitches
through the middle: a single window with every donor fits in neither memory nor
time. Each window leaves its stretch in `<out>/cores/`, so finished work is not
repeated. `--budget SECONDS` stops cleanly so the run can be chopped up.

### Optional: a DVD of the same tape as a reference

If the tape was recorded to DVD years ago, back when it still read well, that
recording is the only source of genuinely **new** information left: everything
else is synthesis from reads that are already degraded. It never replaces real
tape data, only macroblocks that have none in any pass.

```bash
# one sequential decode of the whole video to raw luma (~600 MB per minute)
python3 dvr.py refextract DVD.VOB --out /path/with/room/work

# align it to the tape, once, and cache the map
python3 dvr.py refalign BAD.dv --ref /path/with/room/work

# then just add --ref to the normal run
python3 dvr.py runall BAD.dv OTHERS*.dv --ref /path/with/room/work \
        --out /path/with/room/work --final /path/with/room/restored.dv
```

`refalign` reports how much of the tape the recording actually covers, which is
the number that decides whether any of this is worth it. On the disc this was
built for: 66.8% covered cleanly, 6.2% doubtful, 27% not covered at all.

What makes the two sources worth combining is that they fail differently. A
DVD transfer has roughly constant quality whatever the picture is doing; our
temporal copy is better than it on still content and collapses on movement.
Measured against ground truth, per macroblock, after the full write round-trip:

| conceal's motion score | reference | copy from a neighbour | reference wins |
|---|---|---|---|
| 0-2 | 2.75 | 2.68 | 54% |
| 2-5 | 3.01 | 3.78 | 69% |
| 5-10 | 3.07 | 4.86 | 81% |
| 10-20 | 3.37 | 7.46 | 87% |
| 20-40 | 3.53 | 10.85 | 91% |

The reference is only consulted above a score of 2, and every patch has to pass
a local trust gate: its disagreement with the **healthy** macroblocks around it,
in the same frame. A DVD recorded over an analog connection carries lines and
dirt of its own, and where it does, it disagrees with the tape beside it and is
rejected there and then. Taking the reference unconditionally is worse than not
using it at all (median error 4.61 against 3.65); with the gate it is 3.12,
where the best possible choice every time would be 2.96.

### For iterating and debugging

```bash
# verify the parser and calibrate the shuffle (once per profile)
python3 dvr.py calib capture.dv --shuffle

# cut one window with all its donors, to iterate in seconds
python3 dvr.py window BAD.dv OTHERS*.dv --start 1550 --count 750 --out clips

# see the state of things
python3 dvr.py scan clips/*.dv
python3 dvr.py map clips/w_BAD.dv --frames 21   # damage map over the picture

# step by step
python3 dvr.py index clips/w_*.dv --out work/w.json
python3 dvr.py merge --index work/w.json --out out/merged.dv
python3 dvr.py calconceal out/merged.dv         # pick the threshold from data
python3 dvr.py conceal out/merged.dv --out out/final.dv --threshold 8
```

**Everything large goes to `--out`.** The source `.dv` files are opened
read-only and are never touched.

## Invariants that are checked

- `parse` + `pack` of a segment returns **the exact same 400 bytes**. Verified
  over 19,542 segments from two captures: 100% of the valid ones.
- The shuffle table is deduced from the decoder itself, using bit planes over a
  synthetic grey frame, and verified block by block.
- Every segment that gets written **parses again**. If the coefficients do not
  fit when merging, the highest frequencies are trimmed and the EOB is kept;
  skipping this leaves black stripes in the picture.

## Layout

- `dvr/layout.py` — DV frame geometry.
- `dvr/dcplane.py` — DC, DCT mode, class, STA and QNO, vectorized, without
  decoding the bitstream. The DC sits at a fixed position, so it comes free.
- `dvr/bitstream.py` — segment parser and repacker (reference, Python).
- `dvr/_native/dvbits.c` + `dvr/native.py` — the same in C, 85x faster (7 ms per
  frame). Builds itself; falls back to the Python if gcc fails.
- `dvr/shuffle.py` — stream block -> screen macroblock.
- `dvr/match.py` — N-way matching and the tape-frame index.
- `dvr/merge.py` — merging by segment and by macroblock.
- `dvr/conceal.py` — temporal concealment with a motion gate.
- `dvr/interp.py` — motion estimation and compensated temporal interpolation.
- `dvr/pixels.py`, `dvr/encode.py` — DV to pixels and back, for the macroblocks
  we rebuild ourselves.
- `dvr/fixup.py` — sanitizing and trimming so the segment is always valid.
- `dvr/maps.py`, `dvr/sheet.py`, `dvr/render.py` — visual diagnostics.

The `dv_*.py` scripts at the root are the previous generation: one pair of files
at a time, working on 80-byte blocks. They are kept for reference.
