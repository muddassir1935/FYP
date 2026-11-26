# RQ3 — Temporal Consistency Buffer (FR-05): FP-Suppression Measurement

**Research question (PROJECT_CONTEXT §2):** *How does the implementation of a
sliding-window temporal buffer impact false-positive suppression rates in
real-time threat detection?*

> **Scope — read this first (re-scoped 23 Nov 2026).** This is a
> **component-level measurement of the FR-05 `TemporalConsistencyBuffer`**
> (window=10, required_hits=6, IoU 0.3), NOT a claim about the shipped demo.
> The demo path no longer uses this buffer: `src/pipeline/demo_worker.py`
> runs detection + ByteTrack tracking + class voting, and
> `scripts/inference/run_weapon_demo.py` confirms only while a track
> has `--min-hits` sightings (default 3) within the last
> `--confirm-window` seconds (default 1.0 s), lapsing when the density
> drops. The 98.1% / 92% figures below
> therefore describe the 6-of-10 buffer on combat footage — they must not
> be quoted next to demo results. This is a large part of why RQ3 says 92%
> confirmed-alarm precision while the v4 live field test says 18.5%: they
> measure different confirmation logic in different regimes, not just
> different data. A direct ByteTrack-vs-buffer comparison
> (`scripts/evaluation/compare_tracking_vs_buffer.py`) has not been run yet.

## Protocol (identical for v2 and v3)

- **Clips:** Test1 (91 s, bodycam), Test2 (113 s, desert combat),
  Test4 (63 s, combat) — **267 s total**, weapon-dense footage.
- **Sampling:** 10 fps, conf 0.40, imgsz 640.
- **Buffer:** the repo's `TemporalConsistencyBuffer` (window=10,
  required_hits=6, IoU 0.3) — a detection track confirms only if seen in
  ≥6 of the last 10 frames.
- **Models:** v2 measured 20 Nov 2026, v3 (adopted model) 21 Nov 2026.
- **Scripts / raw data (dataset workspace):** `rq3_measure.py` /
  `rq3_measure_v3.py`, `analyze_rq3.py`; outputs in `mining/rq3/` (v2)
  and `mining/rq3-v3/` (v3). Full narrative:
  `mining/rq3-v3/rq3_v3_summary.md`.

## Results

| | v2 (20 Nov) | v3 (21 Nov) |
|---|---|---|
| Raw detection frames (boxes) | 350 (354) | 357 (368) |
| Raw TP frames | ~233 | ~314 |
| Raw FP frames | ~121 | ~54 |
| Raw precision | 66% | 85% |
| Confirmed frames (TP / FP) | 73 (67 / 6) | 108 (107 / 1) |
| **FP-frame reduction (buffer)** | **95.0% (121→6)** | **98.1% (54→1)** |
| TP-frame retention (buffer) | 28.8% (67/233) | 34.1% (107/314) |
| Confirmed tracks (TP / FP) | 11 (8 / 3) | 13 (12 / 1) |
| **Confirmed-alarm precision** | **73% (8/11)** | **92% (12/13)** |
| Sustained TP tracks (≥6 fr) confirmed | 9/9 | 12/12 |

Per video (v3): Test1 — 0 confirmed FP frames; Test2 — 1 (a degenerate
tall-thin box on empty dirt, 1 frame); Test4 — 0. All detections on these
clips are Gun; 0 Knife tracks (v2 had 1). The buffer's role is unchanged:
it kills short flickers — v3's remaining raw FPs are overwhelmingly 1–2
frame flickers that never confirm.

## Estimation method (why the "~" on raw TP/FP)

Raw TP/FP frame counts are partly estimated. All 13 confirmed tracks and
all 23 long raw tracks (n≥4) were eye-verified from contact sheets, plus
a random sample of 20 short unverified tracks (seed 42); five borderline
tracks were resolved with 2.5× zooms. The 76 remaining unverified short
tracks (91 frames) were estimated from the verified sample mix (73% TP /
27% FP → 19 TP / 7 FP frames). Sensitivity: at v2's 60/40 sample mix the
totals would be TP ~302 / FP ~66 — every headline conclusion is
unchanged. Verdicts: `mining/rq3-v3/v3_eye_verdicts.json`.

## Caveats (quote with the numbers, not after them)

- **GT drift:** the 1 fps ground-truth labels changed after v2's
  measurement (WS5 audit + corrections: test1 POS frames 22→8, test2
  26→18, test4 13→20). The v2 and v3 auto-classifications are not
  strictly like-for-like; the eye-verification step (which judges video
  content, not labels) makes the comparison robust.
- **Conservative window:** 10 fps × window 10 = a 1 s confirmation
  window vs 0.33 s in the live 30 fps demo, so this measurement confirms
  MORE medium-duration FPs than the demo would. The 98.1% figure is
  conservative.
- **Regime:** these clips are full of real guns, so the FP pressure here
  is low. This result **says nothing about the phone/pen/bullet regime**
  — the field sessions showed v3 still false-alarms on phones, pens and
  bullets in everyday scenes, exactly the objects this footage never
  contains. RQ3 says v3 is the right model for the weapon-dense regime
  and the buffer does its job; the live regime is a separate question,
  answered by the field tests.
- **Degenerate boxes persist:** v3 still occasionally fires a box on
  empty ground/sky (test2-R12 tall-thin ground box, n=6 — it confirmed,
  1 frame). The buffer caught all but one frame of it, but the
  model-level defect is not gone.
