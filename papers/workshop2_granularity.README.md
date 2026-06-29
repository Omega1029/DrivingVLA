# Workshop 2 — Granularity, Not Bit-Width

**File:** `workshop2_granularity.tex` · **Venue:** workshop · **Status:** draft (unpublished)
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## Thesis
Under deployment-domain shift (photorealistic neural rendering), **naive per-output-channel** INT4
collapses trajectory-text generation, while **group-wise** INT4 (g128, ~0.1 extra bits/wt) recovers
most of FP16. The deciding factor is **scale granularity**, not 4-bit precision.

## Key results (real-camera NeuroNCAP, 2 scenes, 10 paired seeds)
Generation coherence (well-formed predicted-frame rate):

| Config | 0103 frontal | 0796 stationary |
|--------|--------------|-----------------|
| FP16 | 74.3% | 72.4% |
| INT4 per-channel | 0.0% | 8.6% |
| INT4 group-128 | 72.9% | 57.9% |

Same per-channel INT4 is lossless in-distribution (open-loop) → it's a quantization × domain-shift
interaction. Mechanism: one weight outlier inflates the shared per-channel scale; group-wise confines
it to a 128-weight block.

## Scope / caveats
Two scenes. Group-wise recovery is **partial** on the harder scene (0796) — not exact FP16 equivalence.
Collision/NCAP is a confounded, scene-dependent proxy (on 0796 every config including FP16 collides),
so coherence is the lead metric. INT3/INT2/AWQ were only measured on the earlier (blank-camera)
renderer and are not re-reported.

## Build
`latexmk -pdf workshop2_granularity.tex`. Fill bib placeholders before submission.
