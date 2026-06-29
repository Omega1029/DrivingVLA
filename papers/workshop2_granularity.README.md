# Workshop 2 — Granularity, Not Bit-Width

**File:** `workshop2_granularity.tex` · **Venue:** workshop · **Status:** draft (unpublished)
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## Thesis
Under deployment-domain shift (photorealistic neural rendering), **naive per-output-channel** INT4
collapses trajectory-text generation, while **group-wise** INT4 (g128, ~0.1 extra bits/wt) recovers
most of FP16. The deciding factor is **scale granularity**, not 4-bit precision.

## Key results (real-camera NeuroNCAP, 14 scenes / 16 scenario instances, 10 paired seeds)
Mean generation coherence (well-formed predicted-frame rate):

| Config | mean coherence | per-scenario behaviour |
|--------|----------------|------------------------|
| FP16 | 76.0% | — |
| INT4 per-channel (naive) | **3.7%** | collapses (≤15%) on 15/16 |
| INT4 group-128 | **73.5%** | recovers (≥85% of FP16) on 13/16 |

Same per-channel INT4 is lossless in-distribution (open-loop) → it's a quantization × domain-shift
interaction. Mechanism: one weight outlier inflates the shared per-channel scale; group-wise confines
it to a 128-weight block.

## Scope / caveats
14 scenes / 16 scenario instances (full released benchmark). Group-wise recovery is **partial** on 3
(0099, 0101, 0796) — not exact FP16 equivalence. Collision/NCAP is a confounded, scene-dependent proxy
(several scenes have every config including FP16 colliding 10/10), so coherence is the lead metric.
INT3/INT2/AWQ were only measured on the earlier (blank-camera)
renderer and are not re-reported.

## Build
`latexmk -pdf workshop2_granularity.tex`. Fill bib placeholders before submission.
