# Main Conference — Closed-Loop Reveals What Open-Loop Hides

**File:** `conference_closedloop_reveals.tex` · **Venue:** main conference · **Status:** draft (unpublished)
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## Thesis
The combined story: open-loop L2 **hides** quantization damage (it's ego-extrapolation), and
closed-loop **reveals** it — naive per-channel INT4 collapses generation while group-wise INT4
recovers FP16-level coherence at ~0.1 extra bits/wt.

## Contributions
1. Open-loop L2 is ≈ ego-extrapolation (19× ego-ablation) → structurally blind to quantization.
2. First closed-loop characterization of weight-quant effects in a driving VLA: per-channel INT4
   collapses under deployment-domain shift; group-wise does not.
3. Generation-coherence as the un-confounded metric + a near-free recipe (group-wise INT4) + edge
   memory benchmarks.

## Key results (real-camera NeuroNCAP, 2 scenes, 10 seeds)
Coherence — 0103: FP16 74.3 / per-ch 0.0 / g128 72.9 %; 0796: 72.4 / 8.6 / 57.9 %.
Open-loop INT4 L2 0.33 m; ego-ablation 6.38 m (19×).

## Scope / caveats (in the paper's Limitations)
Two scenes; OOD renderer (even FP16 fails 0796 collisions 10/10); group-wise recovery partial on 0796;
collision/NCAP confounded → lead with coherence; AWQ low-bit path buggy/excluded; fake-quant.

## Path to camera-ready (gating items)
More scenes (12 more staged), fix/justify FP16-fails-0796 perception validity, report standard NCAP
alongside coherence, one mechanism ablation (measure the outlier/scale effect directly).

## Build
`latexmk -pdf conference_closedloop_reveals.tex`. Fill bib placeholders before submission.
