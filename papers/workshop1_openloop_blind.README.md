# Workshop 1 — Open-Loop L2 Is Blind

**File:** `workshop1_openloop_blind.tex` · **Venue:** workshop · **Status:** draft (unpublished)
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## Thesis
The standard open-loop nuScenes L2 metric reports INT4 weight quantization as "lossless," but that
conclusion is a **metric artifact**: open-loop L2 is approximately ego-motion extrapolation, so it is
structurally blind to perception/planning — and to quantization damage.

## Key results
- FP16 / INT8 / INT4 open-loop L2 ≈ **0.33 m** (lossless); INT3 1.91 m (92% well-formed); INT2 6.45 m (dead).
- **Ego-ablation:** zeroing ego-state + 2 s history → L2 **6.38 m (19×)** while output stays well-formed.
- Therefore "INT4 is free" is no evidence of deployment safety → motivates closed-loop testing
  (companion: `workshop2`, `conference`).

## Scope / caveats
Single model (OpenDriveVLA-0.5B), nuScenes-mini∩val (162 samples). The claim is about the *metric's*
structural insensitivity, evidenced by the 19× ablation — not that a specific quantizer is unsafe
(that's the closed-loop companions).

## Build
`latexmk -pdf workshop1_openloop_blind.tex` (or Overleaf). Fill `(Replace with verified citation.)`
bib placeholders before submission.
