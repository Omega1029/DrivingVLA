# Combined / Comprehensive Paper

**File:** `opendrivevla_full_paper.tex` · **Status:** draft (unpublished) · merges all five drafts +
the on-device results into one paper.
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## What it is
The single, self-contained paper: open-loop blindness + closed-loop granularity collapse/fix +
edge-memory + **on-device (Jetson AGX Orin) systems, accuracy, and per-scene success**, with the
quantization methodology explained (RTN, per-channel vs group-wise, the outlier–scale mechanism) and
comparisons to prior methods (UniAD/VAD/OpenDriveVLA open-loop L2; RTN/GPTQ/AWQ quantization).

## Headline numbers (all real measurements)
- **Open-loop blind:** INT4 lossless (0.33 m); ego-ablation 6.38 m (**19×**).
- **Closed-loop (14 scenes / 16 scenarios, 10 seeds):** mean coherence FP16 **76.0%** / naive
  per-channel INT4 **3.7%** / group-128 INT4 **73.5%**; naive ≤15% on 15/16, group ≥85% of FP16 on 13/16.
- **On-Orin systems (AGX Orin 64 GB):** planner 2.88 GB, 14.9 tok/s; packed weights FP16 0.716 →
  INT4-g128 **0.185 GB (3.9×)**; latency flat (fake-quant).
- **On-Orin accuracy (162 mini-val):** FP16 ST-P3 0.146/0.317/0.571 (avg **0.345 ≈ published 0.35**);
  reproduces A100 to **1.1 cm**; per-scene success 58.5–97.6% within 1 m, unchanged by quantization.

## Comparison tables included
- Table 1: open-loop L2 vs UniAD / VAD / OpenDriveVLA-3B/7B (context).
- Table 2: RTN vs GPTQ vs AWQ (methodology positioning).
- Tables 5–8: memory, on-Orin systems, on-Orin accuracy, per-scene success.

## Scope / caveats
Group-wise recovery partial on 3 scenes; several scenes failed by all configs incl FP16 (coherence is
the lead metric); fake-quant (packed speedup needs a kernel); on-Orin accuracy is planner-only replay,
fp16, mini-val; AWQ/INT3/INT2 not re-run on real cameras.

## Build
`latexmk -pdf opendrivevla_full_paper.tex` (inline `thebibliography`; real arXiv citations included).
The five source drafts (`workshop1/2/3`, `conference`, `journal`) remain for venue-specific submission.
