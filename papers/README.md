# Papers

Status after the 2026-08-26 correction (see `../FINDINGS.md`). The NeuroNCAP render server is
stateful and the old harness enumerated configs in a fixed order against one renderer per scene,
so every closed-loop quantization comparison was confounded with run position. Re-measured, INT4
is indistinguishable from FP16 (`../analysis/RESULTS_clean_harness.md`).

## Current

- **`iclr27_phantom_collapse.tex`** — the corrected result. Documents the contamination mechanism,
  the state-restore fix, and the re-measurement showing no 4-bit safety collapse. Target: ICLR 2027
  (needs `iclr2027_conference.sty`, anonymisation, and the AI-disclosure section).

## Still standing

- **`workshop1_openloop_blind.tex`** — open-loop L2 is ego-motion extrapolation (19x ego ablation).
  Entirely replay-based, never touched the renderer. Unaffected. Note its closing paragraph
  proposes closed-loop testing as the remedy; that framing now needs the caveat from the
  correction, but no result in it is falsified.
- **`workshop3_edge_memory.tex`** — memory/latency profile and Jetson Orin measurements. Replay and
  on-device only. Unaffected, except one clause claiming group-wise INT4 "preserves generation
  coherence" — coherence was measured on the contaminated harness and should be cut.

## Needs revision, not deletion

- **`icra27_crossembodiment.tex`** — mixed. Its driving *closed-loop* sections (coherence,
  granularity-under-domain-shift) are void. Its activation-quantization crossover is
  replay-based and survives; its LIBERO/OpenVLA-OFT half has not been audited for an analogous
  state-reuse problem and should be before use.

## Removed 2026-08-26

`conference_closedloop_reveals`, `opendrivevla_full_paper`, `workshop2_granularity`,
`journal_comprehensive`, `paper_drivedvla_quant` — each built its central claim on the
contaminated closed-loop numbers. They remain in git history prior to this commit.
