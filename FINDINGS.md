# OpenDriveVLA quantization study — consolidated findings

Clean entry point. Full experiment log + every fix is in `RESULTS.md`; closed-loop build/plan is in
`NEURONCAP_PLAN.md`. This file is the honest 2-minute summary.

## One paragraph

We evaluate and quantize the released **OpenDriveVLA-0.5B** (Qwen2.5-0.5B planner on UniAD stage-1
track+map perception). Weight-only PTQ is applied **only to the Qwen backbone** (168 Linear layers,
357.8M params); perception, projectors, embeddings, and lm_head stay FP. The headline: **open-loop
nuScenes L2 cannot see quantization damage** (it is ego-motion extrapolation), but **closed-loop
NeuroNCAP can** — and there it reveals that *naive per-channel* INT4 loses generation coherence
under the deployment-domain (neural-rendered) shift, while *group-wise* INT4 (~0.1 extra bits/weight)
stays as robust as FP16.

## Study design

- **Open-loop:** ST-P3/UniAD L2 + collision on the 162 nuScenes-mini∩val samples
  (`scripts/eval_drivevla_mini.sh`).
- **Closed-loop:** 3 native processes (no Docker) — neuro-ncap orchestrator + NeuRAD renderer
  (:8000) + OpenDriveVLA model node (:9000, `inference/server.py`+`runner.py`). Scene-0103 frontal
  (the only NeuRAD checkpoint available here); CAN-bus expansion synthesized from the ego-pose track.
  Per-seed jitter is paired across configs; only the quantization differs.
- **Fast PTQ iteration:** the model node snapshots FP weights once and re-quantizes **in-place** via
  `/quantize` (no reload). `inference/quantizers.py` implements rtn_sym / rtn_asym / awq × group;
  AWQ calibrates by hooking the backbone during a live rendered rollout. Sweeps:
  `neuro-ncap/run_{quant_full,multiseed,bitsep}.sh`.

## Key results

**1. Open-loop is blind (and we know why).** FP16 = W8 = W4 ≈ 0.33 m L2 (≈ paper). Ego-ablation
(zero the ego-state + history text) → L2 0.33 → 6.38 m (**19×**): open-loop planning is ~entirely
ego-extrapolation, so it cannot reflect perception/planning or quantization damage.

**2. Closed-loop generation coherence (the clean metric — well-formed predicted-frame rate, 10
paired seeds, scene-0103 frontal):**

| config                | well-formed rate | note |
|-----------------------|------------------|------|
| FP16                  | 71.4%            | baseline (rendered inputs are hard even for FP16) |
| **W4 per-channel**    | **0.0%**         | total generation collapse |
| **W4 group-g128**     | **71.4%**        | **= FP16** |
| W3 per-channel        | 21.4%            | |
| **W3 group-g128**     | **85.7%**        | advantage grows as bits drop |
| W2 (all techniques)   | 0.0%             | precision floor |

**Thesis: per-channel INT4 is adequate in-distribution (open-loop, real images: lossless) but
collapses generation under deployment-domain shift; group-wise INT4 is robust to the shift and
matches FP16. Open-loop L2 calls both lossless.** Mechanism: collapse = malformed trajectory text →
`retrieve_traj` empty-parse → all-zero "stay-in-place" plan → planner freezes.

## What we explicitly do NOT claim (honest caveats)

- **Not "quantization degrades planning safety."** The earlier "naive-W4 collides 4/10" was a
  *confounded proxy*: collapse → frozen ego → collision only on tight-approach geometries, mediated
  by the chosen degenerate fallback. Use coherence/malformed-rate, not collision rate.
- **Frame-rate is noisy.** W3-group 85.7% > FP16 71.4% is within noise — do not claim group > FP16.
- **AWQ is unreported.** W3-AWQ collapsed to 0% while W4-AWQ was fine → almost certainly a low-bit
  AWQ-implementation bug, not a finding. Left un-debugged by choice.
- **Single OOD scene.** Only scene-0103; its NeuRAD checkpoint has ~0 detection recall (rendered
  inputs are out-of-distribution), which is *why* even FP16 is only 71% coherent. Generality and a
  "the renderer isn't just hard" defense need more scenes + a stronger renderer/perception
  (requires gated trainval metadata + per-scene NeuRAD checkpoints).

## Reproduce

- Open-loop: `CUDA_VISIBLE_DEVICES=0 bash scripts/eval_drivevla_mini.sh checkpoints/OpenDriveVLA-0.5B 1 "--wquant-bits 4" w4`
- Ego-ablation: prefix `ODV_ABLATE_EGO=1`.
- Closed-loop nodes + sweeps: see `NEURONCAP_PLAN.md` (renderer launch) and the `run_*.sh` scripts;
  quantization is swept live via the `/quantize` endpoint.

## Open threads (deferred)

Debug low-bit AWQ; real packed-INT4 latency/memory (bounded — FP32 UniAD perception dominates
wall-clock); 3B/7B variants; more scenes for generality.
