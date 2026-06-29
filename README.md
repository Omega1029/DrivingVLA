# Quantizing OpenDriveVLA: Open-Loop Is Blind, Closed-Loop Reveals a Granularity Collapse

Research code + results for post-training weight quantization of the **OpenDriveVLA-0.5B** driving
VLA (a Qwen2.5-0.5B language/action head on UniAD track+map perception), evaluated **open-loop**
(nuScenes) and **closed-loop** (NeuroNCAP neural-rendering simulator).

> This repo holds **our contributions only**. It depends on, but does not vendor, the upstream
> [OpenDriveVLA](https://drivevla.github.io/), [NeuroNCAP](https://github.com/wljungbergh/NeuroNCAP),
> and [NeuRAD/neurad-studio](https://github.com/georghess/neurad-studio) projects. nuScenes data and
> model checkpoints are **not** included (licensed/gated) and are gitignored.

## Headline findings

1. **Open-loop L2 is blind.** Weight quantization to INT4 is "lossless" on nuScenes L2 (0.33 m), but
   zeroing the ego-state input inflates L2 to 6.38 m (**19×**) — open-loop L2 is approximately
   ego-motion extrapolation, so it cannot see quantization damage.
2. **Closed-loop reveals a collapse (2 scenes, real-camera NeuRAD, 10 seeds each).** Naive
   **per-channel** INT4 collapses trajectory generation (0.0% / 8.6% well-formed plans) while
   **group-wise** INT4 (g128, ~0.1 extra bits/wt) recovers most of FP16 (72.9% / 57.9% vs
   74.3% / 72.4%). It is a quantization x domain-shift interaction governed by **scale granularity,
   not bit-width**.
3. Honest caveats: group-wise recovery is *partial* on the harder scene; one scene (0796) is failed
   by every config including FP16, so collision/NCAP is a confounded, scene-dependent proxy — we lead
   with generation-coherence rate. See `RESULTS.md` / `FINDINGS.md`.

## Layout

| Path | What |
|------|------|
| `inference/quantizers.py` | Weight-only PTQ (RTN sym/asym, group-wise, AWQ) with in-place re-quant `QuantState` |
| `inference/server.py`, `runner.py` | OpenDriveVLA model node for NeuroNCAP (`/infer`, `/quantize`, AWQ calib) |
| `closed_loop_harness/` | Native (Docker-free) closed-loop drivers: CAN-bus synthesis, multi-seed & multi-scene runners, blob fetch, video renderers |
| `papers/` | 5 manuscripts (3 workshop, 1 conference, 1 journal) |
| `RESULTS.md`, `FINDINGS.md`, `NEURONCAP_PLAN.md` | Full results (incl. the renderer/tcnn fix) + harness build log |

## Reproduce (sketch)

1. Set up upstream OpenDriveVLA + NeuroNCAP + neurad-studio (with **tiny-cuda-nn** — without it NeuRAD
   renders blank grey frames and the checkpoint's `tcnn_encoding` weights silently fail to load).
2. Launch the NeuRAD renderer and the model node (`inference/server.py`).
3. `closed_loop_harness/run_multiseed.sh` (scene-0103) / `run_benchmark_12.sh` (full scene set).

Generation-coherence is computed from the logged `trajectories.json` (well-formed, non-degenerate
predicted-frame rate).

---
Built on OpenDriveVLA (AAAI 2026), NeuroNCAP (ECCV 2024), and NeuRAD (CVPR 2024) — see links above.
