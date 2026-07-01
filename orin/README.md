# On-device (Jetson AGX Orin) measurements

Real measurements of the OpenDriveVLA-0.5B **planner** (Qwen2.5-0.5B language/action head) on an
NVIDIA **Jetson AGX Orin 64 GB** (JetPack 6 / L4T R36.4.7, CUDA cap 8.7, PyTorch 2.8). The planner
is the quantized component and needs only torch + transformers — **not** the mmcv/mmdet3d perception
ops — so it runs on a stock Orin. Perception (UniAD) is not ported here.

## Scripts
- `bench_planner_orin.py` — systems benchmark (latency / throughput / runtime memory / packed
  weight footprint) for FP16 / INT8-g128 / W4-g128 / W4-perchannel. Builds a plain `Qwen2ForCausalLM`
  from the LlavaQwen config and loads the 291 real decoder tensors (skips vision tower).
- `replay_accuracy.py` — replays captured fused `inputs_embeds` (per nuScenes sample) through the
  planner at a chosen quantization → `plan_conv` json (open-loop accuracy, no perception needed).
- `subset_l2.py` — scores plan_conv files: ST-P3 L2 vs GT over the captured subset + pairwise
  trajectory agreement (cross-device / cross-quant).
- Capture: `drivevla/inference_drivevla.py` with `ODV_CAPTURE_DIR=… ODV_CAPTURE_N=…` dumps the
  fused `inputs_embeds` from the A100 full pipeline for replay.

## Results (`results/`)
**Systems** (`orin_planner_bench.json`, 1024-ctx → 60-token decode, eager, 30 iters):

| config | latency | tok/s | runtime peak | packed weights |
|---|---|---|---|---|
| FP16 | 4.02 s | 14.9 | 2.88 GB | 0.716 GB |
| INT8 g128 | 4.01 s | 15.0 | 2.88 GB | 0.363 GB |
| **W4 g128** | 4.02 s | 14.9 | 2.88 GB | **0.185 GB** |
| W4 per-channel | 4.02 s | 14.9 | 2.88 GB | 0.179 GB |

Latency identical across precisions — fake-quant (dequant to FP16, no packed kernel), so the win is
the **static footprint** (3.9× at W4-g128), not speed.

**Accuracy** (`orin_accuracy.json`, 40-sample nuScenes-mini val subset, ST-P3 L2, fp16 replay):

| | L2@1s | L2@2s | L2@3s |
|---|---|---|---|
| Orin FP16 | 0.196 | 0.453 | 0.870 |
| Orin W4-g128 | 0.206 | 0.475 | 0.897 |
| Orin W4-perch | 0.204 | 0.467 | 0.882 |

- **Deployment fidelity:** Orin reproduces A100 trajectories to **1.1 cm mean** (FP16; 1.5 cm W4-g128).
- **Open-loop blindness persists on-device:** quantization barely moves aggregate L2 (<3 cm) even
  though per-sample W4 plans differ from FP16 by **14–24 cm** — the metric averages it away. This is
  the central thesis, now reproduced on the real edge target.

## Per-scene success on the Orin (all 162 mini-val samples, 4 scenes)
Success = % of predictions within a distance of the human (GT) trajectory, by mean L2 over 3 s.
Files: `results/replay162_orin_*.json`, `results/scene_success.txt`.

**FP16** (W4-g128 within ~1% per scene; 0 malformed everywhere):

| scene | N | mean L2 | ≤1.0 m | ≤1.5 m | ≤2.0 m |
|---|---|---|---|---|---|
| scene-0553 (wait at intersection) | 41 | 0.00 | 97.6% | 97.6% | 97.6% |
| scene-0796 (busy street) | 40 | 0.30 | 95.0% | 95.0% | 95.0% |
| scene-0103 (peds, turning car) | 40 | 0.88 | 67.5% | 85.0% | 85.0% |
| scene-0916 (parking lot) | 41 | 1.10 | 58.5% | 75.6% | 85.4% |
| **ALL** | **162** | **0.57** | **79.6%** | **88.3%** | **90.7%** |

Overall ~80% of plans land within 1 m of the human path; per-scene ranges 59–98%. W4-g128 tracks
FP16 per scene (ALL 78.4% vs 79.6% ≤1 m); W4-perch similar (80.9%). Note scene-0553 is a
stationary "wait at intersection" clip — GT ≈ stay-put, so "predict stopped" is trivially near-exact
(mean L2 ~0); this is the ego-extrapolation effect and inflates that scene's success. The harder,
dynamic scenes (0103 turning, 0916 parking-lot maneuvering) are where success drops — and where
quantization *could* matter, yet open-loop success still can't separate the configs.

## Caveats
Planner-only (perception not ported to sm_87 yet); fp16 replay of cached perception (so absolute L2
differs slightly from the bf16 full-pipeline 0.33 m, and is over a 40-sample subset); fake-quant
(static footprint is real, a 4-bit *speedup* needs a packed kernel + TensorRT).
