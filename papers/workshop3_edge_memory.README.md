# Workshop 3 — The Planner Is Free, Perception Is the Wall

**File:** `workshop3_edge_memory.tex` · **Venue:** workshop · **Status:** draft (unpublished)
**Authors:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar (Clark Atlanta University)

## Thesis
For edge deployment of driving VLAs, quantizing the language head optimizes the cheap part. The
Qwen-0.5B planner is already trivial for the edge (~1 GB FP16, ~0.26 GB at group-wise INT4); the
real wall is **perception** (UniAD ResNet-101 over six cameras + temporal BEVFormer, FP32, custom
CUDA ops), which dominates wall-clock and is far from real-time on Jetson Orin without a TensorRT-INT8
port.

## Key content
- Backbone footprint table: FP16 16.0 b/w (~1.0 GB) → INT4 per-ch 4.06 (~0.25 GB) → INT4 g128 4.125
  (~0.26 GB). Group-wise overhead ≈ 16/g bits/wt (0.125 at g=128).
- End-to-end node ~3–5 GB (perception-dominated).
- **Measured on a real Jetson AGX Orin 64 GB** (JetPack 6 / L4T R36.4.7, torch 2.8): the actual
  OpenDriveVLA-0.5B planner (291/291 decoder tensors, 357.8M Linear params) — **2.88 GB runtime peak,
  14.9 tok/s** (1024-ctx → 60-token decode, eager, default clocks, 30 iters), load 13.6 s. Packed
  weight footprint **FP16 0.716 → INT8 0.363 → INT4-g128 0.185 GB (3.9×)**. Latency identical across
  precisions (fake-quant → no packed kernel yet). Raw JSON: `../orin/results/orin_planner_bench.json`;
  benchmark script `../orin/bench_planner_orin.py`.
- Edge-feasibility table (A100 / Orin AGX [measured] / Orin NX / Orin Nano / Raspberry Pi).

## Scope / caveats
Planner now **measured on-device** (Orin AGX). Perception throughput rows are still device-class
**estimates** (marked †) — no TensorRT port yet. Fake-quant (dequantized) weights: the *static*
footprint is real/measured, but a packed-INT4 **speedup** needs a custom kernel and is not yet realized
(latency is flat across precisions on-device, as expected).

## Build
`latexmk -pdf workshop3_edge_memory.tex`. Fill bib placeholders before submission.
