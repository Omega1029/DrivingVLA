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
- Edge-feasibility table (A100 / Orin AGX / Orin NX / Orin Nano / Raspberry Pi).

## Scope / caveats
Footprints measured on an A100 node (relative split transfers; absolute Orin numbers differ).
Throughput rows are device-class **estimates** (marked †), not on-Orin measurements. Fake-quant
(dequantized) weights, so packed-INT4 memory/latency savings are projected, not realized.

## Build
`latexmk -pdf workshop3_edge_memory.tex`. Fill bib placeholders before submission.
