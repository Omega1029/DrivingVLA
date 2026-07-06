# E1 smoke (in-distribution, A100): activation quant × rotation on the driving VLA

Decoder-only replay of the 162 real captured nuScenes inputs (same protocol as
orin/replay_accuracy.py; FP16 reference L2avg 0.595 m, 42/162 legitimately-stationary
all-zero plans as base rate).

| config | ST-P3 L2avg (m) | all-zero plans | |xy|>50 m |
|---|---|---|---|
| FP16 (ref) | 0.595 | 42 (base rate) | 0 |
| W16A4, no rotation | **27.79 (47×)** | 107 | 7 |
| W16A4 + DCT | 0.646 | 42 | 0 |
| W4A4, no rotation | 9.34 | 127 | 3 |
| **W4A4 + DCT** | **0.611** | 41 | 0 |

Findings:
1. **A4 alone collapses the driving planner in-distribution** (weights untouched) — where
   weight-only W4 was lossless. The massive-activation wall transfers from the CoRL
   manipulation study (Llama-7B/OpenVLA-OFT) to Qwen-0.5B/OpenDriveVLA.
2. **One data-free DCT rotation restores W4A4 to FP16-level** (0.611 vs 0.595) and returns
   the all-zero count exactly to base rate.
3. **Failure surface differs from the weight axis:** 0 malformed text — the A4 failure is
   well-formed but degenerate *content* (stay-put/wild), vs the OOD weight-quant failure
   which is malformed *syntax*. Both freeze the vehicle; different mechanisms.

Taxonomy refinement for the ICRA paper: activations are the wall *in-distribution and
universally*; weight-granularity is the wall *under domain shift*. Interface determines the
failure surface (text syntax vs regression values), not the failing axis.
