> # ⚠ CORRECTION (2026-08-26) — READ BEFORE THE REST OF THIS FILE
>
> **The closed-loop results below are invalid.** The NeuRAD render server is stateful:
> `POST /update_actors` overwrites its actor set and `neuro_ncap/engine.py:79` reads that set
> **once at engine init**. `run_benchmark_12.sh` / `parallel_benchmark.sh` launched one renderer
> per *scene* and looped configs inside it in fixed order (`fp16`, `naive_w4`, `group_w4_g128`),
> so **only the first config per scene saw a clean actor set**. Badness tracked run position, not
> bit-width.
>
> Re-measured with a state-restore protocol (`closed_loop_harness/run_sweep_clean.sh`),
> 16 scenario instances x 10 seeds, 95% bootstrap CI over instances:
>
> | config | corrected NCAP | previously reported |
> |---|---|---|
> | FP16 | 4.81 [4.62, 4.97] | 4.41 |
> | W8 | 4.81 [4.58, 5.00] | 4.50 |
> | naive W4 | **4.71 [4.38, 5.00]** | 2.62 |
> | group W4 g128 | **4.86 [4.67, 4.98]** | 2.62 |
>
> Paired vs FP16: W8 -0.00 [-0.15,+0.17], naive W4 -0.10 [-0.33,+0.12], group W4 +0.05
> [-0.06,+0.20] — **all indistinguishable from full precision**. Plain RTN W4 already matches W8.
>
> Consequently void: the 4-bit safety collapse, the "no 4-bit recipe survives" claim, the
> granularity story, the coherence-vs-safety hierarchy, and the braking-onset mechanism.
>
> **Unaffected** (never touched the renderer): open-loop replay incl. the activation-quant/DCT
> crossover, all Jetson Orin replay numbers, and the OpenVLA-OFT/LIBERO manipulation study.

# OpenDriveVLA — Results Log

## Baseline: OpenDriveVLA-0.5B, FP16, nuScenes-mini bring-up

- **Date:** 2026-06-26
- **Checkpoint:** `OpenDriveVLA/OpenDriveVLA-0.5B` (released, gated HF)
- **Perception:** UniAD stage-1 track+map (`uniad_base_track_map.pth`), run live, FP32
- **Split:** 162 samples = the nuScenes-mini scenes that fall in the official val split
  (NOT the full 6019-sample val set — this is a pipeline-correctness / sanity number,
  not a paper-comparable figure)
- **Hardware:** 1× A100-80GB, bf16 inference, ~1.9 s/sample

### Open-loop planning (162 mini-val samples)

| Eval protocol | L2 1s | L2 2s | L2 3s | **L2 Avg** | Coll 1s | Coll 2s | Coll 3s | **Coll Avg** |
|---|---|---|---|---|---|---|---|---|
| UniAD-style | 0.18 | 0.60 | 1.21 | **0.66** | 0.00 | 0.62 | 3.09 | **1.23** |
| ST-P3-style | 0.13 | 0.30 | 0.55 | **0.33** | 0.00 | 0.15 | 1.03 | **0.39** |

Paper (OpenDriveVLA-0.5B, full val, ST-P3 avg L2) reports ~0.30–0.40 m, so the mini
subset is in the right ballpark — the pipeline is wired correctly.

### Reproduce
```bash
bash scripts/eval_drivevla_mini.sh checkpoints/OpenDriveVLA-0.5B 1
```

## Caveats / next for apples-to-apples
- Full-val numbers need the 330 GB trainval images (deferred by choice).
- L2 here is over 4 mini scenes; variance is high vs the 150-scene val.

## Phase 5: Quantization (weight-only fake-quant, Qwen backbone only)

Method: per-output-channel symmetric RTN, applied to all 168 Linear layers
(357.8 M params) in `model.model.layers`. Perception (UniAD), the scene/agent/map
projectors, embeddings and `lm_head` left FP. Measures pure W{bits} accuracy impact
on the language/action head in the proven FP inference path.

> Why fake-quant: bitsandbytes' `device_map` dispatch attaches fp16-casting accelerate
> hooks to the whole model, which collides with the FP32 UniAD ResNet BatchNorm
> (`Expected weight Float but got Half`) under the deepspeed inference path. Fake-quant
> isolates the accuracy question from that integration issue.

| Config | ST-P3 L2 (1s/2s/3s/Avg) | UniAD L2 Avg | Malformed out | Notes |
|---|---|---|---|---|
| FP16 baseline | 0.13 / 0.30 / 0.55 / **0.33** | 0.66 | 0% | |
| W8 | 0.13 / 0.30 / 0.55 / **0.33** | 0.66 | 0% | lossless |
| W4 | 0.13 / 0.31 / 0.55 / **0.33** | 0.67 | 0% | lossless |
| W3 | 1.07 / 1.85 / 2.80 / **1.91** | 3.26 | 8% | collapse begins |
| W2 | 4.07 / 6.51 / 8.76 / **6.45** | ~ | 100% | total collapse |

(W2's 6.45 m ≈ the ego-ablation's 6.38 m below: a model emitting no usable trajectory scores
the same as a model handed no ego-state — both are "useless plan" floors.)

**The cliff is W4 → W3.** W8/W4 are perfectly lossless. At W3 the L2 jumps ~6× *and* 8% of
outputs no longer contain a parseable trajectory — the **language generation breaks before
trajectory accuracy does**. "Malformed out" = fraction of 162 samples whose text answer had
no parseable `(x,y)` coordinates (degenerate "stay-in-place" substituted, hence part of the L2
blow-up). This is exactly the degradation open-loop L2 hid until the eval parser was made robust.

Reproduce: `bash scripts/eval_drivevla_mini.sh checkpoints/OpenDriveVLA-0.5B 1 "--wquant-bits 4" "miniW4"`

## Ego-dependency ablation (the key diagnostic)

Zero the ego-state (`gt_ego_lcf_feat`) and history (`gt_ego_his_trajs`) fed to the prompt,
keep perception, FP16:

| FP16 model | ST-P3 L2 Avg | Malformed |
|---|---|---|
| normal (ego-state + history + perception) | **0.33 m** | 0% |
| ego-state & history zeroed (perception only) | **6.38 m** (19×) | 0% |

**Interpretation:** OpenDriveVLA's open-loop nuScenes planning is **almost entirely
ego-motion extrapolation** from the velocity/history it's handed as text. Remove that and the
plan collapses (0.33 → 6.38 m) while outputs stay well-formed — i.e. the perception/VLA
pathway contributes little to this metric. This explains, with one number, why:
- model size barely matters (0.5B = 3B = 7B = 0.33 m in the paper),
- W8/W4 quantization is lossless (no hard reasoning to lose — it's copying velocity forward),
- W3 collapse is a *generation-coherence* failure (malformed text), not a planning-reasoning one.

It is also the case for doing **closed-loop** (NeuroNCAP): only there can you not coast on the
logged ego trajectory, so only there would the perception/VLA — and any quantization damage to
it — actually show up.

Reproduce: `CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29611 ODV_ABLATE_EGO=1 bash scripts/eval_drivevla_mini.sh checkpoints/OpenDriveVLA-0.5B 1 "" "miniNoEgo"`

## ★★ REAL-CAMERA, 14-SCENE RESULT (2026-06-29) — current numbers; read this first

> **The closed-loop tables further down this file were produced with a broken renderer.**
> tiny-cuda-nn (tcnn) was not installed, so NeuRAD fell back to a torch hashgrid the checkpoint's
> `tcnn_encoding.params` blobs could not load into (`strict_load=False`) → every camera frame was
> flat grey (CAM_FRONT std ~3). The model was driving on blank perception. After installing tcnn the
> cameras are photorealistic (std 25–49) and we re-ran on **14 scenes (16 scenario instances)**,
> 10 paired seeds each — the full released NeuroNCAP closed-loop benchmark.

**Mean generation coherence (well-formed predicted-frame rate), real cameras, 10 seeds each:**

| Config                  | mean coherence | per-scenario behaviour              |
|-------------------------|----------------|-------------------------------------|
| FP16                    | 76.0%          | —                                   |
| W4 per-channel (naive)  | **3.7%**       | collapses (≤15%) on 15/16           |
| W4 group-128            | **73.5%**      | recovers (≥85% of FP16) on 13/16    |

**Full per-scene table (coherence %, then NCAP score, then collisions/10):**

| scene_scenario   | FP16 coh | naive coh | g128 coh | FP16 ncap | naive ncap | g128 ncap | FP16 coll | naive coll | g128 coll |
|------------------|---------:|----------:|---------:|----------:|-----------:|----------:|----------:|-----------:|----------:|
| 0099_stationary  | 60.6 | 0.0  | 30.0 | 5.0 | 1.4 | 0.0 | 0  | 10 | 10 |
| 0101_stationary  | 75.0 | 14.3 | 46.7 | 5.0 | 0.3 | 0.0 | 0  | 10 | 10 |
| 0103_frontal     | 74.3 | 0.0  | 72.9 | 5.0 | 4.1 | 5.0 | 0  | 4  | 0  |
| 0106_frontal     | 66.9 | 0.0  | 78.7 | 4.5 | 3.7 | 4.7 | 1  | 4  | 1  |
| 0108_side        | 85.7 | 0.0  | 91.6 | 4.0 | 5.0 | 4.6 | 2  | 0  | 1  |
| 0108_stationary  | 72.4 | 0.0  | 74.7 | 5.0 | 5.0 | 5.0 | 0  | 0  | 0  |
| 0110_frontal     | 70.7 | 0.0  | 74.3 | 5.0 | 5.0 | 5.0 | 0  | 0  | 0  |
| 0278_side        | 75.3 | 0.0  | 67.9 | 5.0 | 2.4 | 1.6 | 0  | 10 | 9  |
| 0278_stationary  | 79.7 | 0.0  | 83.5 | 0.0 | 0.1 | 0.9 | 10 | 10 | 9  |
| 0331_stationary  | 83.2 | 5.0  | 97.5 | 5.0 | 1.5 | 1.4 | 0  | 10 | 10 |
| 0346_frontal     | 83.1 | 15.6 | 86.9 | 4.5 | 4.2 | 5.0 | 1  | 3  | 0  |
| 0783_stationary  | 72.4 | 0.0  | 73.1 | 5.0 | 0.0 | 0.0 | 0  | 10 | 10 |
| 0796_stationary  | 72.4 | 8.6  | 57.9 | 0.9 | 0.0 | 1.6 | 10 | 10 | 10 |
| 0921_side        | 90.0 | 14.2 | 90.4 | 5.0 | 5.0 | 4.5 | 0  | 0  | 1  |
| 0923_frontal     | 74.6 | 0.8  | 72.1 | 4.0 | 3.0 | 3.7 | 3  | 7  | 4  |
| 0966_stationary  | 79.4 | 0.0  | 78.3 | 5.0 | 0.1 | 0.4 | 0  | 10 | 10 |

(2 of 16 cells — 0106_stationary, 0110_side — collided 0/0 for all configs and have no coherence rate;
excluded from coherence means.) Source: `benchmark14_coherence.tsv`.

**Honest reading:** the coherence finding **replicates across all 16 scenario instances** —
per-channel W4 collapses generation (mean 3.7%, ≤15% on 15/16), group-W4 recovers most of FP16
(mean 73.5% vs 76.0%, ≥85% of FP16 on 13/16). Caveats: (1) group-W4 ≠ exact FP16 — partial recovery
on 3 scenes (0099, 0101, 0796); (2) several scenes are failed by **every** config including FP16
(10/10 collisions: 0278_stationary, 0783_stationary, 0796_stationary, 0966_stationary, …), so
collision/NCAP is scene-dependent and confounded — lead with coherence; (3) rendered inputs are
still OOD to real nuScenes.

---

## Closed-loop (NeuroNCAP) — scene-0103 [SUPERSEDED: blank-camera renderer, see section above]

The 3-node closed-loop harness (neuro-ncap orchestrator + NeuRAD renderer + OpenDriveVLA model
node) runs end-to-end on scene-0103. Unlike open-loop L2, here the model drives: each frame is
re-rendered (NeuRAD) at the ego's *driven* pose, so there is no logged ego trajectory to coast
on. CAN bus for 0103 was synthesized from the ego-pose track (gated expansion not needed; see
NEURONCAP_PLAN.md). Score: NCAP 0–5, **5 = no collision**; lower = collision at higher impact
speed relative to reference.

| Config | scenario | runs | ncap_score | impact_speed | collision |
|--------|----------|------|-----------|--------------|-----------|
| FP16   | frontal  | 1    | **5.0**   | 0.0 m/s      | none      |
| W4     | frontal  | 1    | **2.70**  | 4.32 m/s     | **YES**   |

reference_speed 13.28 m/s for both. Same scenario + same perturbation seed (run index 0); the
**only** difference is FP16 vs W4 weights.

**This is the result the closed-loop effort was built to surface.** Open-loop L2 called W4
"lossless" (W4 = FP16 = 0.33 m) because that metric is ~ego extrapolation. In closed-loop —
where the model actually drives and perception/planning is on the critical path — **W4
quantization causes a collision (impact 4.32 m/s) that the FP16 model avoids (impact 0)**, and
NCAP score drops 5.0 → 2.70. The "quantization is free" conclusion is an artifact of the
open-loop benchmark; it does not hold under closed-loop safety testing.

Caveats: (1) runs=1 is a single perturbation — directionally strong but needs runs=N for a
distribution (in progress). (2) object-detection recall on NeuRAD-rendered images is ~0
(perception is weaker on the neural-rendered domain than on real nuScenes), so absolute scores
aren't directly comparable to the NeuroNCAP paper; the FP16-vs-W4 *delta* is the valid signal.

### Closed-loop quantization-technique recovery (scene-0103 frontal, seed-0)

The perturbation where naive W4 collides (FP16 5.0 / naive-RTN-W4 2.70) is a fixed seed-0; we
sweep PTQ techniques in-place on the live model node (snapshot FP once, re-quantize via
`/quantize` — no reload) and re-score the same seed-0 rollout. **Only the quantization changes.**

| technique                         | bits | group | ncap_score | impact (m/s) |
|-----------------------------------|------|-------|-----------|--------------|
| FP16                              | 16   | –     | 5.0       | 0.0          |
| RTN per-channel (W8)              | 8    | per-ch| 5.0       | 0.0          |
| **RTN per-channel (W4, naive)**   | 4    | per-ch| **2.70**  | **4.32 ⟵ COLLISION** |
| **RTN group-wise (W4, g128)**     | 4    | 128   | **5.0**   | **0.0 ⟵ RECOVERED** |
| RTN group-wise asym (W4, g128)    | 4    | 128   | 5.0       | 0.0          |
| RTN group-wise asym (W4, g64)     | 4    | 64    | 5.0       | 0.0          |
| RTN group-wise (W3, g128)         | 3    | 128   | 5.0       | 0.0          |
| AWQ (W4, g128) — calib on rollout | 4    | 128   | 5.0       | 0.0          |
| AWQ (W4, g64)                     | 4    | 64    | 5.0       | 0.0          |

**Finding: the closed-loop W4 collision is an artifact of coarse per-output-channel quantization
granularity, not of 4-bit precision per se.** Switching to group-wise scales (g128 — one knob,
negligible overhead) fully restores FP16 collision-avoidance on the exact perturbation where naive
W4 crashes. Asymmetric and AWQ also recover; group-wise symmetric is the simplest sufficient fix.
(AWQ calibrated on a live FP rollout: 1187 activation batches over the rendered closed-loop frames.)

The "and here's the fix" half of the paper: per-channel W4 is unsafe closed-loop; group-wise W4 is
safe — and open-loop L2 couldn't have told you either way (it calls *all* of them lossless).

Infra: `inference/quantizers.py` (QuantState snapshot + rtn_sym/rtn_asym/awq, group), server
`/quantize` + `/awq_calib_{start,finish}`, sweep `neuro-ncap/run_quant_full.sh`.

### ★ HEADLINE (clean metric): generation coherence under domain shift

Primary claim, stated with the un-confounded metric (well-formed trajectory output, not the noisy
collision proxy). On rendered (deployment-domain) closed-loop inputs, scene-0103 frontal, 10 seeds:

| config                    | well-formed frames | in-dist (open-loop, real imgs) | OOD (rendered) |
|---------------------------|--------------------|--------------------------------|----------------|
| FP16                      | 10/14              | fine (0.33 m L2)               | robust         |
| **naive per-channel W4**  | **0/14 (collapse)**| fine (0.33 m, lossless)        | **collapses**  |
| **group-wise W4 (g128)**  | **10/14 (=FP16)**  | fine                           | **robust**     |

**Thesis: per-channel INT4 is adequate in-distribution but loses generation coherence under
deployment-domain shift; group-wise INT4 (~0.1 extra bits/weight) is robust to the shift and
matches FP16. Open-loop L2 cannot distinguish them — it calls both lossless.** The fix (group-wise)
is the actionable recommendation; naive-W4 is intentionally NOT rescued — its collapse is the
contrast. Use generation-coherence/malformed-rate as the metric; collision rate is a confounded
downstream proxy (see correction note below).

### Multi-seed validation (scene-0103 frontal, seeds 0–9, paired)

All seeds run via the renderable-actor-substitution fix in `scenario.py` (preserves the per-seed
jitter RNG; seed-0 still the 2.70 failure). Paired per seed (jitter seeded by run-index, identical
across configs); only the quantization differs.

| config                       | mean NCAP | collision rate | per-seed ncap_score |
|------------------------------|-----------|----------------|---------------------|
| FP16                         | 5.00      | **0/10**       | all 5.0 |
| **naive per-channel W4**     | 4.08      | **4/10 (40%)** | [2.7, 5, 2.7, 2.7, 5, 5, 5, 5, 5, 2.7] |
| **group-wise W4 (g128)**     | 5.00      | **0/10**       | all 5.0 |

Outcomes are discrete: a clean avoid scores 5.0, the collision scores 2.7 (impact 4.32 m/s).

⚠️ **CORRECTION / mechanism audit (do not cite the "40% collision" as planning degradation).**
On inspecting `trajectories.json`, the naive-W4 "collisions" are NOT graded mis-planning — they are
**generation collapse**: naive per-channel W4 emits malformed/empty trajectory text on **14/14
frames every seed**, hitting the `retrieve_traj` all-zero "stay-in-place" fallback (the planner
freezes). FP16 and group-W4 emit real trajectories (only ~4/14 priming frames empty). The collision
then occurs on the 4/10 seed geometries that hit a non-driving ego. Evidence it's collapse, not a
precision-graded safety effect:
  1. **Non-monotonic:** W3-naive does NOT collapse (0/10) but W4-naive does — lower bits can't be
     "safer" under a real precision effect; collapse under greedy decoding is fragile/input-specific.
  2. **OOD-triggered:** on real nuScenes images (open-loop) naive-W4 is lossless/well-formed (0.33 m);
     it only collapses on NeuRAD-RENDERED inputs (which have ~0 detection recall — out of
     distribution). So this is a quantization × domain-shift interaction, confounded by the renderer.
  3. **Fallback-mediated:** the collision count depends on the chosen degenerate fallback (freeze);
     a constant-velocity fallback would change it.

Honest residual signal: under distribution shift, naive per-channel W4 loses generation coherence
(100% malformed frames) while group-wise W4 keeps FP16-level robustness (~4/14). That is a real
*robustness-of-quantization* result, but the collision rate is a noisy, confounded proxy for it —
not evidence that "per-channel INT4 degrades planning safety."

| config (10 seeds)            | malformed frames | collisions | note |
|------------------------------|------------------|-----------|------|
| FP16                         | 4/14             | 0/10      | baseline |
| naive per-channel W4         | **14/14**        | 4/10      | generation collapse on rendered inputs |
| group-wise W4 (g128)         | 4/14             | 0/10      | matches FP16 robustness |

(`neuro-ncap/run_multiseed.sh`, output/multiseed/.)

### Generation-coherence frontier (clean metric, from existing trajectories.json — no new runs)

Well-formed (non-degenerate) predicted-frame rate, scene-0103 frontal, 10 seeds each:

| config                  | well-formed rate | fully-collapsed seeds |
|-------------------------|------------------|-----------------------|
| FP16                    | 71.4%            | 0/10                  |
| **W4 naive (per-ch)**   | **0.0%**         | 10/10                 |
| **W4 group g128**       | **71.4% (=FP16)**| 0/10                  |
| W3 naive                | 21.4%            | 0/10                  |
| **W3 group g128**       | **85.7%**        | 0/10                  |
| W3 AWQ g128             | 0.0% ⚠           | 10/10                 |
| W2 naive / group / AWQ  | 0.0%             | 10/10                 |

**Clean result:** group-wise quantization preserves generation coherence where per-channel destroys
it — W4: group 71.4% (=FP16) vs naive 0%; W3: group 85.7% vs naive 21.4%. The advantage GROWS as
bits drop. W2 is the precision floor (all collapse). This is the un-confounded version of the
"group-wise is the robust recipe" finding and it's monotone/clean for naive-vs-group.

⚠ Caveats: (1) frame-level rate is noisy (W3-group 86% > FP16 71% is within noise — don't claim
group beats FP16). (2) **W3-AWQ collapsed to 0% while W4-AWQ was fine — almost certainly a bug in
the AWQ low-bit path, NOT evidence AWQ is bad. Do not report AWQ until debugged.** (3) Even FP16 is
only 71% coherent on rendered inputs — the OOD renderer is hard for everyone; the claim is relative
(group ≈ FP16, naive ≪). Compute: trajectories.json all-zero frames = retrieve_traj empty-parse
fallback = malformed model output.

### TODO
- [ ] **W3/W2 technique-separation** sweep: naive vs group vs AWQ at lower bits, where group-wise
      should start failing and AWQ (salience-protection) should pull ahead — earns AWQ its place.
- [ ] W2 result.
- [ ] Real (packed) INT4 memory/latency on the LLM portion — note perception (FP32)
      dominates wall-clock, so latency win is bounded.
- [ ] Mechanism plot: per-channel-W4 vs FP16/group-W4 planned-trajectory divergence over the rollout.
