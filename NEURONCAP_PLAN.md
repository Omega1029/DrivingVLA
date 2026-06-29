# NeuroNCAP closed-loop for OpenDriveVLA — plan & status

**Goal:** measure OpenDriveVLA in *closed-loop* (where the logged ego trajectory is NOT
available to coast on), FP16 vs W4, to find out whether the perception/VLA — and any
quantization damage to it — actually matters. This is the test the ego-ablation proved is
necessary (open-loop L2 is ~all ego extrapolation; see `RESULTS.md`).

## Architecture (no Docker needed)

NeuroNCAP is a **networked-node** design — three processes talking over localhost HTTP, so we
run each as a **native Python venv process** (this box has no Docker/Singularity, which is fine):

```
neuro-ncap (orchestrator)  --HTTP-->  OpenDriveVLA model node (/infer, /reset)
        |                  --HTTP-->  neurad-studio renderer (/render_image)
        controller + vehicle model + collision check + NCAP scoring
```

Per timestep: renderer produces 6 photorealistic nuScenes-domain camera PNGs at the ego's
*current* (possibly drifted) pose → model node returns a trajectory → orchestrator advances the
kinematic bicycle model → re-render. 50 perturbed sims per scenario.

## Model-node interface (from the UniAD template)

`/infer` **in:** `images{cam→PNG}`, `ego2world` 4×4, `timestamp` µs, `calibration`
(cam intrinsics, camera2ego, lidar2ego). **out:** `trajectory` = list of (x,y) ego-frame BEV.

Our node (`inference/server.py` + `inference/runner.py`) must:
1. Build the UniAD perception input (lidar2img, can_bus 16-vec, command) from images+pose+calib
   — **reuse the UniAD runner's preprocessing nearly verbatim** (OpenDriveVLA perception *is*
   UniAD stage-1 track+map).
2. Run perception → scene/agent/map tokens.
3. Track ego-state across timesteps (velocity/accel/yaw-rate from the ego2world sequence) and
   build the VLA prompt (ego-state text + 2 s history + mission command).
4. Run the Qwen VLA → trajectory text → `retrieve_traj` → 6×2 waypoints.
5. Return trajectory (supports `--wquant-bits` for the FP16-vs-W4 comparison).

## Scenes / data

16 nuScenes scenes need NeuRAD per-scene checkpoints (one Dropbox tarball, downloading).
Scenarios: stationary / frontal / side. **scene-0103 is in our nuScenes-mini** → first bring-up
targets 0103 only (we already have its data), mirroring the open-loop mini strategy. Other 15
scenes need their nuScenes metadata (gated trainval meta) — deferred until 0103 works.

## Phases

- [x] Clone neuro-ncap, neurad-studio, UniAD-server-template; confirm no-Docker native path.
- [x] Download + extract NeuRAD weights (14 scenes incl. 0103).
- [x] **A. Renderer up:** neurad-studio venv built; **render server loads scene-0103 and serves
      valid PNGs** on :8000. (Pose-frame for a *correct* (non-empty) standalone render is the
      orchestrator's job — it normalizes; my hand-built nuScenes-global pose rendered empty.)
- [x] **B. Model node (implemented, pending live test):** `inference/runner.py` now implements
      image normalize/pad (img_norm_cfg, /32), lidar2img + can_bus(18) + l2g construction from
      calibration, BEVFormer temporal-delta tracking, the perception-input dict, the VLA prompt
      (driven ego-state), and trajectory decode. `inference/server.py` complete. Syntax verified.
      NOT yet run against the model+renderer — that's part of D (calibration must match training).
- [x] **C. Sim up:** neuro-ncap venv built (ncap-venv: pyyaml, nuplan-devkit --no-deps,
      numpy 1.26.4 + opencv 4.9 for nuscenes/matplotlib compat). data/ symlink → OpenDriveVLA data.
      **Synthesized the gated CAN-bus expansion** for scene-0103 from the dense LIDAR_TOP ego-pose
      track (scratchpad/synth_canbus.py → data/nuscenes/can_bus/scene-0103_{pose,zoe_veh_info}.json);
      orchestrator only uses CAN for priming/initial velocity, then emulates from driven poses.
- [x] **D. Integrate:** 3 nodes live natively (renderer :8000 GPU2, model :9000 GPU0,
      orchestrator). Fixes to get the loop running on 0103-frontal:
        * renderer closed_loop/server.py: nuScenes actor_transform is already 4x4 (don't to4x4 it),
          and cast world/actor transforms to float32 (WLH_TO_LWH was float64).
        * model node (pydantic v1: drop Base64Bytes; images are torch.save'd uint8 tensors).
        * runner: use orchestrator can_bus(16)+command; can_bus preproc +rotation+patch_angle (no
          manual temporal delta — forward_test does it); img=[t], img_metas=[[meta]] with
          pts_filename/sample_idx/ori_shape; pass indexable command/sdc_planning(_mask).
        * panseg_head.forward_test: guard the eval-only lane-IoU block when gt_lane_masks is None
          (closed-loop has no GT map; ret_iou is popped before the VLM tokens anyway).
      Closed loop now steps through 0103-frontal (step 0..N) producing trajectories each frame.
- [ ] **E. Measure:** FP16 vs W4 NCAP score + collision rate on 0103 (then expand). [in progress]

## Renderer — working setup (reproduce)

neurad venv: `/home/justin_williams1/neuroncap/neurad-venv` (torch 2.1.2+cu121, nerfacc 0.5.2).
Fixes applied: (1) pinned torch/torchvision back to 2.1.2/0.16.2 after the `-e .` install pulled
torch 2.12; (2) `nerfstudio/utils/misc.py::torch_compile` made a no-op (NeuRAD's
`@torch_compile(mode="reduce-overhead", backend="eager")` is invalid on torch 2.1 — cleared
`__pycache__` after editing); (3) launch via `run_render_server.py` (disables dynamo).
Data: `neurad-studio/data/nuscenes` -> symlink to OpenDriveVLA mini; `0103/config.yml` version
edited `v1.0-trainval`→`v1.0-mini`.

```
cd /home/justin_williams1/neuroncap/neurad-studio
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 TORCH_COMPILE_DISABLE=1 \
  /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
  --port 8000 --load-config checkpoints/0103/config.yml --adjust_pose
```

## Key risks
- neurad-studio (nerfstudio) env / CUDA compatibility — isolate in its own venv.
- Perception on rendered images: feeding NeuRAD output through UniAD preproc outside the dataset
  pipeline (calibration/normalization must match training) — the main debugging surface.
- nuScenes metadata for the other 15 scenes is gated (trainval meta) — 0103-first sidesteps it.
