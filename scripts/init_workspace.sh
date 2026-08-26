#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenDriveVLA autonomous-pipeline workspace initializer.
#
# NOTE: OpenDriveVLA has NOT released training scripts, and there is no `main.py`.
# The real entry points are:
#     drivevla/inference_drivevla.py   (planning + VQA inference, via torchrun)
#     drivevla/eval_drivevla.py        (open-loop L2 / collision metrics)
# wrapped by scripts/eval_drivevla.sh <CKPT_PATH> <NUM_GPU>.
#
# This script is idempotent: re-running skips artifacts already present.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="$ROOT/venv/bin"
export CUDA_HOME=/usr
mkdir -p data/nuscenes data/infos checkpoints output logs data/eval_share

# ── Phase 1: Environment ─────────────────────────────────────────────────────
$VENV/pip install --upgrade -q pip setuptools wheel
$VENV/pip install -q torch==2.1.2 torchvision==0.16.2
# flash-attn: use the prebuilt wheel (source build is slow/fragile)
FA="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
$VENV/python -c "import flash_attn" 2>/dev/null || $VENV/pip install -q --no-deps "$FA"
$VENV/pip install -e ".[train]"
# mmcv 1.7.2 + mmdet3d from source (compiled CUDA ops)
$VENV/python -c "import mmcv" 2>/dev/null || ( cd third_party/mmcv_1_7_2 && \
    $VENV/pip install -r requirements/optional.txt && MMCV_WITH_OPS=1 $VENV/pip install . )
$VENV/pip install -q mmdet==2.26.0 mmsegmentation==0.29.1 mmengine==0.9.0 motmetrics==1.4.0 casadi==3.6.0
$VENV/python -c "import mmdet3d" 2>/dev/null || ( cd third_party/mmdetection3d_1_0_0rc6 && \
    $VENV/pip install -q scipy==1.10.1 scikit-image==0.19.3 fsspec && $VENV/pip install . )
# quantization libs
$VENV/pip install -q autoawq auto-gptq bitsandbytes nuscenes-devkit || true

# ── Phase 1b: API keys (set these in your shell before running) ──────────────
: "${WANDB_API_KEY:=}"   # export WANDB_API_KEY=... for remote logging (optional)
[ -n "${WANDB_API_KEY}" ] && $VENV/wandb login "$WANDB_API_KEY" || echo "WANDB: local-only (no key set)"
export NUSCENES_DATASET_ROOT="$ROOT/data/nuscenes"

# ── Phase 2: Data + checkpoints (mini for bring-up) ──────────────────────────
[ -d data/nuscenes/samples ] || ( cd data/nuscenes && \
    wget -c https://www.nuscenes.org/data/v1.0-mini.tgz -O v1.0-mini.tgz && tar -xzf v1.0-mini.tgz )
[ -f data/infos/nuscenes_infos_temporal_val.pkl ] || ( cd data/infos && \
    wget -c https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl && \
    wget -c https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl )
[ -f checkpoints/uniad_base_track_map.pth ] || ( cd checkpoints && \
    wget -c https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth )
[ -f data/nuscenes/cached_nuscenes_info.pkl ] || ( cd data/nuscenes && \
    $VENV/gdown 16X0_-v-iXP9hVLNaDMmIiGhZKj24YOnb -O cached_nuscenes_info.pkl )
[ -d data/eval_share/gt ] || ( cd data/eval_share && \
    $VENV/gdown --folder https://drive.google.com/drive/folders/1NCqPtdK8agPi1q3sr9-8-vPdYj08OCAE -O gt )
# Gated VLA checkpoint (requires `hf auth login` + web license acceptance)
[ -f checkpoints/OpenDriveVLA-0.5B/model.safetensors ] || \
    $VENV/hf download OpenDriveVLA/OpenDriveVLA-0.5B --local-dir checkpoints/OpenDriveVLA-0.5B

# ── Phase 4: Baseline eval (REAL entry point — there is no main.py) ───────────
echo ">>> To run baseline open-loop eval:"
echo "    PYTHONPATH=$ROOT bash scripts/eval_drivevla.sh checkpoints/OpenDriveVLA-0.5B 8"
