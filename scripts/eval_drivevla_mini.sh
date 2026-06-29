#!/usr/bin/env bash
# Bring-up eval on nuScenes-mini (the 162 mini samples that fall in the official
# val split). Requires the gated map expansion v1.3 unpacked under data/nuscenes/maps/.
#
#   bash scripts/eval_drivevla_mini.sh [CKPT_PATH] [NUM_GPU]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT_PATH="${1:-checkpoints/OpenDriveVLA-0.5B}"
NUM_GPU="${2:-1}"
EXTRA_FLAGS="${3:-}"   # e.g. "--load-4bit" or "--load-8bit"
TAG="${4:-mini}"

# Point the perception dataset at the 162-sample subset infos (stamped v1.0-mini).
export ODV_VAL_INFO="nuscenes_infos_temporal_val_mini.pkl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_RESULT_DIR="output/$(basename "$CKPT_PATH")_${TAG}/${TIMESTAMP}"
mkdir -p "${LOG_RESULT_DIR}/log" "${LOG_RESULT_DIR}/results"
EVAL_LOG_FILE="${LOG_RESULT_DIR}/log/eval.log"
PLAN_CONV_PATH="${LOG_RESULT_DIR}/results/plan_conv.json"

echo ">>> MINI eval | ckpt=${CKPT_PATH} gpus=${NUM_GPU} | $(date)" | tee -a "${EVAL_LOG_FILE}"

VENV="$(pwd)/venv/bin"
PYTHONPATH="$(pwd)":${PYTHONPATH:-} \
"${VENV}/torchrun" --nproc_per_node="${NUM_GPU}" --master_port="${MASTER_PORT:-29500}" \
    drivevla/inference_drivevla.py \
    --num-workers 4 --bf16 ${EXTRA_FLAGS} \
    --model-path "${CKPT_PATH}" \
    --output "${PLAN_CONV_PATH}" \
    2>&1 | tee -a "${EVAL_LOG_FILE}"

echo ">>> Evaluating ${PLAN_CONV_PATH}" | tee -a "${EVAL_LOG_FILE}"
PYTHONPATH="$(pwd)":${PYTHONPATH:-} \
"${VENV}/python" drivevla/eval_drivevla.py --output "${PLAN_CONV_PATH}" 2>&1 | tee -a "${EVAL_LOG_FILE}"
