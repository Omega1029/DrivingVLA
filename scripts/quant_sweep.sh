#!/usr/bin/env bash
# Weight-only fake-quant bit-width sweep on nuScenes-mini, with eval after each.
# Usage: bash scripts/quant_sweep.sh 8 3 2
set -uo pipefail
cd "$(dirname "$0")/.."
VENV="$(pwd)/venv/bin"
CKPT="checkpoints/OpenDriveVLA-0.5B"
rm -f logs/sweep_progress.log
for B in "$@"; do
  echo ">>> [$(date +%H:%M:%S)] W${B} inference" | tee -a logs/sweep_progress.log
  bash scripts/eval_drivevla_mini.sh "$CKPT" 1 "--wquant-bits ${B}" "miniW${B}" > "logs/eval_mini_w${B}.log" 2>&1
  PLAN=$(ls -t output/OpenDriveVLA-0.5B_miniW${B}/*/results/plan_conv.json 2>/dev/null | head -1)
  if [ -z "$PLAN" ]; then echo ">>> W${B} produced no predictions (see logs/eval_mini_w${B}.log)" | tee -a logs/sweep_progress.log; continue; fi
  echo ">>> [$(date +%H:%M:%S)] W${B} eval" | tee -a logs/sweep_progress.log
  PYTHONPATH="$(pwd)" "${VENV}/python" drivevla/eval_drivevla.py --output "$PLAN" > "logs/eval_metrics_w${B}.log" 2>&1
  echo "DONE_W${B}" | tee -a logs/sweep_progress.log
done
echo "ALL_SWEEP_DONE" | tee -a logs/sweep_progress.log
