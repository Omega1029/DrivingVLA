#!/bin/bash
# Prepare the 12 non-mini NeuRAD-checkpointed scenes for closed-loop once v1.0-trainval data is present.
# (a) synth CAN-bus from each scene's lidar ego-track  (b) leave configs at v1.0-trainval (data is trainval).
set -u
SCENES=(0099 0101 0106 0108 0110 0278 0331 0346 0783 0921 0923 0966)
SYN=/tmp/claude-1012/-home-justin-williams1-OpenDriveVLA/daebd0c0-5f2e-4b21-9ce2-529f0f6f3334/scratchpad/synth_canbus.py
VENV=/home/justin_williams1/OpenDriveVLA/venv/bin/python
for s in "${SCENES[@]}"; do
  tmp=/tmp/synth_${s}.py
  sed -e "s/scene-0103/scene-${s}/" -e "s/v1.0-mini/v1.0-trainval/" "$SYN" > "$tmp"
  echo "=== synth canbus scene-${s} ==="; $VENV "$tmp" 2>&1 | tail -2
done
echo "PREP DONE"
