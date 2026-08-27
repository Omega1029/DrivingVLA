#!/bin/bash
# Closed-loop quantization-technique sweep on scene-0103 frontal, seed-0 (the perturbation where
# FP16 avoids the crash and naive RTN-W4 collides). For each config: re-quantize the live model
# node in-place via /quantize, run one seed-0 rollout, record ncap_score + impact_speed.
set -u
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUTROOT=/home/justin_williams1/neuroncap/neuro-ncap/output/quant_sweep
RESULTS=$OUTROOT/results.tsv
mkdir -p "$OUTROOT"
echo -e "config\tbits\tgroup\tmode\tncap_score\timpact_speed" > "$RESULTS"

# name | bits | group | mode
CONFIGS=(
  "fp16|0|0|fp16"
  "w4_rtn_sym_g0|4|0|rtn_sym"
  "w4_rtn_sym_g128|4|128|rtn_sym"
  "w4_rtn_asym_g128|4|128|rtn_asym"
  "w4_rtn_asym_g64|4|64|rtn_asym"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name bits group mode <<< "$cfg"
  echo "=== [$name] quantize bits=$bits group=$group mode=$mode ==="
  curl -s -m 120 -X POST http://localhost:9000/quantize -H 'Content-Type: application/json' \
    -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}" ; echo
  logdir="$OUTROOT/$name"
  rm -rf "$logdir"
  $NCV -u main.py \
    --engine.renderer.port 8000 --engine.model.port 9000 \
    --engine.dataset.data_root data/nuscenes --engine.dataset.version v1.0-mini \
    --engine.dataset.sequence 103 \
    --engine.logger.log-dir "$logdir" \
    --scenario-category frontal --runs 1 > "$OUTROOT/$name.log" 2>&1
  score=$(grep -oE "ncap_score: [0-9.]+" "$OUTROOT/$name.log" | tail -1 | grep -oE "[0-9.]+")
  impact=$(grep -oE "impact_speed: [0-9.]+" "$OUTROOT/$name.log" | tail -1 | grep -oE "[0-9.]+")
  echo "=== [$name] ncap_score=$score impact=$impact ==="
  echo -e "$name\t$bits\t$group\t$mode\t$score\t$impact" >> "$RESULTS"
done

echo "=== SWEEP DONE ==="
cat "$RESULTS"
