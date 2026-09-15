#!/bin/bash
# Closed-loop quantization-technique sweep on scene-0103 frontal, seed-0 WITH RENDERER STATE
# RESTORE.
#
# The original run_quant_sweep.sh (snapshot in external_reference/) assumes a renderer is already
# running on :8000 and loops CONFIGS against it sequentially with zero /update_actors restores --
# the same defect class as run_benchmark_12.sh, just without this script owning the renderer's
# launch. Only the first config saw whatever actor state the renderer happened to be in when this
# script started; every config after that ran against whatever the previous rollout's collision
# left behind.
#
# Fix: capture the actor state present on :8000 once at startup and POST it back before every
# config's rollout. NOTE: this only guarantees identical starting state ACROSS configs within one
# run of this script -- it does NOT guarantee that starting state is the scene's true pristine
# state, since this script never launches the renderer itself. Run this immediately after a fresh
# `run_render_server.py` launch for scene 0103, before anything else touches port 8000.
set -u
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUTROOT=/home/justin_williams1/neuroncap/neuro-ncap/output/quant_sweep_clean
RESULTS=$OUTROOT/results.tsv
mkdir -p "$OUTROOT"
echo -e "config\tbits\tgroup\tmode\tncap_score\timpact_speed" > "$RESULTS"

# *** the fix: capture actor state once, at the top, before any config runs ***
curl -s -m120 "http://localhost:8000/get_actors" > "$OUTROOT/pristine_0103.json"
N_PRISTINE=$(python3 -c "import json;print(len(json.load(open('$OUTROOT/pristine_0103.json'))))" 2>/dev/null || echo ERR)
echo "captured actor snapshot, actors=$N_PRISTINE"
if [ "$N_PRISTINE" = "ERR" ] || [ "$N_PRISTINE" = "0" ]; then
  echo "FATAL: actor snapshot capture failed -- is the renderer up on :8000?"; exit 1
fi

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
  # *** the fix: restore captured actor state before every invocation ***
  curl -s -m120 -X POST http://localhost:8000/update_actors -H Content-Type:application/json \
    -d @"$OUTROOT/pristine_0103.json" >/dev/null
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
