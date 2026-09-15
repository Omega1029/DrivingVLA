#!/bin/bash
# AWQ closed-loop evaluation on scene-0103 frontal, seed-0 WITH RENDERER STATE RESTORE.
#
# The original run_awq.sh (snapshot in external_reference/) assumes a renderer already running on
# :8000, runs an AWQ calibration rollout against it, then two AWQ-W4 scoring rollouts -- zero
# /update_actors restores anywhere. Flagged in PROGRESS.md 2026-08-27 10:05: if the renderer was
# already mutated before this script started (e.g. by a prior sweep sharing the same port), the
# calibration rollout itself runs on contaminated activations, which would make logs/awq_scales.pt
# a second-order-contaminated artifact, not just the scores measured with it.
#
# Fix: capture actor state once at startup and restore it before every rollout, including the
# calibration rollout. This does not retroactively fix any awq_scales.pt already produced by the
# original script -- recalibrate from scratch with this version if that's suspected.
#
# NOTE: like run_quant_sweep_clean.sh, this script does not launch the renderer itself -- run it
# immediately after a fresh scene-0103 render server start, before anything else hits port 8000.
# Assumes the model node (port 9000) is freshly (re)started in FP and warm-loaded.
set -u
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUTROOT=/home/justin_williams1/neuroncap/neuro-ncap/output/quant_sweep_clean
RESULTS=$OUTROOT/results_awq.tsv
mkdir -p "$OUTROOT"
echo -e "config\tbits\tgroup\tmode\talpha\tncap_score\timpact_speed" > "$RESULTS"

# *** the fix: capture actor state once, at the top, before any rollout ***
curl -s -m120 "http://localhost:8000/get_actors" > "$OUTROOT/pristine_0103.json"
N_PRISTINE=$(python3 -c "import json;print(len(json.load(open('$OUTROOT/pristine_0103.json'))))" 2>/dev/null || echo ERR)
echo "captured actor snapshot, actors=$N_PRISTINE"
if [ "$N_PRISTINE" = "ERR" ] || [ "$N_PRISTINE" = "0" ]; then
  echo "FATAL: actor snapshot capture failed -- is the renderer up on :8000?"; exit 1
fi
restore_actors () {
  curl -s -m120 -X POST http://localhost:8000/update_actors -H Content-Type:application/json \
    -d @"$OUTROOT/pristine_0103.json" >/dev/null
}

run_seed0 () {  # $1=logdir  $2=name
  rm -rf "$1"
  restore_actors
  $NCV -u main.py \
    --engine.renderer.port 8000 --engine.model.port 9000 \
    --engine.dataset.data_root data/nuscenes --engine.dataset.version v1.0-mini \
    --engine.dataset.sequence 103 --engine.logger.log-dir "$1" \
    --scenario-category frontal --runs 1 > "$OUTROOT/$2.log" 2>&1
  grep -oE "ncap_score: [0-9.]+" "$OUTROOT/$2.log" | tail -1 | grep -oE "[0-9.]+"
}

# --- 1. calibrate AWQ on FP over one rollout ---
echo "=== AWQ calibration rollout (FP, capturing activations) ==="
curl -s -m 30 -X POST http://localhost:9000/quantize -d '{"bits":0,"mode":"fp16"}' >/dev/null
curl -s -m 30 -X POST http://localhost:9000/awq_calib_start >/dev/null
run_seed0 "$OUTROOT/awq_calib_rollout" "awq_calib_rollout" >/dev/null
curl -s -m 60 -X POST "http://localhost:9000/awq_calib_finish?alpha=0.5" ; echo

# --- 2. AWQ-W4 at a few group sizes ---
for grp in 128 64; do
  echo "=== AWQ-W4 group=$grp ==="
  curl -s -m 120 -X POST http://localhost:9000/quantize \
    -d "{\"bits\":4,\"group\":$grp,\"mode\":\"awq\"}" ; echo
  name="w4_awq_g${grp}"
  score=$(run_seed0 "$OUTROOT/$name" "$name")
  impact=$(grep -oE "impact_speed: [0-9.]+" "$OUTROOT/$name.log" | tail -1 | grep -oE "[0-9.]+")
  echo "=== [$name] ncap_score=$score impact=$impact ==="
  echo -e "$name\t4\t$grp\tawq\t0.5\t$score\t$impact" >> "$RESULTS"
done

echo "=== AWQ DONE ==="
cat "$RESULTS"
