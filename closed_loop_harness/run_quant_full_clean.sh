#!/bin/bash
# Closed-loop quantization technique sweep on scene-0103 frontal SEED-0 WITH RENDERER STATE
# RESTORE.
#
# The original run_quant_full.sh (snapshot in external_reference/) assumes a renderer already
# running on :8000 and calls record() repeatedly against it with zero /update_actors restores --
# same defect class as run_benchmark_12.sh. Also flagged in PROGRESS.md 2026-08-27 10:05: the AWQ
# calibration rollout in this script runs on whatever state the renderer is in after six prior RTN
# rollouts, so logs/awq_scales.pt from this script's original run may itself be calibrated on
# contaminated activations -- a second-order contamination that this fix alone does not undo. This
# clean version at least makes every rollout (RTN sweep AND the AWQ calibration rollout) start
# from the same captured actor state, so re-running this script gives a trustworthy comparison
# even though the ORIGINAL awq_scales.pt this produced historically may not be.
#
# NOTE: like run_quant_sweep_clean.sh, this script does not launch the renderer itself -- run it
# immediately after a fresh scene-0103 render server start, before anything else hits port 8000.
#
# Requires the model node (FP) up on :9000 with /quantize + /awq_calib_* endpoints, original
# scenario (3 uuids; seed-0 picks the available 2cc8 actor).
set -u
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUTROOT=/home/justin_williams1/neuroncap/neuro-ncap/output/quant_full_clean
RESULTS=$OUTROOT/results.tsv
mkdir -p "$OUTROOT"
echo -e "config\tbits\tgroup\tmode\tncap_score\timpact_speed" > "$RESULTS"
Q="curl -s -m 120 -X POST http://localhost:9000/quantize -H Content-Type:application/json"

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

run_seed0 () {  # $1=logdir $2=name -> echoes score
  rm -rf "$1"
  restore_actors
  $NCV -u main.py \
    --engine.renderer.port 8000 --engine.model.port 9000 \
    --engine.dataset.data_root data/nuscenes --engine.dataset.version v1.0-mini \
    --engine.dataset.sequence 103 --engine.logger.log-dir "$1" \
    --scenario-category frontal --runs 1 > "$OUTROOT/$2.log" 2>&1
  grep -oE "ncap_score: [0-9.]+" "$OUTROOT/$2.log" | tail -1 | grep -oE "[0-9.]+"
}
record () {  # $1=name $2=bits $3=group $4=mode
  local name=$1 bits=$2 group=$3 mode=$4
  echo "=== [$name] bits=$bits group=$group mode=$mode ==="
  $Q -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}"; echo
  local score; score=$(run_seed0 "$OUTROOT/$name" "$name")
  local impact; impact=$(grep -oE "impact_speed: [0-9.]+" "$OUTROOT/$name.log" | tail -1 | grep -oE "[0-9.]+")
  echo "=== [$name] ncap_score=$score impact=$impact ==="
  echo -e "$name\t$bits\t$group\t$mode\t$score\t$impact" >> "$RESULTS"
}

# --- RTN variants + bit-width curve ---
record fp16              0   0   fp16
record w8_rtn_sym_g0     8   0   rtn_sym
record w4_rtn_sym_g0     4   0   rtn_sym
record w4_rtn_sym_g128   4   128 rtn_sym
record w4_rtn_asym_g128  4   128 rtn_asym
record w4_rtn_asym_g64   4   64  rtn_asym
record w3_rtn_sym_g128   3   128 rtn_sym

# --- AWQ: calibrate on a live FP rollout, then AWQ-W4 ---
echo "=== AWQ calibration rollout (FP, capturing activations) ==="
$Q -d '{"bits":0,"mode":"fp16"}' >/dev/null
curl -s -m 30 -X POST http://localhost:9000/awq_calib_start >/dev/null
run_seed0 "$OUTROOT/awq_calib_rollout" "awq_calib_rollout" >/dev/null
curl -s -m 60 -X POST "http://localhost:9000/awq_calib_finish?alpha=0.5"; echo
record w4_awq_g128       4   128 awq
record w4_awq_g64        4   64  awq

echo "=== FULL SWEEP DONE ==="
column -t "$RESULTS"
