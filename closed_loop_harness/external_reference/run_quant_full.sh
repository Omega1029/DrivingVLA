#!/bin/bash
# Closed-loop quantization technique sweep on scene-0103 frontal SEED-0, the perturbation where
# FP16 avoids the crash (5.0) and naive RTN-W4 collides (2.70). Question: which technique recovers
# W4 safety? Runs RTN variants + a bit-width curve, then AWQ (calibrated on a live FP rollout).
# Requires the model node (FP) up on :9000 with /quantize + /awq_calib_* endpoints, original
# scenario (3 uuids; seed-0 picks the available 2cc8 actor).
set -u
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUTROOT=/home/justin_williams1/neuroncap/neuro-ncap/output/quant_full
RESULTS=$OUTROOT/results.tsv
mkdir -p "$OUTROOT"
echo -e "config\tbits\tgroup\tmode\tncap_score\timpact_speed" > "$RESULTS"
Q="curl -s -m 120 -X POST http://localhost:9000/quantize -H Content-Type:application/json"

run_seed0 () {  # $1=logdir $2=name -> echoes score
  rm -rf "$1"
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
