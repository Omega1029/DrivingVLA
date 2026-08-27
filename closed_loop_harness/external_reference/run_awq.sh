#!/bin/bash
# AWQ closed-loop evaluation on scene-0103 frontal, seed-0.
# Assumes the model node (port 9000) is freshly (re)started in FP and warm-loaded.
# 1) calibrate AWQ scales on the FP model over one closed-loop rollout (real rendered-domain
#    activations), 2) apply AWQ-W4, 3) score seed-0.  alpha grid configurable.
set -u
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUTROOT=/home/justin_williams1/neuroncap/neuro-ncap/output/quant_sweep
RESULTS=$OUTROOT/results_awq.tsv
mkdir -p "$OUTROOT"
echo -e "config\tbits\tgroup\tmode\talpha\tncap_score\timpact_speed" > "$RESULTS"

run_seed0 () {  # $1=logdir  $2=name
  rm -rf "$1"
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
