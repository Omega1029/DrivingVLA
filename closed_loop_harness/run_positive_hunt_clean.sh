#!/bin/bash
# POSITIVE-HUNT sweep WITH RENDERER STATE RESTORE.
#
# The original run_positive_hunt.sh (snapshot in external_reference/) has the same renderer-reuse
# defect as run_benchmark_12.sh / run_freedrive.sh: launch_lane starts one renderer per scene and
# then loops CONFIGS x categories inside it, with zero /update_actors restores. Worse here: config
# order is `w8 awq_w4_g128 w4a4_dct`, so w8 always ran first against a clean scene and both 4-bit
# recipes ran against whatever w8's rollout left behind -- see PROGRESS.md 2026-08-27 10:05 for why
# this is the likely explanation for w8 looking clean and the 4-bit recipes looking broken in this
# sweep specifically.
#
# Fix, identical to run_sweep_clean.sh / run_freedrive_clean.sh: capture the pristine actor list
# once per scene right after a fresh renderer launch, and POST it back before EVERY main.py
# invocation (both the freedrive run and every adversarial category run, for every config).
#
#   Lane-parametrized: RG/MG/RP/MP (renderer/model gpu+port), SCENES.
#   Usage: RG=2 MG=3 RP=8000 MP=9000 SCENES="0099 0101 ..." ./run_positive_hunt_clean.sh 10
set -u
NRUNS="${1:-10}"
SCENES="${SCENES:-0099 0101 0103 0106 0108 0110 0278 0331 0346 0783 0796 0921 0923 0966}"
RG="${RG:-2}"; MG="${MG:-3}"; RP="${RP:-8000}"; MP="${MP:-9000}"
# The paper's 16 adversarial instances (scene -> categories); freedrive added for every scene.
declare -A CATS=( [0099]="stationary" [0101]="stationary" [0103]="frontal" [0106]="frontal" \
  [0108]="side stationary" [0110]="frontal" [0278]="side stationary" [0331]="stationary" \
  [0346]="frontal" [0783]="stationary" [0796]="stationary" [0921]="side" [0923]="frontal" \
  [0966]="stationary" )
# name -> JSON body for /quantize
declare -A CFG=(
  [w8]='{"bits":8,"group":0,"mode":"rtn_sym"}'
  [awq_w4_g128]='{"bits":4,"group":128,"mode":"awq"}'
  [w4a4_dct]='{"bits":4,"group":0,"mode":"rtn_sym","a_bits":4,"rot":"dct"}'
)
CONFIGS="${CONFIGS:-w8 awq_w4_g128 w4a4_dct}"
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
ND=/home/justin_williams1/neuroncap/neurad-studio
OD=/home/justin_williams1/OpenDriveVLA
OUT=/home/justin_williams1/neuroncap/neuro-ncap/output/positive_hunt_clean
mkdir -p "$OUT"

# Kill whatever holds a port and WAIT for actual release -- same fix as run_sweep_clean.sh. The
# original's pkill -9 -f fired-and-forgot, which let the next launch_lane's wait loop succeed
# against the dying old process instead of the fresh one.
kill_port () {
  local port=$1 tries=0 p
  while :; do
    p=$(ss -lptn "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -z "$p" ] && return 0
    kill -9 "$p" 2>/dev/null
    sleep 2; tries=$((tries+1))
    [ $tries -ge 30 ] && { echo "FATAL: port $port still held after 60s"; return 1; }
  done
}

launch_lane(){  # $1 scene
  kill_port $RP; kill_port $MP; sleep 2
  ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$RG TORCH_COMPILE_DISABLE=1 \
      exec /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
      --port $RP --load-config checkpoints/$1/config.yml --adjust_pose > $OUT/rend_${1}_p${RP}.log 2>&1 ) &
  ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$MG ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
      ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 exec venv/bin/python -m uvicorn inference.server:app \
      --host 0.0.0.0 --port $MP > $OUT/model_${1}_p${MP}.log 2>&1 ) &
  for i in $(seq 1 40); do curl -s -m4 http://localhost:$RP/alive|grep -q true && \
     curl -s -m4 http://localhost:$MP/quant_cfg|grep -qi bits && break; sleep 8; done
}

for s in $SCENES; do
  if [[ "$s" == "0103" || "$s" == "0796" ]]; then VER=v1.0-mini; else VER=v1.0-trainval; fi
  launch_lane "$s"
  # *** the fix: capture pristine actor state once, right after the fresh renderer comes up ***
  curl -s -m120 "http://localhost:$RP/get_actors" > "$OUT/pristine_${s}.json"
  n_pristine=$(python3 -c "import json;print(len(json.load(open('$OUT/pristine_${s}.json'))))" 2>/dev/null || echo ERR)
  echo "scene $s up (ver=$VER), pristine actors=$n_pristine"
  if [ "$n_pristine" = "ERR" ] || [ "$n_pristine" = "0" ]; then
    echo "!! $s pristine capture failed, skipping scene"; continue
  fi
  for name in $CONFIGS; do
    body="${CFG[$name]}"
    for cat in freedrive ${CATS[$s]:-}; do
      # *** the fix: restore pristine actor state before every invocation ***
      curl -s -m120 -X POST "http://localhost:$RP/update_actors" \
           -H Content-Type:application/json -d @"$OUT/pristine_${s}.json" >/dev/null
      curl -s -m180 -X POST http://localhost:$MP/quantize -H Content-Type:application/json \
        -d "$body" > "$OUT/${s}_${cat}_${name}.quant.json"
      ld="$OUT/${s}_${cat}_${name}"; rm -rf "$ld"
      echo "=== $s / $cat / $name (ver=$VER, runs=$NRUNS, lane $RP/$MP) ==="
      $NCV -u main.py --engine.renderer.port $RP --engine.model.port $MP \
        --engine.dataset.data_root data/nuscenes --engine.dataset.version $VER \
        --engine.dataset.sequence ${s#0} --engine.logger.log-dir "$ld" \
        --scenario-category $cat --runs "$NRUNS" >> "$OUT/${s}_${cat}_${name}.log" 2>&1
    done
  done
done
kill_port $RP; kill_port $MP
echo "POSITIVE HUNT CLEAN DONE (lane $RP/$MP: $SCENES)"
