#!/bin/bash
# Closed-loop sweep with RENDERER STATE RESTORE.
#
# The NeuRAD render server is stateful: POST /update_actors overwrites its actor set and
# neuro_ncap/engine.py reads that set ONCE at engine init. The old scripts launched one
# renderer per scene and looped configs inside it, so only the FIRST config per scene saw a
# clean actor set -- which confounded every quantization comparison with run order.
#
# Fix: capture the pristine actor list right after a fresh renderer launch, and POST it back
# before EVERY main.py invocation. Verified to reproduce the archive exactly (0099 stationary
# FP16 -> 5.0, 33 actors) on both a fresh and a restored renderer.
#
# Usage: run_sweep_clean.sh [NRUNS] [OUTDIR]
set -u
NRUNS="${1:-10}"
OUT="${2:-/home/justin_williams1/neuroncap/neuro-ncap/output/remeasure_clean}"
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
ND=/home/justin_williams1/neuroncap/neurad-studio
OD=/home/justin_williams1/OpenDriveVLA
NC=/home/justin_williams1/neuroncap/neuro-ncap
mkdir -p "$OUT"

declare -A CATS=( [0099]="stationary" [0101]="stationary" [0106]="frontal stationary" \
  [0108]="side stationary" [0110]="frontal side" [0278]="side stationary" [0331]="stationary" \
  [0346]="frontal" [0783]="stationary" [0921]="side" [0923]="frontal" [0966]="stationary" )

CONFIGS=( "fp16|0|0|fp16" "w8|8|0|rtn_sym" "naive_w4|4|0|rtn_sym" "group_w4_g128|4|128|rtn_sym" )

# lane -> gpu:rport:mport   (renderer and model co-locate; ~4 GB total on an 80 GB card)
LANES=( "1:8000:9000" "2:8001:9001" "3:8002:9002" "4:8003:9003" )
NLANES=${#LANES[@]}

# Kill whatever holds a port and WAIT until it is actually released. The previous version
# fired kill -9 and moved on; the old renderer survived long enough that the next scene's
# wait_up() succeeded against IT, so scene N+1 ran against scene N's actor set (KeyError on a
# foreign uuid, or silently wrong geometry). Never trust kill without confirming the release.
kill_port () {
  local port=$1 tries=0 p
  while :; do
    p=$(ss -lptn "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -z "$p" ] && return 0
    kill -9 "$p" 2>/dev/null
    sleep 2; tries=$((tries+1))
    [ $tries -ge 30 ] && { echo "FATAL: port $port still held by $p after 60s"; return 1; }
  done
}

start_model () {  # gpu port log
  ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
      ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 \
      exec venv/bin/python -m uvicorn inference.server:app --host 0.0.0.0 --port $2 > $3 2>&1 ) &
}
start_render () {  # gpu port scene log
  ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 TORCH_COMPILE_DISABLE=1 \
      exec /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
      --port $2 --load-config checkpoints/$3/config.yml --adjust_pose > $4 2>&1 ) &
}
wait_up () {  # url tries
  for _ in $(seq 1 ${2:-60}); do curl -s -m10 "$1" >/dev/null 2>&1 && return 0; sleep 5; done; return 1
}

run_lane () {  # lane_idx scenes...
  local li=$1; shift
  IFS=: read gpu rp mp <<< "${LANES[$li]}"
  kill_port $rp; kill_port $mp; sleep 2
  start_model $gpu $mp "$OUT/model_lane$li.log"
  wait_up "http://localhost:$mp/quant_cfg" 90 || { echo "lane$li model FAILED"; return 1; }

  for s in "$@"; do
    kill_port $rp; sleep 3
    start_render $gpu $rp "$s" "$OUT/rend_${s}.log"
    wait_up "http://localhost:$rp/alive" 90 || { echo "lane$li rend $s FAILED"; continue; }
    # capture pristine actor state for this scene
    curl -s -m120 "http://localhost:$rp/get_actors" > "$OUT/pristine_${s}.json"
    local n_pristine; n_pristine=$(python3 -c "import json;print(len(json.load(open('$OUT/pristine_${s}.json'))))" 2>/dev/null || echo ERR)
    echo "[lane$li] scene $s renderer up, pristine actors=$n_pristine"
    if [ "$n_pristine" = "ERR" ] || [ "$n_pristine" = "0" ]; then
      echo "[lane$li] !! scene $s pristine capture failed -- skipping scene"; continue
    fi

    for cat in ${CATS[$s]}; do
      for cfg in "${CONFIGS[@]}"; do
        IFS='|' read name bits group mode <<< "$cfg"
        # *** the fix: restore pristine actor state before every invocation ***
        curl -s -m120 -X POST "http://localhost:$rp/update_actors" \
             -H Content-Type:application/json -d @"$OUT/pristine_${s}.json" >/dev/null
        curl -s -m180 -X POST "http://localhost:$mp/quantize" \
             -H Content-Type:application/json -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}" >/dev/null
        local ld="$OUT/${s}_${cat}_${name}"; rm -rf "$ld"
        ( cd $NC && $NCV -u main.py --engine.renderer.port $rp --engine.model.port $mp \
            --engine.dataset.data_root data/nuscenes --engine.dataset.version v1.0-trainval \
            --engine.dataset.sequence ${s#0} --engine.logger.log-dir "$ld" \
            --scenario-category $cat --runs "$NRUNS" > "$OUT/${s}_${cat}_${name}.log" 2>&1 )
        local got; got=$(grep -o 'ncap_score: [0-9.]*' "$OUT/${s}_${cat}_${name}.log" | sed 's/ncap_score: //' | tr '\n' ' ')
        if [ -z "$got" ]; then
          echo "[lane$li] !! $s $cat $name produced NO SCORES -- $(grep -m1 -E 'Error|error|Traceback' "$OUT/${s}_${cat}_${name}.log" | cut -c1-90)"
        else
          echo "[lane$li] $s $cat $name -> $got"
        fi
      done
    done
  done
  kill_port $rp; kill_port $mp
  echo "[lane$li] DONE"
}

# deal scenes round-robin across lanes
SCENES=( "${!CATS[@]}" )
for li in $(seq 0 $((NLANES-1))); do
  assigned=()
  for i in "${!SCENES[@]}"; do [ $((i % NLANES)) -eq $li ] && assigned+=("${SCENES[$i]}"); done
  echo "lane$li scenes: ${assigned[*]}"
  ( sleep $((li * 100)); run_lane $li "${assigned[@]}" ) &
done
wait
echo "SWEEP_CLEAN_DONE"
