#!/bin/bash
# Parallel 12-scene NeuroNCAP benchmark across 4 GPU lanes WITH RENDERER STATE RESTORE.
#
# The original parallel_benchmark.sh (snapshot in external_reference/) has the same renderer-reuse
# defect as run_benchmark_12.sh: one renderer per scene, categories and configs looped inside it
# with zero /update_actors restores between them. Only the first (cat, config) pair per scene saw
# a clean actor set; everything after that ran against whatever the previous rollout left behind.
#
# Fix, identical to run_sweep_clean.sh: capture the pristine actor list once per scene right after
# the renderer comes up, and POST it back before EVERY main.py invocation.
set -u
NRUNS="${1:-10}"
ND=/home/justin_williams1/neuroncap/neurad-studio
OD=/home/justin_williams1/OpenDriveVLA
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUT=/home/justin_williams1/neuroncap/neuro-ncap/output/benchmark12_clean
mkdir -p "$OUT"

# lane -> "rgpu mgpu rport mport ; scene:cats scene:cats ..."
LANE0="2 0 8000 9000 ; 0106:frontal,stationary 0099:stationary 0101:stationary"
LANE1="3 1 8001 9001 ; 0108:side,stationary 0110:frontal,side"
LANE2="4 5 8002 9002 ; 0278:side,stationary 0331:stationary 0346:frontal"
LANE3="6 7 8003 9003 ; 0921:side 0923:frontal 0966:stationary 0783:stationary"

# Kill whatever holds a port and WAIT for actual release -- same fix as run_sweep_clean.sh. The
# original's pkill -9 -f fired-and-forgot, which let the next scene's wait loop succeed against
# the dying old renderer instead of the fresh one.
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

run_lane(){
  local spec="$1"; local head="${spec%%;*}"; local jobs="${spec#*; }"
  read rgpu mgpu rport mport <<< "$head"
  local lg="$OUT/lane_${rport}.log"
  echo "[lane $rport] gpus r=$rgpu m=$mgpu jobs: $jobs" > "$lg"
  kill_port $mport
  ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$mgpu ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
      ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 exec venv/bin/python -m uvicorn inference.server:app \
      --host 0.0.0.0 --port $mport >> "$lg" 2>&1 ) &
  for i in $(seq 1 120); do curl -s -m4 http://localhost:$mport/quant_cfg 2>/dev/null | grep -qi bits && break; sleep 5; done
  for job in $jobs; do
    local scene="${job%%:*}"; local cats="${job#*:}"
    echo "[lane $rport] === scene $scene ($cats) renderer load $(date +%H:%M:%S) ===" >> "$lg"
    kill_port $rport; sleep 3
    ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$rgpu TORCH_COMPILE_DISABLE=1 \
        exec /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
        --port $rport --load-config checkpoints/$scene/config.yml --adjust_pose >> "$lg" 2>&1 ) &
    local ok=0; for i in $(seq 1 200); do curl -s -m4 http://localhost:$rport/alive 2>/dev/null|grep -q true && { ok=1; break; }; sleep 6; done
    [ $ok -eq 1 ] || { echo "[lane $rport] renderer $scene FAILED to start" >> "$lg"; continue; }
    # *** the fix: capture pristine actor state once, right after the fresh renderer comes up ***
    curl -s -m120 "http://localhost:$rport/get_actors" > "$OUT/pristine_${scene}.json"
    local n_pristine; n_pristine=$(python3 -c "import json;print(len(json.load(open('$OUT/pristine_${scene}.json'))))" 2>/dev/null || echo ERR)
    echo "[lane $rport] scene $scene pristine actors=$n_pristine" >> "$lg"
    if [ "$n_pristine" = "ERR" ] || [ "$n_pristine" = "0" ]; then
      echo "[lane $rport] !! scene $scene pristine capture failed, skipping" >> "$lg"; continue
    fi
    IFS=',' read -ra CATARR <<< "$cats"
    for cat in "${CATARR[@]}"; do
      for cfg in "fp16|0|0|fp16" "naive_w4|4|0|rtn_sym" "group_w4_g128|4|128|rtn_sym"; do
        IFS='|' read name bits group mode <<< "$cfg"
        # *** the fix: restore pristine actor state before every invocation ***
        curl -s -m120 -X POST "http://localhost:$rport/update_actors" \
             -H Content-Type:application/json -d @"$OUT/pristine_${scene}.json" >/dev/null
        curl -s -m120 -X POST http://localhost:$mport/quantize -H Content-Type:application/json \
          -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}" >/dev/null
        local ld="$OUT/${scene}_${cat}_${name}"; rm -rf "$ld"
        echo "[lane $rport] run $scene/$cat/$name $(date +%H:%M:%S)" >> "$lg"
        (cd /home/justin_williams1/neuroncap/neuro-ncap && $NCV -u main.py \
          --engine.renderer.port $rport --engine.model.port $mport \
          --engine.dataset.data_root data/nuscenes --engine.dataset.version v1.0-trainval \
          --engine.dataset.sequence ${scene#0} --engine.logger.log-dir "$ld" \
          --scenario-category $cat --runs "$NRUNS" >> "$OUT/${scene}_${cat}_${name}.log" 2>&1) || \
          echo "[lane $rport] run $scene/$cat/$name ERROR" >> "$lg"
      done
    done
    kill_port $rport
  done
  echo "[lane $rport] LANE DONE $(date +%H:%M:%S)" >> "$lg"
}

d=0
for spec in "$LANE0" "$LANE1" "$LANE2" "$LANE3"; do
  ( sleep $d; run_lane "$spec" ) &
  d=$((d+150))
done
wait
echo "ALL LANES DONE $(date +%H:%M:%S)" > "$OUT/ALL_DONE"
