#!/bin/bash
# Free-drive (adversary-free) closed-loop sweep WITH RENDERER STATE RESTORE.
#
# Why this exists: the original run_freedrive.sh (snapshot in external_reference/) has the same
# defect as the old run_benchmark_12.sh -- it launches one renderer per scene via launch_lane and
# then loops configs inside it, with no /update_actors restore between them. The NeuRAD server is
# stateful: /update_actors overwrites its actor set, and neuro_ncap/engine.py:79 reads that set
# ONCE at engine init. So only the first config per scene sees a clean scene, and since fp16 is
# always first in the config list, contamination is perfectly confounded with the treatment. That
# defect produced a fake 4-bit safety collapse in the adversarial benchmark (see
# papers/four_bits_without_loss.tex); the free-drive numbers were generated the same way and are
# therefore unverified until re-measured here.
#
# Fix, identical to run_sweep_clean.sh: capture the pristine actor list once per scene right after
# a fresh renderer launch, and POST it back before EVERY main.py invocation.
#
# Usage: run_freedrive_clean.sh [NRUNS] [OUTDIR]
set -u
NRUNS="${1:-10}"
OUT="${2:-/home/justin_williams1/neuroncap/neuro-ncap/output/freedrive_clean}"
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
ND=/home/justin_williams1/neuroncap/neurad-studio
OD=/home/justin_williams1/OpenDriveVLA
NC=/home/justin_williams1/neuroncap/neuro-ncap
mkdir -p "$OUT"

SCENES="${SCENES:-0099 0101 0103 0106 0108 0110 0278 0331 0346 0783 0796 0921 0923 0966}"
CONFIGS=( "fp16|0|0|fp16" "w8|8|0|rtn_sym" "naive_w4|4|0|rtn_sym" "group_w4_g128|4|128|rtn_sym" )
LANES=( "1:8000:9000" "2:8001:9001" "3:8002:9002" "4:8003:9003" )
NLANES=${#LANES[@]}

# Kill whatever holds a port and WAIT for actual release. pkill -f (as the original used) both
# risks matching the caller's own command line and returns before the port is free -- which lets
# the next wait_up succeed against the dying old server.
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
start_model ()  { ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
                    ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 \
                    exec venv/bin/python -m uvicorn inference.server:app --host 0.0.0.0 --port $2 > $3 2>&1 ) & }
start_render () { ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$1 TORCH_COMPILE_DISABLE=1 \
                    exec /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
                    --port $2 --load-config checkpoints/$3/config.yml --adjust_pose > $4 2>&1 ) & }
wait_up () { for _ in $(seq 1 ${2:-90}); do curl -s -m10 "$1" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }

run_lane () {
  local li=$1; shift
  IFS=: read gpu rp mp <<< "${LANES[$li]}"
  kill_port $rp; kill_port $mp; sleep 2
  start_model $gpu $mp "$OUT/model_lane$li.log"
  wait_up "http://localhost:$mp/quant_cfg" 90 || { echo "[lane$li] model FAILED"; return 1; }

  for s in "$@"; do
    # 0103 and 0796 are nuScenes-mini; the rest are staged under v1.0-trainval.
    local VER; if [[ "$s" == "0103" || "$s" == "0796" ]]; then VER=v1.0-mini; else VER=v1.0-trainval; fi
    kill_port $rp; sleep 3
    start_render $gpu $rp "$s" "$OUT/rend_${s}.log"
    wait_up "http://localhost:$rp/alive" 90 || { echo "[lane$li] !! rend $s FAILED"; continue; }
    curl -s -m120 "http://localhost:$rp/get_actors" > "$OUT/pristine_${s}.json"
    local np; np=$(python3 -c "import json;print(len(json.load(open('$OUT/pristine_${s}.json'))))" 2>/dev/null || echo ERR)
    echo "[lane$li] scene $s up (ver=$VER), pristine actors=$np"
    if [ "$np" = "ERR" ] || [ "$np" = "0" ]; then echo "[lane$li] !! $s pristine capture failed, skipping"; continue; fi

    for cfg in "${CONFIGS[@]}"; do
      IFS='|' read name bits group mode <<< "$cfg"
      # *** the fix: restore pristine actor state before every invocation ***
      curl -s -m120 -X POST "http://localhost:$rp/update_actors" \
           -H Content-Type:application/json -d @"$OUT/pristine_${s}.json" >/dev/null
      curl -s -m180 -X POST "http://localhost:$mp/quantize" \
           -H Content-Type:application/json -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}" >/dev/null
      local ld="$OUT/${s}_freedrive_${name}"; rm -rf "$ld"
      ( cd $NC && $NCV -u main.py --engine.renderer.port $rp --engine.model.port $mp \
          --engine.dataset.data_root data/nuscenes --engine.dataset.version $VER \
          --engine.dataset.sequence ${s#0} --engine.logger.log-dir "$ld" \
          --scenario-category freedrive --runs "$NRUNS" > "$OUT/${s}_freedrive_${name}.log" 2>&1 )
      local n_done; n_done=$(ls -d "$ld"/run_* 2>/dev/null | wc -l)
      if [ "$n_done" -eq 0 ]; then
        echo "[lane$li] !! $s $name produced NO RUNS -- $(grep -m1 -E 'Error|Traceback' "$OUT/${s}_freedrive_${name}.log" | cut -c1-80)"
      else
        echo "[lane$li] $s $name -> $n_done runs"
      fi
    done
  done
  kill_port $rp; kill_port $mp
  echo "[lane$li] DONE"
}

SC=( $SCENES )
for li in $(seq 0 $((NLANES-1))); do
  assigned=()
  for i in "${!SC[@]}"; do [ $((i % NLANES)) -eq $li ] && assigned+=("${SC[$i]}"); done
  echo "lane$li scenes: ${assigned[*]}"
  ( sleep $((li * 100)); run_lane $li "${assigned[@]}" ) &
done
wait
echo "FREEDRIVE_CLEAN_DONE"
