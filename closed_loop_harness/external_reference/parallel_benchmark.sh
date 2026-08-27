#!/bin/bash
# Parallel 12-scene NeuroNCAP benchmark across 4 GPU lanes (renderer+model each).
# fp16 / naive_w4 / group_w4_g128  x  N seeds, per scene x scenario category.
set -u
NRUNS="${1:-10}"
ND=/home/justin_williams1/neuroncap/neurad-studio
OD=/home/justin_williams1/OpenDriveVLA
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUT=/home/justin_williams1/neuroncap/neuro-ncap/output/benchmark12
SC=/tmp/claude-1012/-home-justin-williams1-OpenDriveVLA/daebd0c0-5f2e-4b21-9ce2-529f0f6f3334/scratchpad
mkdir -p "$OUT"

# lane -> "rgpu mgpu rport mport ; scene:cats scene:cats ..."
LANE0="2 0 8000 9000 ; 0106:frontal,stationary 0099:stationary 0101:stationary"
LANE1="3 1 8001 9001 ; 0108:side,stationary 0110:frontal,side"
LANE2="4 5 8002 9002 ; 0278:side,stationary 0331:stationary 0346:frontal"
LANE3="6 7 8003 9003 ; 0921:side 0923:frontal 0966:stationary 0783:stationary"

run_lane(){
  local spec="$1"; local head="${spec%%;*}"; local jobs="${spec#*; }"
  read rgpu mgpu rport mport <<< "$head"
  local lg="$OUT/lane_${rport}.log"
  echo "[lane $rport] gpus r=$rgpu m=$mgpu jobs: $jobs" > "$lg"
  # model node (scene-agnostic) once per lane
  pkill -9 -f "uvicorn inference.server:app --host 0.0.0.0 --port $mport" 2>/dev/null
  ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$mgpu ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
      ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 venv/bin/python -m uvicorn inference.server:app \
      --host 0.0.0.0 --port $mport >> "$lg" 2>&1 ) &
  for i in $(seq 1 120); do curl -s -m4 http://localhost:$mport/quant_cfg 2>/dev/null | grep -qi bits && break; timeout 5 tail -f /dev/null; done
  for job in $jobs; do
    local scene="${job%%:*}"; local cats="${job#*:}"
    echo "[lane $rport] === scene $scene ($cats) renderer load $(date +%H:%M:%S) ===" >> "$lg"
    pkill -9 -f "run_render_server.py --port $rport" 2>/dev/null
    ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$rgpu TORCH_COMPILE_DISABLE=1 \
        /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
        --port $rport --load-config checkpoints/$scene/config.yml --adjust_pose >> "$lg" 2>&1 ) &
    local ok=0; for i in $(seq 1 200); do curl -s -m4 http://localhost:$rport/alive 2>/dev/null|grep -q true && { ok=1; break; }; timeout 6 tail -f /dev/null; done
    [ $ok -eq 1 ] || { echo "[lane $rport] renderer $scene FAILED to start" >> "$lg"; continue; }
    IFS=',' read -ra CATARR <<< "$cats"
    for cat in "${CATARR[@]}"; do
      for cfg in "fp16|0|0|fp16" "naive_w4|4|0|rtn_sym" "group_w4_g128|4|128|rtn_sym"; do
        IFS='|' read name bits group mode <<< "$cfg"
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
  done
  echo "[lane $rport] LANE DONE $(date +%H:%M:%S)" >> "$lg"
}

d=0
for spec in "$LANE0" "$LANE1" "$LANE2" "$LANE3"; do
  ( [ $d -gt 0 ] && timeout $d tail -f /dev/null 2>/dev/null; run_lane "$spec" ) &
  d=$((d+150))
done
wait
echo "ALL LANES DONE $(date +%H:%M:%S)" > "$OUT/ALL_DONE"
