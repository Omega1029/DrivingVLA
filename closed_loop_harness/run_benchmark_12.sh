#!/bin/bash
# Full benchmark over the 12 non-mini ckpt scenes x their scenario categories x {fp16,naive_w4,group_w4}
# x N seeds. Uses 4 GPU pairs (renderer+model) round-robin for parallelism. Run AFTER prep_12_scenes.sh
# and after v1.0-trainval data is staged under data/nuscenes.
set -u
NRUNS="${1:-10}"
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
ND=/home/justin_williams1/neuroncap/neurad-studio
OD=/home/justin_williams1/OpenDriveVLA
OUT=/home/justin_williams1/neuroncap/neuro-ncap/output/benchmark12
mkdir -p "$OUT"
# scene -> scenario categories it has a yaml for
declare -A CATS=( [0099]="stationary" [0101]="stationary" [0106]="frontal stationary" \
  [0108]="side stationary" [0110]="frontal side" [0278]="side stationary" [0331]="stationary" \
  [0346]="frontal" [0783]="stationary" [0921]="side" [0923]="frontal" [0966]="stationary" )
# GPU pairs (renderer_gpu:model_gpu:rport:mport) — 4 lanes
LANES=( "2:0:8000:9000" "3:1:8001:9001" "5:4:8002:9002" "7:6:8003:9003" )
launch_lane(){  # $1 scene $2 lane
  IFS=: read rg mg rp mp <<< "${LANES[$2]}"
  ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$rg TORCH_COMPILE_DISABLE=1 \
      /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
      --port $rp --load-config checkpoints/$1/config.yml --adjust_pose > $OUT/rend_${1}.log 2>&1 ) &
  ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$mg ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
      ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 venv/bin/python -m uvicorn inference.server:app \
      --host 0.0.0.0 --port $mp > $OUT/model_${1}.log 2>&1 ) &
  # wait for both up
  for i in $(seq 1 40); do curl -s -m4 http://localhost:$rp/alive|grep -q true && \
     curl -s -m4 http://localhost:$mp/quant_cfg|grep -qi bits && break; sleep 8; done
}
# NOTE: sequential per-scene here for clarity; lanes allow parallelizing across scenes if desired.
for s in "${!CATS[@]}"; do
  IFS=: read rg mg rp mp <<< "${LANES[0]}"
  pkill -9 -f "run_render_server.py --port $rp" 2>/dev/null; pkill -9 -f "uvicorn inference.server:app --host 0.0.0.0 --port $mp" 2>/dev/null
  launch_lane "$s" 0
  for cat in ${CATS[$s]}; do
    for cfg in "fp16|0|0|fp16" "naive_w4|4|0|rtn_sym" "group_w4_g128|4|128|rtn_sym"; do
      IFS='|' read name bits group mode <<< "$cfg"
      curl -s -m120 -X POST http://localhost:$mp/quantize -H Content-Type:application/json \
        -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}" >/dev/null
      ld="$OUT/${s}_${cat}_${name}"; rm -rf "$ld"
      $NCV -u main.py --engine.renderer.port $rp --engine.model.port $mp \
        --engine.dataset.data_root data/nuscenes --engine.dataset.version v1.0-trainval \
        --engine.dataset.sequence ${s#0} --engine.logger.log-dir "$ld" \
        --scenario-category $cat --runs "$NRUNS" >> "$OUT/${s}_${cat}_${name}.log" 2>&1
    done
  done
done
echo "BENCHMARK12 DONE"
