#!/bin/bash
# Free-drive (adversary-free) closed-loop sweep: ego follows the logged route through
# the NeuRAD renderer, only original reconstructed actors present. Measures route
# competence (no stay-in-place freeze confound). Mirrors run_benchmark_12.sh.
#   Usage: SCENES="0103" CONFIGS="fp16" ./run_freedrive.sh 3      # smoke
#          ./run_freedrive.sh 10                                   # full 14-scene sweep
set -u
NRUNS="${1:-10}"
SCENES="${SCENES:-0099 0101 0103 0106 0108 0110 0278 0331 0346 0783 0796 0921 0923 0966}"
# name|bits|group|mode
CONFIGS="${CONFIGS:-fp16 naive_w4 group_w4_g128}"
declare -A CFG=( [fp16]="0|0|fp16" [naive_w4]="4|0|rtn_sym" [group_w4_g128]="4|128|rtn_sym" )
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
ND=/home/justin_williams1/neuroncap/neurad-studio
OD=/home/justin_williams1/OpenDriveVLA
OUT=/home/justin_williams1/neuroncap/neuro-ncap/output/freedrive
mkdir -p "$OUT"
RG=2; MG=3; RP=8000; MP=9000   # single lane (renderer gpu2, model gpu3 — 0/1 are busy)

launch_lane(){  # $1 scene
  pkill -9 -f "run_render_server.py --port $RP" 2>/dev/null
  pkill -9 -f "uvicorn inference.server:app --host 0.0.0.0 --port $MP" 2>/dev/null
  sleep 2
  ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$RG TORCH_COMPILE_DISABLE=1 \
      /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
      --port $RP --load-config checkpoints/$1/config.yml --adjust_pose > $OUT/rend_${1}.log 2>&1 ) &
  ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$MG ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
      ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 venv/bin/python -m uvicorn inference.server:app \
      --host 0.0.0.0 --port $MP > $OUT/model_${1}.log 2>&1 ) &
  for i in $(seq 1 40); do curl -s -m4 http://localhost:$RP/alive|grep -q true && \
     curl -s -m4 http://localhost:$MP/quant_cfg|grep -qi bits && break; sleep 8; done
}

for s in $SCENES; do
  # 0103 and 0796 are nuScenes-mini; the rest are staged under v1.0-trainval.
  if [[ "$s" == "0103" || "$s" == "0796" ]]; then VER=v1.0-mini; else VER=v1.0-trainval; fi
  launch_lane "$s"
  for name in $CONFIGS; do
    IFS='|' read bits group mode <<< "${CFG[$name]}"
    curl -s -m120 -X POST http://localhost:$MP/quantize -H Content-Type:application/json \
      -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}" >/dev/null
    ld="$OUT/${s}_freedrive_${name}"; rm -rf "$ld"
    echo "=== $s / freedrive / $name (ver=$VER, runs=$NRUNS) ==="
    $NCV -u main.py --engine.renderer.port $RP --engine.model.port $MP \
      --engine.dataset.data_root data/nuscenes --engine.dataset.version $VER \
      --engine.dataset.sequence ${s#0} --engine.logger.log-dir "$ld" \
      --scenario-category freedrive --runs "$NRUNS" >> "$OUT/${s}_freedrive_${name}.log" 2>&1
  done
done
pkill -9 -f "run_render_server.py --port $RP" 2>/dev/null
pkill -9 -f "uvicorn inference.server:app --host 0.0.0.0 --port $MP" 2>/dev/null
echo "FREEDRIVE DONE ($SCENES)"
