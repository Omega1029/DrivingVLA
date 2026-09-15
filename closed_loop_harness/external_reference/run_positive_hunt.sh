#!/bin/bash
# POSITIVE-HUNT sweep: the three recipes that held NCAP=5.0 in single-scene pilots
# (INT8, AWQ-W4-g128, W4A4+DCT) run at full scale on BOTH closed-loop benchmarks:
#   - adversarial NeuroNCAP (paper's 16 scene/scenario instances) -> graded NCAP success
#   - free-drive (14 scenes)                                      -> route competence
# Single-scene pilots are proven liars here (group-W4 also looked perfect on one scene),
# so whichever of these survives 14-scene scale is the paper's positive headline.
#   Lane-parametrized: RG/MG/RP/MP (renderer/model gpu+port), SCENES.
#   Usage: RG=2 MG=3 RP=8000 MP=9000 SCENES="0099 0101 ..." ./run_positive_hunt.sh 10
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
OUT=/home/justin_williams1/neuroncap/neuro-ncap/output/positive_hunt
mkdir -p "$OUT"

launch_lane(){  # $1 scene
  pkill -9 -f "run_render_server.py --port $RP" 2>/dev/null
  pkill -9 -f "uvicorn inference.server:app --host 0.0.0.0 --port $MP" 2>/dev/null
  sleep 2
  ( cd $ND && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$RG TORCH_COMPILE_DISABLE=1 \
      /home/justin_williams1/neuroncap/neurad-venv/bin/python -u run_render_server.py \
      --port $RP --load-config checkpoints/$1/config.yml --adjust_pose > $OUT/rend_${1}_p${RP}.log 2>&1 ) &
  ( cd $OD && PYTHONPATH=. CUDA_VISIBLE_DEVICES=$MG ODV_CKPT=checkpoints/OpenDriveVLA-0.5B \
      ODV_DEVICE=cuda:0 ODV_AUTOCAST=float16 venv/bin/python -m uvicorn inference.server:app \
      --host 0.0.0.0 --port $MP > $OUT/model_${1}_p${MP}.log 2>&1 ) &
  for i in $(seq 1 40); do curl -s -m4 http://localhost:$RP/alive|grep -q true && \
     curl -s -m4 http://localhost:$MP/quant_cfg|grep -qi bits && break; sleep 8; done
}

for s in $SCENES; do
  if [[ "$s" == "0103" || "$s" == "0796" ]]; then VER=v1.0-mini; else VER=v1.0-trainval; fi
  launch_lane "$s"
  for name in $CONFIGS; do
    body="${CFG[$name]}"
    for cat in freedrive ${CATS[$s]:-}; do
      # re-apply quant before every category run (cheap; guards against node hiccups)
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
pkill -9 -f "run_render_server.py --port $RP" 2>/dev/null
pkill -9 -f "uvicorn inference.server:app --host 0.0.0.0 --port $MP" 2>/dev/null
echo "POSITIVE HUNT DONE (lane $RP/$MP: $SCENES)"
