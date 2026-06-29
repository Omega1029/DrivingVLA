#!/bin/bash
# Multi-seed validation on scene-0103 frontal (original scenario, all seeds now run via the
# renderable-actor-substitution fix). For each quantization config: re-quantize the live node,
# run seeds 0..N-1, record per-seed ncap_score + collision. Answers: does naive-W4 collide on
# multiple perturbations, and does group-wise W4 recover across that set?
set -u
NRUNS="${1:-10}"
NCV=/home/justin_williams1/neuroncap/ncap-venv/bin/python
OUTROOT=/home/justin_williams1/neuroncap/neuro-ncap/output/multiseed
mkdir -p "$OUTROOT"
SUMMARY=$OUTROOT/summary.tsv
echo -e "config\tmean_ncap\tn_collisions\tn_runs\tper_seed_scores" > "$SUMMARY"
Q="curl -s -m 120 -X POST http://localhost:9000/quantize -H Content-Type:application/json"

# name|bits|group|mode
CONFIGS=(
  "naive_w4|4|0|rtn_sym"
  "group_w4_g128|4|128|rtn_sym"
  "fp16|0|0|fp16"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name bits group mode <<< "$cfg"
  echo "=== [$name] quantize bits=$bits group=$group mode=$mode ==="
  $Q -d "{\"bits\":$bits,\"group\":$group,\"mode\":\"$mode\"}"; echo
  logdir="$OUTROOT/$name"
  rm -rf "$logdir"
  $NCV -u main.py \
    --engine.renderer.port 8000 --engine.model.port 9000 \
    --engine.dataset.data_root data/nuscenes --engine.dataset.version v1.0-mini \
    --engine.dataset.sequence 103 --engine.logger.log-dir "$logdir" \
    --scenario-category frontal --runs "$NRUNS" > "$OUTROOT/$name.log" 2>&1
  # aggregate per-seed metrics
  $NCV - "$logdir" "$name" "$SUMMARY" <<'PY'
import sys, json, glob, os
logdir, name, summary = sys.argv[1], sys.argv[2], sys.argv[3]
scores, ncol = [], 0
for d in sorted(glob.glob(os.path.join(logdir, "run_*"))):
    mp = os.path.join(d, "metrics.json")
    if not os.path.exists(mp): continue
    m = json.load(open(mp))
    s = m.get("ncap_score")
    scores.append(round(s,2) if s is not None else None)
    if m.get("any_collide@0.0s"): ncol += 1
valid = [s for s in scores if s is not None]
mean = round(sum(valid)/len(valid),3) if valid else None
with open(summary, "a") as f:
    f.write(f"{name}\t{mean}\t{ncol}\t{len(scores)}\t{scores}\n")
print(f"[{name}] mean_ncap={mean} collisions={ncol}/{len(scores)} per_seed={scores}")
PY
done
echo "=== MULTISEED DONE ==="
column -t -s$'\t' "$SUMMARY"
