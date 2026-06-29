#!/bin/bash
set -u
cd /home/justin_williams1/neuroncap/neuro-ncap/data/nuscenes
NEED=/tmp/claude-1012/-home-justin-williams1-OpenDriveVLA/daebd0c0-5f2e-4b21-9ce2-529f0f6f3334/scratchpad/needed_files.txt
PY=/home/justin_williams1/OpenDriveVLA/venv/bin/python
TOTAL=$(wc -l < "$NEED")
# Phase 1: parallel resumable downloads of all 10 blobs
for n in 01 02 03 04 05 06 07 08 09 10; do
  url="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval${n}_blobs.tgz"
  wget -c -q "$url" -O blob_${n}.tgz &
done
echo "launched 10 parallel downloads $(date +%H:%M:%S)"
wait
echo "all downloads finished $(date +%H:%M:%S)"
# Phase 2: selective extract + cleanup
for n in 01 02 03 04 05 06 07 08 09 10; do
  [ -s blob_${n}.tgz ] || { echo "blob $n missing/empty"; continue; }
  tar -xzf blob_${n}.tgz -T "$NEED" --skip-old-files 2>/dev/null || true
  rm -f blob_${n}.tgz
  echo "extracted+removed blob $n $(date +%H:%M:%S)"
done
have=$($PY -c "import os;n=[l.strip() for l in open('$NEED')];print(sum(os.path.exists(f) for f in n))")
echo "FETCH DONE coverage $have / $TOTAL $(date +%H:%M:%S)"
