#!/bin/bash
# usage: bench.sh <image> <stock.py> <pr.py> <tag> ; one run of PR's bench_gqa.py; records load/GPU-use before+after
cd "$(dirname "$0")/../../.."; R=$PWD; O=$R/results/mi300-2026-11/phase1-hip3
IMG=$1; STOCK=$(realpath $2); PR=$(realpath $3); TAG=$4; L=$O/raw/bench_$TAG.txt
snap(){ echo "== $1 $(date +%T)"; uptime; rocm-smi --showuse 2>&1 | grep -E "GPU\[[0-9]\].*GPU use"; docker ps --format '{{.Names}}' | grep -E "mi300|vllm-qwen38|flash" ; }
{ snap before
docker run --rm --device /dev/kfd --device /dev/dri/renderD152 --group-add video --group-add $(getent group render | cut -d: -f3) --ipc host --security-opt seccomp=unconfined \
  -v $STOCK:/work/stock.py:ro -v $PR:/work/pr3.py:ro -v $R/upstream/vllm-62590-sparse-gqa-gfx942/evidence/scripts/bench_gqa.py:/work/bench_gqa.py:ro \
  -v $O/scripts:/s:ro --entrypoint python3 $IMG /work/bench_gqa.py
snap after; } > $L 2>&1; cat $L
