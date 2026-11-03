#!/bin/bash
# usage: run_nightly.sh <tag> [grid]  ; stock = image's own ops/qsa.py (copy nightly/stock_qsa.py), PR = nightly/pr_qsa.py (== PR head 404857c8)
D=$(cd "$(dirname "$0")" && pwd); T=$1
IMG=vllm/vllm-openai-rocm@sha256:5800b3b4cd79defc1cbcc66adb83b3c49e574b49ca5c6ba556e17388515dc84f
{ date; uptime; rocm-smi --showuse | grep 'GPU use'; } > $D/nightly-$T-pre.txt
docker run --rm --name p1hip1-nightly --device /dev/kfd --device /dev/dri/renderD136 --group-add video --group-add $(getent group render | cut -d: -f3) --ipc host \
  -e HIP_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 -e GRID=${2:-full} -v $D:/work:ro --entrypoint bash $IMG \
  -c 'python3 -I /work/bench_qsa.py /work/nightly/stock_qsa.py /work/nightly/pr_qsa.py' > $D/nightly-$T.log 2>&1
{ date; uptime; rocm-smi --showuse | grep 'GPU use'; } > $D/nightly-$T-post.txt
