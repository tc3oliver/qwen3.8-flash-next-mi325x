#!/bin/bash
# exact command used for the v0.32.0 stock-vs-PR run on HIP 1 (renderD136 only)
D=$(cd "$(dirname "$0")" && pwd)
IMG=vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905
{ date; uptime; rocm-smi --showuse | grep 'GPU use'; } > $D/main-pre.txt
docker run --rm --name p1hip1-bench --device /dev/kfd --device /dev/dri/renderD136 --group-add video --group-add render --ipc host \
  -e HIP_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 -v $D:/work:ro --entrypoint bash $IMG \
  -c 'python3 -I /work/bench_qsa.py /work/main/main_qsa.py /work/main/pr_head_qsa.py' > $D/main-run1.log 2>&1
{ date; uptime; rocm-smi --showuse | grep 'GPU use'; } > $D/main-post.txt
