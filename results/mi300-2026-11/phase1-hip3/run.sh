#!/bin/bash
# usage: run.sh <name> <cmd...>  ; runs inside v0.32.0 on renderD152 only (HIP 0 inside), overlay mounted ro, scripts at /s, out at /o
cd "$(dirname "$0")/../../.."   # repo root
PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
R=$PWD; O=$R/results/mi300-2026-11/phase1-hip3
IMG=${IMG:-vllm/vllm-openai-rocm:v0.32.0}
docker run --rm --device /dev/kfd --device /dev/dri/renderD152 --group-add video --group-add $(getent group render | cut -d: -f3) --ipc host \
  --security-opt seccomp=unconfined -e PYTHONDONTWRITEBYTECODE=1 \
  -v $R/overlay/qsa.py:$PKG/ops/qsa.py:ro -v $R/overlay/indexer_qsa.py:$PKG/indexer_qsa.py:ro \
  -v $R/overlay/test_qsa_kernels.py:/t/test_qsa_kernels.py:ro -v $O/scripts:/s:ro \
  --entrypoint "${1:-python3}" $IMG "${@:2}"
