#!/usr/bin/env bash
# Isolated TP4+EP4 1M-context experiment on HIP4-7 (renderD160/168/176/184 on our host).
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)   # repo root
W=${W:-$R/runtime/lc1m}
MODEL_DIR=${MODEL_DIR:?model dir}
IMG=${IMG:-vllm/vllm-openai-rocm:v0.32.0}
NAME=lc1m-tp4
docker rm -f $NAME >/dev/null 2>&1 || true
mkdir -p $W/logs
docker run -d --name $NAME --ipc=host --shm-size 32g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd $(for d in 160 168 176 184; do echo --device /dev/dri/renderD$d; done) \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  -p 127.0.0.1:8190:8000 \
  -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_CUSTOM_AR=0 -e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0 -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 -e OMP_NUM_THREADS=8 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v $MODEL_DIR:/models/qwen3.8-flash-next-fp8:ro \
  -v $W/aiter:/root/.aiter -v $W/vllm-cache:/cache/vllm \
  -v $R/overlay/history/qsa.2026-10-20-d02.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro \
  $IMG /models/qwen3.8-flash-next-fp8 --served-model-name=qwen3.8-flash-next --host=0.0.0.0 --port=8000 \
  --tensor-parallel-size=4 --enable-expert-parallel --max-model-len=1000000 --max-num-seqs=1 \
  --max-num-batched-tokens=8192 --gpu-memory-utilization=0.90 --enable-chunked-prefill --enable-prefix-caching \
  --async-scheduling --engram-config='{"cpu_offload":false}' \
  --compilation-config='{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --hf-overrides='{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}}}' \
  --reasoning-parser=qwen3
