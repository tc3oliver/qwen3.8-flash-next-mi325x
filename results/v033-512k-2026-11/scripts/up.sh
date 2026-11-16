#!/usr/bin/env bash
# TP2+EP2 replica shaped like deploy/compose.yaml tp2x2. Args via env:
#   NAME, PORT, RENDERS="160 168", YARN=2 (512K, factor 2) | 1 (1M, factor 4) | 0 (stock 262144); YaRN via config.json copy, TAG (a|b for cache dirs)
# Flags = compose.yaml x-cmd-tp2 + x-tp2-env, plus VLLM_ALLOW_LONG_MAX_MODEL_LEN for 1M.
set -euo pipefail
R=${R:?repo root}; W=${W:-$PWD}; ORIG=${ORIG_DIR:?original model dir}   # run from the work dir that holds these scripts
IMG=${IMG:-vllm/vllm-openai-rocm:v0.32.0}   # v0.32.0 = sha256:14f6a575...
: "${NAME:?}" "${PORT:?}" "${RENDERS:?}" "${TAG:?}"; YARN=${YARN:-1}; UTIL=${UTIL:-0.93}
if [ "$YARN" = 2 ]; then LEN=524288; MODEL_DIR=$W/model-yarn512; [ -n "${DEPLOY:-}" ] && MODEL_DIR=$W/deploy/model-yarn512
elif [ "$YARN" = 1 ]; then LEN=1000000; MODEL_DIR=$W/model-yarn1m; else LEN=262144; MODEL_DIR=$ORIG; fi
DEV=(); for r in $RENDERS; do DEV+=(--device /dev/dri/renderD$r); done
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --ipc=host --shm-size 32g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd "${DEV[@]}" --group-add video --group-add "$(getent group render | cut -d: -f3)" -p 127.0.0.1:$PORT:8000 \
  -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_CUSTOM_AR=0 -e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0 -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 -e OMP_NUM_THREADS=8 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e AITER_CONFIG_FMOE=/usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv:/aiter-tuned/tuned_fmoe_qwen38_ep2.csv \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6 -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 ${QRMAX:+-e VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=$QRMAX} \
  -v $MODEL_DIR:/models/qwen3.8-flash-next-fp8:ro -v $ORIG:$ORIG:ro ${DEPLOY:+-v $ORIG:/models-orig:ro} \
  -v $R/tuned/aiter:/aiter-tuned:ro -v $W/aiter-$TAG:/root/.aiter -v $W/cache-$TAG:/cache/vllm \
  $( if [ -n "${OVL33:-}" ]; then echo "-v $R/overlay/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro -v $R/overlay/indexer_qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/indexer_qsa.py:ro"; else echo "-v $R/overlay/history/qsa.2026-10-20-d02.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro"; fi ) \
  $IMG /models/qwen3.8-flash-next-fp8 --served-model-name=qwen3.8-flash-next --host=0.0.0.0 --port=8000 \
  --tensor-parallel-size=2 --enable-expert-parallel --max-model-len=$LEN --max-num-seqs=64 \
  --gpu-memory-utilization=$UTIL --max-num-batched-tokens=32768 --enable-chunked-prefill --enable-prefix-caching \
  --enable-prompt-tokens-details --async-scheduling \
  --speculative-config='{"method":"mtp","num_speculative_tokens":3}' \
  --engram-config='{"cpu_offload":false}' --compilation-config='{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3
