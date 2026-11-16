#!/usr/bin/env bash
# TP2+EP2, shaped like deploy/compose.yaml tp2x2 (MTP 3, 32768 batched tokens, QuickReduce, tuned EP2 MoE table) with max-model-len 1M + YaRN.
# GPUs HIP4+5 = renderD160/168 on our host. YARN=0 drops the rope override and uses max-model-len 262144.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd); W=${W:-$R/runtime/lc1m-tp2}   # R = repo root; runtime/ is git-ignored
IMG=${IMG:-vllm/vllm-openai-rocm:v0.32.0}; NAME=lc1m-tp2; YARN=${YARN:-1}; UTIL=${UTIL:-0.90}; MTP=${MTP:-1}; BATCH=${BATCH:-32768}; PREFIX=${PREFIX:-1}; EXTRA_ENV=${EXTRA_ENV:-}; MODEL_DIR=${MODEL_DIR:?model dir}; ORIG_DIR=${ORIG_DIR:-$MODEL_DIR}   # YARN=2: MODEL_DIR = config.json copy whose other files link into ORIG_DIR
if [ "$YARN" = 2 ]; then LEN=1000000; OVR=()  # YaRN comes from MODEL_DIR/config.json, so the MTP draft reads it too
elif [ "$YARN" = 1 ]; then LEN=1000000; OVR=(--hf-overrides='{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}}}'); else LEN=262144; OVR=(); fi
if [ "$MTP" = 1 ]; then SPEC=(--speculative-config='{"method":"mtp","num_speculative_tokens":3}'); else SPEC=(); fi
if [ "$PREFIX" = 1 ]; then PFX=(--enable-prefix-caching); else PFX=(--no-enable-prefix-caching); fi
ENVX=(); for kv in $EXTRA_ENV; do ENVX+=(-e "$kv"); done
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --ipc=host --shm-size 32g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri/renderD160 --device /dev/dri/renderD168 \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" -p 127.0.0.1:8191:8000 \
  -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e VLLM_ROCM_USE_AITER_CUSTOM_AR=0 -e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0 -e SAFETENSORS_FAST_GPU=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 -e OMP_NUM_THREADS=8 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e AITER_CONFIG_FMOE=/usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv:/aiter-tuned/tuned_fmoe_qwen38_ep2.csv \
  "${ENVX[@]}" -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6 -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  -v $MODEL_DIR:/models/qwen3.8-flash-next-fp8:ro -v $ORIG_DIR:$ORIG_DIR:ro \
  -v $R/tuned/aiter:/aiter-tuned:ro -v $W/aiter:/root/.aiter -v $W/vllm-cache:/cache/vllm \
  -v $R/overlay/history/qsa.2026-10-20-d02.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro \
  $IMG /models/qwen3.8-flash-next-fp8 --served-model-name=qwen3.8-flash-next --host=0.0.0.0 --port=8000 \
  --tensor-parallel-size=2 --enable-expert-parallel --max-model-len=$LEN --max-num-seqs=${SEQS:-64} \
  --max-num-batched-tokens=$BATCH --gpu-memory-utilization=$UTIL --enable-chunked-prefill \
  "${PFX[@]}" \
  --enable-prompt-tokens-details --async-scheduling \
  "${SPEC[@]}" \
  --engram-config='{"cpu_offload":false}' --compilation-config='{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  "${OVR[@]}" --reasoning-parser=qwen3
