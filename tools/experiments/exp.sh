#!/usr/bin/env bash
# Start one experiment engine on a GPU slot, with the deploy/compose.yaml settings as base.
# Run from tools/. Reads ../deploy/.env (VLLM_IMAGE, MODEL_DIR, REPLICA_*_HIP).
#   experiments/exp.sh <slot a|b> <name> [VAR=val ...] [-- extra vllm args...]
# VAR=val pairs become container env (override base env); BASE_<flag>=x replaces a base flag:
#   experiments/exp.sh a mtp1 BASE_speculative-config='{"method":"mtp","num_speculative_tokens":1}'
#   experiments/exp.sh b nomtp BASE_speculative-config=        (empty value drops the flag)
# Container: qwen38-exp-<slot> (replaces any previous qwen38-exp-<slot>; never touches qwen38-vllm-*).
# Waits until /health is OK (or the container dies) and prints the endpoint.
set -euo pipefail
cd "$(dirname "$0")/.."; set -a; source ../deploy/.env; set +a
REPO=$(cd .. && pwd)
slot=$1 name=$2; shift 2
case $slot in a) HIP=$REPLICA_A_HIP PORT=8011;; b) HIP=$REPLICA_B_HIP PORT=8012;;
  g0) HIP=1 PORT=8020;; g1) HIP=3 PORT=8021;; g2) HIP=2 PORT=8022;; g3) HIP=0 PORT=8023;;
  *) echo "slot a|b|g0..g3"; exit 1;; esac
[[ $slot == g* ]] && TP1=1 || TP1=0
C=qwen38-exp-$slot
declare -A BASE=(
  [tensor-parallel-size]=2 [enable-expert-parallel]= [max-model-len]=262144 [max-num-seqs]=64
  [gpu-memory-utilization]=0.93 [max-num-batched-tokens]=32768 [enable-chunked-prefill]= [enable-prefix-caching]=
  [enable-prompt-tokens-details]= [async-scheduling]=
  [speculative-config]='{"method":"mtp","num_speculative_tokens":3}'
  [engram-config]='{"cpu_offload":false}' [compilation-config]='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  [enable-auto-tool-choice]= [tool-call-parser]=qwen3_coder [reasoning-parser]=qwen3
)
declare -A ENV=(
  [HF_HUB_OFFLINE]=1 [VLLM_CACHE_ROOT]=/cache/vllm [VLLM_ROCM_USE_AITER]=1 [VLLM_ROCM_USE_AITER_MOE]=1
  [VLLM_ROCM_USE_AITER_CUSTOM_AR]=0 [VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT]=0 [SAFETENSORS_FAST_GPU]=1
  [HIP_FORCE_DEV_KERNARG]=1 [HSA_NO_SCRATCH_RECLAIM]=1 [VLLM_WORKER_MULTIPROC_METHOD]=spawn
  [TORCH_BLAS_PREFER_HIPBLASLT]=1 [OMP_NUM_THREADS]=8
)
EXTRA=(); MOUNTS=()
if ((TP1)); then BASE[tensor-parallel-size]=1; unset "BASE[enable-expert-parallel]"; fi
while (($#)); do
  case $1 in
    --) shift; EXTRA=("$@"); break;;
    BASE_*=*) k=${1%%=*}; k=${k#BASE_}; v=${1#*=}; if [[ -z $v && $k != enable-* && $k != async-scheduling ]]; then unset "BASE[$k]"; else BASE[$k]=$v; fi;;
    BASEDEL_*) k=${1#BASEDEL_}; unset "BASE[$k]";;
    PATCH_QSA=1) MOUNTS+=(-v "$REPO/overlay/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro");;
    MOUNT=*) MOUNTS+=(-v "${1#MOUNT=}");;
    HIP=*) HIP=${1#HIP=};;
    *=*) ENV[${1%%=*}]=${1#*=};;
    *) echo "bad arg $1"; exit 1;;
  esac; shift
done
ARGS=(/models/qwen3.8-flash-next-fp8 --served-model-name=qwen3.8-flash-next --host=0.0.0.0 --port=8000)
for k in "${!BASE[@]}"; do [[ -n ${BASE[$k]} ]] && ARGS+=("--$k=${BASE[$k]}") || ARGS+=("--$k"); done
ARGS+=("${EXTRA[@]}")
EV=(); for k in "${!ENV[@]}"; do EV+=(-e "$k=${ENV[$k]}"); done
mkdir -p runtime/exp-$slot experiments/results experiments/profiles
docker rm -f "$C" >/dev/null 2>&1 || true
LOG=experiments/results/$name.server.log
echo "[$name] slot=$slot HIP=$HIP port=$PORT"; printf '  %s\n' "${ARGS[@]:4}" "${EV[@]}" "${MOUNTS[@]}" | grep -v '^  -e$' > experiments/results/$name.config.txt
docker run -d --name "$C" --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined -p 127.0.0.1:$PORT:8000 \
  -e HIP_VISIBLE_DEVICES=$HIP "${EV[@]}" "${MOUNTS[@]}" \
  -v "$MODEL_DIR":/models/qwen3.8-flash-next-fp8:ro -v "$PWD/runtime/exp-$slot":/cache/vllm \
  -v "$PWD/experiments/profiles":/profiles -v "$REPO/tuned/aiter":/tuned:ro \
  "$VLLM_IMAGE" "${ARGS[@]}" >/dev/null
t0=$SECONDS
until curl -fsS localhost:$PORT/health >/dev/null 2>&1; do
  if [[ $(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null) != true ]]; then
    docker logs "$C" > "$LOG" 2>&1; echo "[$name] FAILED to start ($((SECONDS-t0))s); log: $LOG"; grep -E 'Error|error:' "$LOG" | tail -5; exit 1; fi
  (( SECONDS - t0 > 2400 )) && { docker logs "$C" > "$LOG" 2>&1; echo "[$name] TIMEOUT"; exit 1; }
  sleep 10
done
docker logs "$C" > "$LOG" 2>&1
echo "[$name] healthy after $((SECONDS-t0))s at http://127.0.0.1:$PORT"
