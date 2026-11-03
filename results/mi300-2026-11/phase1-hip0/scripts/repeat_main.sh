#!/usr/bin/env bash
# usage: repeat_main.sh <N> [renderD128]   nightly image + PR #62589 head files mounted read-only over the image
set -u
N=$1 RN=${2:-renderD128}; D=$(cd "$(dirname "$0")/.." && pwd); M=$D/main
PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd; TAG=nightly-pr62589
mkdir -p $D/runs/$TAG
docker run --rm --name phase1-hip0-main --device /dev/kfd --device /dev/dri/$RN --group-add video --group-add 993 --ipc host --security-opt seccomp=unconfined \
  -e TRITON_CACHE_DIR=/cache -e N=$N -v $D/triton-cache-main:/cache -v $D/runs/$TAG:/out \
  -v $M/qsa.py:$PKG/ops/qsa.py:ro -v $M/qsa_indexer.py:$PKG/qsa_indexer.py:ro -v $M/test_qsa_amd.py:/t/test_qsa_amd.py:ro \
  --entrypoint bash vllm/vllm-openai-rocm:nightly -c '
p=0; f=0
for i in $(seq 1 $N); do
  if python3 -m pytest -q -p no:cacheprovider "/t/test_qsa_amd.py::test_qsa_select_decode_rows_replays_in_cuda_graph" >/out/run$i.log 2>&1; then p=$((p+1)); else f=$((f+1)); fi
done
echo "RESULT nightly+pr62589 pass=$p fail=$f" | tee /out/summary.txt'
