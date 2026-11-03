#!/usr/bin/env bash
# usage: repeat.sh <image> <tag> <N>   HIP0 only (renderD128). Fresh pytest process per run.
set -u
IMG=$1 TAG=$2 N=$3;  # optional env TESTFILE=<path> to swap the test
 R=$(cd "$(dirname "$0")/../../../.." && pwd); D=$R/results/mi300-2026-11/phase1-hip0
PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
mkdir -p $D/runs/$TAG
docker run --rm --device /dev/kfd --device /dev/dri/renderD128 --group-add video --group-add render --ipc host --security-opt seccomp=unconfined \
  -e TRITON_CACHE_DIR=/cache -e N=$N -e TAG=$TAG -v $D/triton-cache:/cache -v $D/runs/$TAG:/out \
  -v $R/overlay/qsa.py:$PKG/ops/qsa.py:ro -v $R/overlay/indexer_qsa.py:$PKG/indexer_qsa.py:ro \
  -v ${TESTFILE:-$R/overlay/test_qsa_kernels.py}:/t/test_qsa_kernels.py:ro --entrypoint bash $IMG -c '
p=0; f=0
for i in $(seq 1 $N); do
  if python3 -m pytest -q -p no:cacheprovider "/t/test_qsa_kernels.py::test_qsa_select_decode_rows_replays_in_cuda_graph" >/out/run$i.log 2>&1; then p=$((p+1)); else f=$((f+1)); fi
done
echo "RESULT $TAG pass=$p fail=$f" | tee /out/summary.txt'
