#!/usr/bin/env bash
# usage: exp_run.sh <mode> <renderD> ; modes: single (one container, N fresh procs) | sleep | k (loop k=0..9)
D=$(cd "$(dirname "$0")/.." && pwd); M=$D/main; PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
docker run --rm --name phase1-hip0-exp --device /dev/kfd --device /dev/dri/$2 --group-add video --group-add 993 --ipc host --security-opt seccomp=unconfined \
 -e TRITON_CACHE_DIR=/cache -e MODE=$1 -v $D/../../../../serving-stack/vllm-services/mi300-verify/cache/phase1-hip0/triton-cache-main:/cache -v $D/scripts:/s:ro \
 -v $M/qsa.py:$PKG/ops/qsa.py:ro -v $M/qsa_indexer.py:$PKG/qsa_indexer.py:ro -v $M/test_qsa_amd.py:/t/test_qsa_kernels.py:ro \
 --entrypoint bash vllm/vllm-openai-rocm:nightly -c '
case $MODE in
 single) for i in $(seq 1 10); do echo "run$i $(uptime | sed "s/.*load/load/")"; python3 /s/fp.py 2>&1 | grep ^FP; done;;
 sleep) for i in $(seq 1 10); do echo "run$i"; python3 /s/fp.py 2>&1 | grep ^FP; sleep 5; done;;
 k) for k in 0 1 2 3 4 5 6 7 8 9; do echo "k=$k"; python3 /s/fp.py $k 2>&1 | grep ^FP; done;;
esac'
