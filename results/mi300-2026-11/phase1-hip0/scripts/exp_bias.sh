#!/usr/bin/env bash
D=$(cd "$(dirname "$0")/.." && pwd); M=$D/main; PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
docker run --rm --name phase1-hip0-exp --device /dev/kfd --device /dev/dri/$1 --group-add video --group-add 993 --ipc host --security-opt seccomp=unconfined \
 -e TRITON_CACHE_DIR=/cache -v /home/oliver/projects/mct-ai-stack/vllm-services/mi300-verify/cache/phase1-hip0/triton-cache-main:/cache -v $D/scripts:/s:ro \
 -v $M/qsa.py:$PKG/ops/qsa.py:ro -v $M/qsa_indexer.py:$PKG/qsa_indexer.py:ro -v $M/test_qsa_amd.py:/t/test_qsa_kernels.py:ro \
 --entrypoint bash vllm/vllm-openai-rocm:nightly -c 'for i in $(seq 1 10); do echo "--- proc $i"; python3 /s/bias.py 2>&1 | grep "^BIAS"; done'
