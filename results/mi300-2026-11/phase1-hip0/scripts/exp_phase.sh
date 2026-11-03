#!/usr/bin/env bash
# E9: does the pass/fail alternation follow the number of GPU processes created? A: pytest x12. B: (dummy GPU process, pytest) x10.
D=$(cd "$(dirname "$0")/.." && pwd); M=$D/main; PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
docker run --rm --name phase1-hip0-exp --device /dev/kfd --device /dev/dri/$1 --group-add video --group-add 993 --ipc host --security-opt seccomp=unconfined \
 -e TRITON_CACHE_DIR=/cache -v /home/oliver/projects/mct-ai-stack/vllm-services/mi300-verify/cache/phase1-hip0/triton-cache-main:/cache \
 -v $M/qsa.py:$PKG/ops/qsa.py:ro -v $M/qsa_indexer.py:$PKG/qsa_indexer.py:ro -v $M/test_qsa_amd.py:/t/test_qsa_amd.py:ro \
 --entrypoint bash vllm/vllm-openai-rocm:nightly -c '
T="/t/test_qsa_amd.py::test_qsa_select_decode_rows_replays_in_cuda_graph"
pt() { python3 -m pytest -q -p no:cacheprovider "$T" >/dev/null 2>&1 && printf P || printf F; }
echo -n "A pytest only:        "; for i in $(seq 12); do pt; done; echo
echo -n "B dummy+pytest:       "; for i in $(seq 10); do python3 -c "import torch;torch.zeros(1).cuda()" >/dev/null 2>&1; pt; done; echo
echo -n "C pytest only again:  "; for i in $(seq 10); do pt; done; echo'
