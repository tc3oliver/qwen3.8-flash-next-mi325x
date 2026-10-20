#!/bin/bash
# usage: HIP=<n> repotest.sh <worktree> <qsa.py to mount> <pytest args...>
WT=$1; QSA=$2; shift 2
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc host -e PYTHONDONTWRITEBYTECODE=1 -e HIP_VISIBLE_DEVICES=${HIP:?set HIP} -w /work --entrypoint bash \
  -v $WT/tests:/work/tests:ro -v $WT/pyproject.toml:/work/pyproject.toml:ro \
  -v $QSA:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro \
  vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905 \
  -c "python3 -m pytest --version >/dev/null 2>&1 || pip install -q pytest >/dev/null 2>&1; pip install -q tblib ${EXTRA_PIP} >/dev/null 2>&1; python3 -c 'import vllm,torch;print(vllm.__file__, torch.cuda.get_device_properties(0).gcnArchName)'; python3 -m pytest -p no:cacheprovider tests/models/qwen4_exp/test_qsa_amd.py $*"
