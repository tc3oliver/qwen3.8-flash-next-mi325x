#!/usr/bin/env bash
# Build the benchmark corpus from real source code: the vLLM Python package inside the pinned image.
set -euo pipefail
cd "$(dirname "$0")"; source ../../deploy/.env
rm -rf corpus && mkdir corpus
docker run --rm --entrypoint tar "$VLLM_IMAGE" -C /usr/local/lib/python3.12/dist-packages -cf - \
  --exclude='__pycache__' vllm/v1 vllm/entrypoints vllm/model_executor/layers | tar -C corpus -xf -
find corpus -name '*.py' -size -2k -delete
echo "corpus: $(find corpus -name '*.py' | wc -l) files, $(du -sh corpus | cut -f1)"
