# overlay/qsa.py

Mount read-only over `vllm/models/qwen4_exp/amd/ops/qsa.py` in `vllm/vllm-openai-rocm:v0.32.0`:

```
-v $PWD/overlay/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro
```

**Only valid for that image** (stock file sha256 `7a7c8501e0b92a4236e382c6ef494670d684ee1dfeb2d673f942a4f38fccf69b`,
overlay sha256 `2d0c0ee8e2e605604d973497659371644247999de975af6d3907e44bab0ccef7`). After any image change, re-port it
and re-check it with `tools/kernels/test_qsa.py`, or drop it.

Changes (everything else is the stock file):

* `_qsa_mqa_paged_kernel_v2` (default, `VLLM_QSA_MQA_IMPL=v2`): QSA indexer scoring with 16x64 tiles,
  `tl.dot` (MFMA), one key tile per request block, early exit past the visible range. The stock kernel
  scores every row against max_model_len/4 = 65,536 columns. 4.6-20x faster; logits |diff| <= 5e-7; every
  differing top-k pick is an exact score tie. `orig` / `v1` remain selectable for A/B.
* `VLLM_QSA_GQA_PREFILL="block_n,splits,warps,stages"`: overrides the sparse-GQA launch profile used for
  more than 512 programs (prefill). Use `32,1,2,1` (2.0x faster kernel on gfx942). Unset = stock profile.
  Do not use (64,1,4,1) or (128,1,4,1): they fault the GPU on gfx942.
* `VLLM_QSA_LOGITS_WORKSPACE_MB` (default 128 = stock).

Measured end to end: prefill +70%, decode c64 +41% (e02), then +9-10% prefill from the GQA profile (e14, t14).
GSM8K-1319 0.9666 vs stock 0.9621 (TP1). Details: `docs/lab-notebook.md`, raw data in `results/experiments/`.

## When it becomes unnecessary

| overlay part | upstream | 
|---|---|
| v2 indexer kernel | [vllm-project/vllm#62589](https://github.com/vllm-project/vllm/pull/62589) |
| `VLLM_QSA_GQA_PREFILL=32,1,2,1` | [vllm-project/vllm#62590](https://github.com/vllm-project/vllm/pull/62590) (same profile, applied automatically on gfx942 when BLOCK_M >= 16) |

Once both are in a released vLLM ROCm image, use that image without the overlay and without `VLLM_QSA_GQA_PREFILL`.
The upstream versions are not byte-identical to this overlay; their own measurements are in `upstream/`.
