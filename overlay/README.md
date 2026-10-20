# overlay/qsa.py

Mount read-only over `vllm/models/qwen4_exp/amd/ops/qsa.py` in `vllm/vllm-openai-rocm:v0.32.0`:

```
-v $PWD/overlay/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro
```

**Only valid for that image.** Stock file sha256 `7a7c8501e0b92a4236e382c6ef494670d684ee1dfeb2d673f942a4f38fccf69b`
(identical to vLLM main at the time), overlay sha256 `6c84b3f942a23b36842caf81d41ce15fe06879d389d2bd473c90131147a767eb`.

Content: the stock file plus

* [vllm-project/vllm#62589](https://github.com/vllm-project/vllm/pull/62589): tiled QSA indexer scoring
  (16 rows × 64 keys per program, `tl.dot`), per-request skip of column blocks past the visible range, fp16 dot
  for fp16 inputs;
* [vllm-project/vllm#62590](https://github.com/vllm-project/vllm/pull/62590): sparse-GQA prefill BLOCK_N=32 on
  gfx942 when BLOCK_M >= 16.

No environment variables. Drop the overlay once both PRs are in the image you run.

Validated as `d02-tp2-combo-upstream` (TP2 + EP, QuickReduce INT6, tuned MoE table), raw data in
`results/experiments/`:

| | stock (e00) | d02 |
|---|---:|---:|
| prefill 8 × 32K | 11,469 tok/s | 24,266 tok/s |
| decode c64 | 2,167 tok/s | 3,208 tok/s |
| agent 32K × 16 sessions, wall | 63.1 s | 31.9 s |

Gate (`gate-tp2-d02.*`): functional 9/9, GSM8K-1319 0.9629, 238K needle 3/3. Through LiteLLM on 2 × TP2:
functional 9/9 (`results/gateway/functional-gateway-tp2x2-upstream-overlay.json`).

Not measured end to end on TP1. Both PRs cover the TP1 shapes (the microbenchmarks in `upstream/` include them).
