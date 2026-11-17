# 1M context on 4x MI325X (TP4 + EP4), 2026-11-03

Question: can Qwen3.8-Flash-Next-FP8 serve a 1M-token context on four MI325X, and is it correct?
Answer for this setup: **yes, up to 985K input tokens, in 5 of 5 runs, with correct retrieval and no kernel errors.** Details and limits below.

## Setup

| item | value |
|---|---|
| GPUs | HIP 4-7 = renderD160/168/176/184 = amd-smi GPU 4-7 (PCI f7/87/e7/97 on the test host; the mapping is not shown in this run's logs and was not re-verified); only these four render nodes plus `/dev/kfd` mounted |
| Image | `vllm/vllm-openai-rocm:v0.32.0`, `sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905` (vLLM 0.32.0) |
| Overlay | d02 `qsa.py` ([`overlay/history/qsa.2026-10-20-d02.py`](../../overlay/history/qsa.2026-10-20-d02.py): #62589 earlier head + #62590, sha256 prefix `6c84b3f942a23b36`); no `indexer_qsa.py` change |
| Layout | TP4 + EP4, one replica, `--max-model-len 1000000`, `--max-num-seqs 1`, `--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.90`, BF16 KV, no MTP, prefix caching on, `FULL_DECODE_ONLY` |
| RoPE | `--hf-overrides` `text_config.rope_parameters` (passed as an override; the log shows it in the non-default args and `Using max model len 1000000`, but the resolved engine config does not echo the rope parameters, so YaRN being active is inferred from no NaN and correct retrieval, not shown directly) = yarn, factor 4.0, original_max_position_embeddings 262144, rope_theta 1e7, partial_rotary_factor 0.25, mrope_interleaved, mrope_section [11,11,10]; `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| Not used | our TP1/TP2 `AITER_CONFIG_FMOE` tables (the image's stock aiter tuned CSVs still load), QuickReduce, AITER custom all-reduce (off, as in `deploy/compose.yaml` for TP>1) |
| Also set | `--engram-config {"cpu_offload":false}`, `--async-scheduling`, `--reasoning-parser qwen3`, AITER FP8 MoE (`VLLM_ROCM_USE_AITER_MOE=1`) as in `deploy/compose.yaml` |
| Launch | `scripts/up.sh` (full command). Startup about 4 min (container started 09:52:55, `/health` OK at 09:56:56) |

The vLLM recipe page gives no literal 1M command or YaRN JSON; the override above is the one reported in a community post (RTX PRO 6000 FP8, same keys) and matches the model's own `rope_parameters` plus the YaRN keys. It is not an official recipe value.

## Results (single request each, no concurrency)

Prompt tokens are the server's `usage.prompt_tokens`. Prefill tok/s = prompt tokens / TTFT of a cold request. Decode = a second request on the same prompt (prefix-cached, `ignore_eos`, 256 tokens) measured on the stream.

| stage | prompt tokens | TTFT (s) | prefill tok/s | TPOT (ms) | decode tok/s | retrieval (3 passcodes at 10/50/90% + 1 log line at 70%) |
|---|---:|---:|---:|---:|---:|---|
| A | 262,042 | 12.0 | 21,886 | 7.53 | 132.8 | 4/4 |
| B | 523,464 | 25.9 | 20,234 | 7.65 | 130.7 | 4/4 |
| C | 786,454 | 41.3 | 19,063 | 7.65 | 130.7 | 4/4 |
| D | 984,294 | 54.5 | 18,049 | 7.66 | 130.6 | 4/4 |
| D2 (other seed) | 985,165 | 54.4 | 18,100 | 7.65 | 130.7 | 4/4 |

* Prefill throughput falls 18% from 262K to 985K (21.9K -> 18.0K tok/s), so TTFT grows slightly faster than linearly (12.0 s -> 54.5 s for 3.75x the tokens). TPOT is flat (7.5 -> 7.7 ms).
* An 8K smoke run before stage A: 4/4, TPOT 7.4 ms.
* Raw JSON per stage: `data/*.json`. Logs: `logs/startup.log`, `logs/server-full.log`.

## Memory

* Per GPU: weights 44.64 GiB; "Available KV cache memory" 182.88 GiB = 14,927,626 tokens (14.93 concurrent 1M requests) at 0.90 utilization.
* `rocm-smi` used VRAM per GPU was 233.7 GiB (233.6 on GPU 5) from the first stage to the last. The script stores only the per-stage maximum, and the maxima are equal (233.55-233.74 GiB); vLLM preallocates the KV pool, so **the peak says nothing about the minimum**.
* Derived need for one 1M request: KV 182.88 GiB x 1.0M / 14.93M = about 12.3 GiB per GPU; 233.7 - 44.6 - 182.9 = 6.2 GiB of everything else; the log attributes only 1.49 GiB peak activation and 0.04 GiB CUDA graphs to it, the remaining ~4.7 GiB is unattributed. Linear KV scaling is assumed. So roughly 63 GiB per GPU at minimum, 4 x ~63 GiB for this layout. This is arithmetic from the startup log, not a measured minimum run.
* HBM sampling every 3 s via `rocm-smi`; The server log's `kv_cache_usage_perc` reached 6.6% during D2 (6.6% x 14.93M = about 985K tokens), which agrees with the KV arithmetic.

## Errors

None: 0 matches for `Memory access`, `HSA_STATUS`, `Traceback`, `hipError`, `Out of resources`, `EngineDead` in the full server log; container restarts 0.

## Limits (what this does not show)

* Retrieval is four exact-string facts in synthetic log text, one cold prompt per stage and two at 985K. It is not a reasoning or long-document QA benchmark, and no short-context quality check was run with the YaRN override (static YaRN can change short-context quality; the recipe says to check).
* One request at a time (`--max-num-seqs 1`). No concurrency, no MTP, no mixed load, no sustained-load or soak test. "Stable for 1M" is supported for single requests only.
* QSA kernel time at 1M was not profiled, and no kernel was modified. Nothing failed that would point at one. The 18% prefill drop is consistent with a growing indexer/attention cost but this run does not attribute it.
* The decode request reused the cached prompt (second-request TTFT 0.5-1.5 s; server cumulative hit rate about 50% after each stage; `cached_tokens` was not returned in the usage). Request 1 was cold (2.8% hit rate in the log). Prefix caching was on for all stages.
* The retrieval check is a substring match: strict for the `word-NNNN-word` passcodes, loose for PRESSURE (an extra digit would still pass). The needles are easy lexical targets (`NOTE:` lines, query repeats their wording); depth is by line index, not token index.
* `containers restarts 0` comes from `docker inspect` at the time; `docker inspect` output is not saved here.
* Triton JIT warnings for `_qsa_*` kernels appear during the 8K smoke run, before stage A.
* Max tested input 985K; 1,000,000 minus output was not pushed to the limit.
