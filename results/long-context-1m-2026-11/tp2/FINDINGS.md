# TP2 + EP2 + MTP 3 at max-model-len 1M (HIP 4+5), 2026-11-04: stopped after a GPU memory fault

Config (`up.sh`): v0.32.0 image, d02 `qsa.py` overlay, TP2+EP2, MTP 3, `--max-num-batched-tokens 32768`, `--max-num-seqs 64`, gpu-memory-utilization 0.90, YaRN factor 4 override, QuickReduce INT6, tuned EP2 MoE table, prefix caching on.

* Startup OK (about 5 min): weights 91.32 GiB per GPU, KV 130.96 GiB per GPU = 9,617,647 tokens (9.62x for 1M requests).
* 8K smoke: 4/4 correct, decode TPOT 8.4 ms, completed (request 1 and request 2).
* 256K request 1 (cold): 262,042 tokens, TTFT 13.3 s (19,671 tok/s), 4/4 correct.
* 256K request 2 (same prompt, prefix cache hit, ignore_eos, 256 tokens, MTP on): at 01:03:03 `Memory access fault by GPU node-2` and `node-3` (both GPUs), EngineCore dead, container exited (0). The client got no usage for request 2.
* Stopped there. No retry, no GPU reset. Afterwards HIP 4-7 idle (0% activity, 285 MB used).
* Not known: which of MTP 3, the 32768 chunk, or the prefix-cache hit on a 262K prompt triggers it. The TP4 run (no MTP, 8192 chunk) did the same two-request sequence at 256K to 985K without a fault.
* Logs: `logs/startup.log`, `logs/server-full.log` (fault at line 411).

## Root cause (2026-11-04)

The fault needs MTP and a decode position past 262,144. The YaRN override was passed as a dict `--hf-overrides`; vLLM applies dict overrides to the target model only, not to the MTP draft (`vllm/config/speculative.py`, `compose_draft_hf_overrides`: "Dict overrides are target-specific key patches and are not applied to the draft"). The draft keeps `rope_type: default` and `max_position_embeddings: 262144`.

Evidence (`rootcause/`, `bisect/`, same TP2+EP2+MTP 3 server config, all at 256K-class prompts):

| run | change | result |
|---|---|---|
| B0 | same as the first failing run | fault again (2 GPUs) |
| B1 | MTP off | no fault, 256K two-request sequence completes |
| B2 | `--max-num-batched-tokens 8192` | fault |
| R1 | dict override; cold 200K prompt + 1500 tokens (201,129 positions) | OK |
| R1 | dict override; cold 262K prompt + 1500 tokens (263,270 positions) | fault |
| R2 | YaRN written into a copy of `config.json` (no `--hf-overrides`; draft reads it); same crossing request (263,270 positions) | OK, TPOT 4.4 ms |

B3 (prefix caching off) was not possible: the model requires `--enable-prefix-caching` (`--mamba-block-size can only be set with --enable-prefix-caching`). It is not needed: R1 reproduces the fault with a cold prompt.

Mechanism, inferred and not observed: the base RoPE builds `cos_sin_cache = arange(max_position_embeddings)`, so the draft's cache has 262,144 rows and the draft's MTP step at a position >= 262,144 reads past it. The target is not affected because YaRN builds its cache from `original_max_position_embeddings` and the factor. The first request of the failing run decoded only about 40 tokens (EOS) at 262,042 positions, so it stayed below the boundary; the second used `ignore_eos` with 256 tokens and crossed it.

Fix for deployment: give YaRN through `config.json` (a model directory with its own `config.json` and every other file symlinked to the original), not through dict `--hf-overrides`, whenever MTP is on. Without MTP the dict override works (TP4 run). Upstream: the dict-override behaviour for the draft is documented in code but nothing warns at startup.
