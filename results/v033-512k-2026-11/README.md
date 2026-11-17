# 512K context and vLLM v0.33.0, 2026-11-16

Two questions, MI325X, TP2 + EP2 on 2 GPUs per engine (HIP 4+5 or HIP 6+7, each container saw only its own render nodes):

1. Can the TP2 deployment (MTP 3, 32768 batched tokens) serve more than 262K tokens, and what does it cost?
2. Does vLLM v0.33.0 pass the full accuracy gate that the [overlay-refresh](../overlay-refresh-2026-11/) runs left open?

Answers: 512K (YaRN factor 2 in `config.json`) passes every check and costs 4.5% cold TTFT (sim50 p50); 1M works but costs 28%.
v0.33.0 (ROCm 10.0 build) passes GSM8K-1319 at 0.9644 and is faster on decode. Both are now the `deploy/` defaults for tp2x2.

## Setup

Flags = `deploy/compose.yaml` tp2x2: expert parallel, MTP 3, AITER FP8 MoE, QuickReduce INT6, tuned fMoE table (EP2),
`FULL_DECODE_ONLY`, `--max-num-seqs 64`, `--max-num-batched-tokens 32768`, `--gpu-memory-utilization 0.93`. Model revision `236dfdf2…`.

| arm | image | overlay | context | GPUs |
|---|---|---|---|---|
| `b` / `b2` | v0.32.0 | d02 `qsa.py` ([history](../../overlay/history/qsa.2026-10-20-d02.py)) | 262,144 | 6+7 / 4+5 |
| `512` | v0.32.0 | d02 | 524,288, YaRN factor 2 | 6+7 |
| `y` | v0.32.0 | d02 | 1,000,000, YaRN factor 4 | 4+5 |
| `qr` | v0.32.0 | d02 | 524,288 + QuickReduce max 256 MB | 4+5 (gates), 6+7 (sim) |
| `dep` | v0.32.0 | d02 | as `qr`, `deploy/` model-dir layout | 4+5 |
| `r10` | v0.33.0 (ROCm 10.0) `sha256:4fcd39b7…` | `overlay/` (2026-11-03) | as `dep` | 4+5 |
| `r72` | v0.33.0-rocm72 `sha256:4fe7bc12…` | `overlay/` | as `dep` | 6+7 |
| `b32` | v0.32.0 | d02 | as `dep` (decode check only) | 4+5 |

YaRN comes from a copy of the model directory whose `config.json` sets `rope_type: yarn`, `factor`, `original_max_position_embeddings: 262144`
(every other file symlinked; [`deploy/scripts/make_model_yarn512.sh`](../../deploy/scripts/make_model_yarn512.sh)), plus `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
Not `--hf-overrides`: see "MTP and `--hf-overrides`" below.

## Results

| | 262K (`b`) | 512K (`512`) | 1M (`y`) | 512K + QR 256 (`qr`) | v0.33.0 ROCm 10 (`r10`) | v0.33.0-rocm72 (`r72`) |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K-1319 | 0.9568 | 0.9644 | 0.9621 | 0.9674 | **0.9644** | 0.9613 |
| functional | 9/9 | 8/9 ¹ | 9/9 | 9/9 | 8/9 ¹ | 8/9 ² |
| KV pool (tokens) ³ | 9,363,500 | 9,913,399 | 10,184,210 | 9,913,399 | 9,892,368 | 9,952,458 |
| 500K retrieval (4 facts), TTFT | n/a | 4/4, 24.9 s | 4/4, 27.0 s | 4/4, 24.3 s | 4/4, **22.8 s** | 4/4, 23.0 s |
| TPOT at 500K, MTP past 262,144 | n/a | 4.28 ms | 3.89 ms | 4.17 ms | **3.19 ms** | 3.83 ms |
| sim50: requests in 15 min | 1,230 | 1,218 | 1,175 | 1,222 | **1,254** | 1,241 |
| sim50: cold TTFT p50 | 14.2 s | 14.8 s | 18.2 s | 14.6 s | **14.0 s** | 14.4 s |
| sim50: warm TTFT p50 / p95 / p99 | 0.53 / 0.90 / 4.95 s | 0.56 / 1.00 / 6.50 s | 0.61 / 1.06 / 16.0 s | 0.55 / 1.01 / 5.71 s | **0.49 / 0.84 / 4.15 s** | 0.49 / 0.82 / 3.69 s |
| sim50: errors | 0 | 0 | 0 | 0 | 0 | 0 |
| GPU faults | not recorded | 0 | 0 | 0 | 0 | 0 |

Also: 1M at 985K (`y`): 4/4, TTFT 58.9 s, TPOT 3.81 ms. `b2` (262K on the other pair): sim50 1,230 requests, cold p50 14.2 s, warm p50 0.56 s,
so the 1M slowdown comes from the 1M configuration (max-model-len and YaRN factor 4), not the GPU pair. `dep` (deploy layout): functional 9/9, 300K retrieval 4/4.

Short-prompt decode (`scripts/dec.py`, 512 tokens, ignore_eos, T=0), per-stream tok/s:

| | v0.32.0 + d02 (`b32`) | v0.33.0 ROCm 10 (`r10`) | v0.33.0-rocm72 (`r72`) |
|---|---:|---:|---:|
| 1 stream | 166 | **241** | 192 |
| 16 streams (aggregate) | 107 (1,317) | **130 (1,922)** | 116 (1,716) |

¹ `tool_result_loop`. ² `short_prompt_determinism`. Both depend on sampling or near-ties and fail on every arm at some rate, see Caveats.
³ `b` and `y`: `data/kv-cache-capacity.txt` (copied from `docker logs`); the others from `logs/run*.log`.

## MTP and `--hf-overrides`

With MTP on, YaRN passed as a dict `--hf-overrides` reaches the target model but not the MTP draft
(`vllm/config/speculative.py`, `compose_draft_hf_overrides`: dict overrides are target-only). The draft keeps `rope_type: default` and
a 262,144-row RoPE cache, and the engine dies with `Memory access fault by GPU` once a decode position passes 262,144.
Bisect and the fix check (cold 262K prompt + 1500 tokens: dict override faults, `config.json` passes) are in
[`../long-context-1m-2026-11/tp2/FINDINGS.md`](../long-context-1m-2026-11/tp2/FINDINGS.md). Without MTP the dict override works (TP4 run there).

## Caveats

- One boot per arm. Run-to-run noise on this suite is a few percent; GSM8K-1319 has a standard error of about 0.55 points.
- `b32` vs `r10` changes two things: the vLLM image and the overlay revision (d02 vs 2026-11-03). The newer overlay adds a per-row decode
  kernel that is inactive with MTP 3 ([overlay/README.md](../../overlay/README.md)), so the decode gain is most likely the image, but it was not isolated.
- `b32` 16-stream TTFT was 1.21 s (0.20 s on the others), so its aggregate may be understated; per-stream rate is the fairer number.
- `tool_result_loop` samples at the model default (T=1.0). Idle A/B, 60 trials per arm and pair: 262K 45/60 + 50/60, 512K 44/60 + 49/60,
  1M 44/60 (`logs/ab-*.log`). An earlier run under sim50 load (`logs/trl-*.log`) showed 29/40 = 73% (512K) vs 37/40 = 93% (262K); the idle A/B did not reproduce it.
- `short_prompt_determinism` flips one word ("opposites" / "antonyms") in "Opposite of hot?" at about token 25: 3/20 runs on 262K,
  2/20 on 512K (`scripts/det3.py`, idle servers, `data/determinism-ab.txt`). A near-tie, not the #61486 failure (which produces garbage); present before this change.
- sim50 (`tools/tests/sim_users.py --users 50 --minutes 15 --seed 7`) talks to one replica directly, no gateway; same seed, same requests on every arm.
- QuickReduce max 256 MB did not change the KV pool (9,913,399 both), so the buffer is apparently allocated outside the `gpu-memory-utilization` budget.
  The largest TP all-reduce here is 32768 × 2560 × 2 B = 160 MB.
- The v0.33.0 image has no `render` group in `/etc/group`; `group_add: [video, render]` makes the container fail to start
  (`unable to find group render`). `deploy/compose.yaml` now takes numeric GIDs.

## Files

`data/`: `gsm8k-<arm>.json`, `functional-<arm>.json`, `sim50-<arm>.json` (summary + per-request rows), `<arm>_<size>.json` (retrieval:
`Y_*` = `y`, `P512_*` = `512`, `QR_*` = `qr`, `DEP_*` = `dep`). `logs/`: console output of each phase (`run*.log`, KV pool lines,
decode results in `run33.log`). `scripts/`: launch (`up.sh`: `YARN=0|1|2`, `OVL33=1` for `overlay/`, `QRMAX`, `DEPLOY=1`, `IMG`),
phase runners, probes. Run from a work dir with `R=<repo root>` and `ORIG_DIR=<model dir>` set. `up.sh` expects config.json copies in the work dir:
`LINK_ROOT=$ORIG_DIR ../../deploy/scripts/make_model_yarn512.sh $ORIG_DIR $W/model-yarn512` (`FACTOR=4` and `model-yarn1m` for 1M;
`DEPLOY=1` uses `$W/deploy/model-yarn512`, built with the default `/models-orig` links); the retrieval stage is
[`../long-context-1m-2026-11/scripts/stage.py`](../long-context-1m-2026-11/scripts/stage.py).
