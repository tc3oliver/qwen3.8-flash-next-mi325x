# Raw results

All runs: 2026-10-19/20, MI325X, `vllm/vllm-openai-rocm:v0.32.0` (digest in the top-level README), model revision
`236dfdf2…`. Run ids (e00, d01, c02, ...) match `docs/lab-notebook.md`.

## experiments/

One engine per run, started by `tools/experiments/exp.sh` and measured by `tools/experiments/run.sh`.
e = TP2 (+EP) on 2 GPUs, t/c = TP1 on 1 GPU, d = TP2 combination. All e02-e14, t*, c* and d01 runs used the first
overlay revision (own indexer kernel, `VLLM_QSA_GQA_PREFILL` env var), which is not shipped; d02 uses `overlay/qsa.py`.

| file | content |
|---|---|
| `<run>.json` | perfsuite: `prefill_1` (one cold 32K prompt), `prefill_8` (8 × 32K cold), `decode_c1/c16/c64` (1K in, 512 out, ignore_eos), `agent_32k_c16`, `agent_96k_c8`, `agent_32k_c8`, `agent_96k_c4` (3 turns, 256 out per turn, prefix reuse), MTP acceptance, `gsm8k200` |
| `<run>.gsm8k.json` | GSM8K first 200 questions, zero-shot, thinking off, per-question rows |
| `<run>.config.txt` | exact vLLM flags, env and mounts of the run. Mount paths are relative to the original tree (see the lab notebook header) |
| `<run>.run.log` | console output of the run |
| `gate-<run>.functional.json` | functional suite (9 checks: health, models, chat, streaming, tool calls, reasoning, multi-turn, #61486 short-prompt determinism) |
| `gate-<run>.gsm8k1319.json` | full GSM8K, 1319 questions, zero-shot |
| `gate-<run>.needle.json` | 238K-token needle at 3 positions |
| `kv-cache-capacity.txt` | "GPU KV cache size" lines from the server logs (logs not included) |
| `invalid/` | first d02 attempt, kept as a failed run: the overlay was mounted at a wrong path (zsh `:r` modifier), so it ran stock `qsa.py`. Not valid; d02 was re-run |
| `sweep_gqa2.txt` | sparse-GQA prefill launch-profile sweep, each config in its own process |

Key runs:

| run | config |
|---|---|
| `e00-base` | TP2 stock settings (no overlay, untuned AITER, no QuickReduce) |
| `e02-qsa`, `e02b-qsa` | + QSA overlay (e02b = same on the other GPU pair, noise reference) |
| `e03-qsa-qrint8`, `e11-qsa-qrint6` | + QuickReduce INT8 / INT6 |
| `e14-qsa-qrint8-gqa32` | + sparse-GQA profile 32,1,2,1 |
| `d02-tp2-combo-upstream` | TP2 final with the shipped overlay (vllm#62589 + #62590) + QuickReduce INT6 + tuned MoE (ep2) |
| `d01-tp2-combo` | same with the first overlay revision + GQA profile env |
| `t01-tp1`, `e06-tp1-qsa` | TP1 with overlay |
| `t14-tp1-gqa32`, `c01-tp1-combo`, `c02-tp1-combo-gdnfly` (+ `c01r`, `c02r` repeats) | TP1: + GQA profile, + tuned MoE (tp1), + GDN FlyDSL = TP1 final |
| `gate-tp2-d02`, `gate-tp1-stock`, `gate-tp1-c02`, `gate-tp2-d01` | accuracy gates |
| `pr-stock-tp2`, `fin2-tiled-tp2`, `fin-pr3-tp2` | stock vs vllm#62589 (final) vs vllm#62590 `qsa.py`, used in the PR texts. `fin-tiled-tp2` = first revision of #62589 |
| `br-pr*-tp2`, `pr-v*-tp2` | earlier PR-branch candidates (the `qsa.py` mounted is in each `.config.txt`; those files are not included) |
| `e01`, `e05`, `t15`, `t16`, `t18a` | profiling / TunableOp tuning runs: config only, profiles not included |
| other `t*` / `e*` | single-change experiments listed under "Tried and dropped" |

Compare any runs: `python3 tools/experiments/compare.py e00-base d01-tp2-combo`.

## benchmark-grid/

`tools/benchmarks/agent_bench.py` through the gateway or directly, real-code agent workload: 3 turns per session,
256 output tokens per turn (fixed), 400-token tool result per turn, thinking off, session starts staggered 0.25 s.
`.csv` = one row per (context, concurrency) cell; `.json` = the same plus per-request rows.

| file | setup |
|---|---|
| `tp1x4-gateway-20261019-125941.*` | 4 × TP1 through LiteLLM, 8K-128K × 1/4/8/16 sessions. The 245760 rows are a failed attempt (client-side context overflow, HTTP 400) |
| `tp1x4-gateway-224k-20261020-003711.*` | same at 229,376 tokens |
| `tp1-single-replica-20261019-130321.*` | one TP1 replica direct, 8K-128K × 1/4/8/16 |
| `single-a-untuned-mtp3-20261019-091207.*` | first deployment (TP2, no overlay, untuned), one replica direct |

## gateway/

Through LiteLLM v1.105.1.

| file | content |
|---|---|
| `sim100-tp2x2.json` | 100 simulated users, 15 min, 2 × TP2, first overlay revision (`tools/tests/sim_users.py`). `summary` + one row per request |
| `affinity-*.json` | 16 sessions × 4 turns, 24K prefix, with/without `x-session-id`, shuffle vs least-busy, staggered or not (`tools/tests/affinity_cache.py`) |
| `failover-tp1x4*.json` | pinned replica stopped mid-session (`tools/tests/failover.py`). v1: `allowed_fails: 1`; v2-v4: `allowed_fails: 0`, aiohttp; v5: httpx transport |
| `session-id-affinity-tp2x2.json` | pinning per session-id format, 8 sessions × 4 requests each; `.` and `:` break it (`tools/tests/session_id_probe.py`) |
| `functional-gateway-*.json` | functional suite through the gateway. `tp2x2-upstream-overlay` = shipped overlay, the others the first revision |
| `opencode-run*`, `pi-run*` | real agent task (`tools/tests/agent_real.sh`): summary `.json`, pytest output, agent transcript `.jsonl` (paths anonymized) |

## first-deploy/

First deployment (TP2, no overlay, untuned AITER), one replica: functional 9/9, GSM8K-500 0.962, needle 3/3 at
~168K (`needle-220k-a`) and ~238K (`needle-240k-a`) prompt tokens,
prefill bursts of 32 × 8K and 16 × 60K, all 200 and the server healthy afterwards.
