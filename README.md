# Qwen3.8-Flash-Next-FP8 on AMD MI325X

Tuning results and drop-in settings for serving Qwen3.8-Flash-Next-FP8 with vLLM v0.32.0 on MI325X (gfx942),
with LiteLLM in front for session affinity. Everything here was measured; raw data is in [`results/`](results/).

## Results

One engine, same perfsuite (fixed 512-token decode, unique prompts per run). "Stock" is vLLM v0.32.0 with the
same flags and no overlay, no tuned tables, no QuickReduce. Run-to-run noise is about ±3-5%.

TP2 "tuned" is run d02: the shipped [`overlay/qsa.py`](overlay/), QuickReduce INT6 and the tuned MoE table.

| | stock | tuned | raw |
|---|---:|---:|---|
| TP2 prefill, 8 × 32K concurrent | 11,469 tok/s | **24,266 tok/s** | [e00](results/experiments/e00-base.json) → [d02](results/experiments/d02-tp2-combo-upstream.json) |
| TP2 32K cold TTFT, single request | 3.48 s | **1.51 s** | same |
| TP2 decode, 64 concurrent | 2,167 tok/s | **3,208 tok/s** | same |
| TP2 agent 32K, 16 sessions × 3 turns, wall | 63.1 s | **31.9 s** | same |
| TP2 agent 96K, 8 sessions × 3 turns, wall | 84.9 s | **41.7 s** | same |
| TP2 gate: GSM8K 1319 questions zero-shot / 238K needle / functional | | 0.9629 / 3/3 / 9/9 | [gate-tp2-d02](results/experiments/gate-tp2-d02.gsm8k1319.json) ([needle](results/experiments/gate-tp2-d02.needle.json), [functional](results/experiments/gate-tp2-d02.functional.json)) |
| TP1 GSM8K 1319, stock vs previous overlay revision ¹ | 0.9621 | 0.9666 | [stock](results/experiments/gate-tp1-stock.gsm8k1319.json), [c02](results/experiments/gate-tp1-c02.gsm8k1319.json) |

¹ e02-e14, d01 and all t*/c* runs used the previous overlay revision; see [History](#history).

**2 × TP2 vs 4 × TP1 on the same 4 GPUs.** TP1 is faster per GPU because TP2 all-reduces over one XGMI link.
TP2 keeps two weight copies instead of four, so it has 2.5× the KV cache.

| | 2 × TP2 (d02) | 4 × TP1 (c02 ¹) | raw |
|---|---:|---:|---|
| KV cache per replica | 9,362,083 tokens | 1,866,181 tokens | [kv-cache-capacity.txt](results/experiments/kv-cache-capacity.txt) |
| KV cache total | **18.7M tokens** | 7.46M tokens | same |
| 262K-token requests that fit in KV (vLLM "maximum concurrency") | 2 × 35.71 | 4 × 7.12 | same |
| agent 32K, 16 sessions on 2 GPUs, wall / TTFT p95 | 31.9 s / 19.7 s | **22.3 s / 13.0 s** (8 per engine) | [d02](results/experiments/d02-tp2-combo-upstream.json), [c02](results/experiments/c02-tp1-combo-gdnfly.json) |
| agent 96K, 8 sessions on 2 GPUs, wall | 41.7 s | **26.6 s** (4 per engine) | same |
| idle 32K cold TTFT | **1.51 s** | 2.11 s | same |

A single TP1 engine with 16 × 128K sessions (2.1M tokens > 1.87M) preempted continuously: cache hit 0, TPOT 257 ms
([tp1-single-replica](results/benchmark-grid/tp1-single-replica-20261019-130321.csv)).

**100 simulated users on 2 × TP2 through LiteLLM**, previous overlay revision ([sim100-tp2x2.json](results/gateway/sim100-tp2x2.json)).
Each user starts with a 20K-120K token code context that only grows (never cleared), 256 output tokens per turn,
~30 s mean think time, 15 minutes:

| requests | errors | warm TTFT p50 / p95 / p99 | cold TTFT p50 / p95 | warm-turn cache hit | mean warm prompt | split a / b |
|---:|---:|---:|---:|---:|---:|---:|
| 2,312 | 0 | **0.58 s** / 6.65 s / 49.8 s | 11.4 s / 31.1 s | **94%** | 92,965 tokens | 1,164 / 1,148 |

**Production grid, 4 × TP1 through LiteLLM**, previous overlay revision, 16 concurrent agent sessions, 3 turns × 256 tokens
([tp1x4-gateway](results/benchmark-grid/tp1x4-gateway-20261019-125941.csv),
[224K](results/benchmark-grid/tp1x4-gateway-224k-20261020-003711.csv)). The 1/4/8-session cells are in the same files.

| context | aggregate tok/s | TTFT p50 | TTFT p95 | TPOT p50 | errors |
|---|---:|---:|---:|---:|---:|
| 8K | 1,270 | 0.36 s | 0.70 s | 6.8 ms | 0 |
| 32K | 880 | 0.58 s | 4.49 s | 6.9 ms | 0 |
| 64K | 585 | 0.85 s | 10.8 s | 6.7 ms | 0 |
| 128K | 340 | 0.76 s | 25.8 s | 7.2 ms | 0 |
| 224K | 190 | 1.08 s | 47.5 s | 7.3 ms | 3 / 48 ¹ |

¹ HTTP 400 from client-side sizing: the benchmark's 4.5 chars/token estimate undershoots on some code, so
prompt + max_tokens exceeded 262,144. TTFT p50 is mostly warm turns, p95 is the cold first turn.

At 16 sessions the 4 replicas give 2.0× (8K) to 3.3× (64K) the throughput of one replica
([single replica](results/benchmark-grid/tp1-single-replica-20261019-130321.csv)).

## Use it

Scope: these settings were validated only for

| | |
|---|---|
| vLLM image | `vllm/vllm-openai-rocm:v0.32.0` @ `sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905` (ROCm 7.2.3, torch 2.13, aiter 0.1.24.post1) |
| GPU | MI325X, gfx942, 304 CUs, 256 GiB HBM |
| Model | `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| Host | Ubuntu 24.04, kernel 6.8.0-124, amdgpu 6.16.13 |

The overlay replaces one file of that exact image. The tuned AITER tables are keyed on this model's MoE shapes and
304 CUs and do nothing elsewhere.

### What to apply

| optimization | how | TP2 | TP1 | measured gain | accuracy check |
|---|---|:-:|:-:|---|---|
| QSA overlay: tiled indexer (vllm#62589) + sparse-GQA BLOCK_N=32 (vllm#62590) | mount [`overlay/qsa.py`](overlay/), no env vars | yes | yes ² | TP2, #62589 alone: 8×32K prefill 11,486 → 20,987 tok/s, decode c64 2,167 → 3,023 tok/s ([pr-stock](results/experiments/pr-stock-tp2.json) → [fin2-tiled](results/experiments/fin2-tiled-tp2.json)); #62590 alone: 11,486 → 12,272 tok/s ([fin-pr3](results/experiments/fin-pr3-tp2.json)) | TP2 combination (d02): GSM8K-1319 0.9629, needle 3/3, functional 9/9 |
| QuickReduce INT6 all-reduce ¹ | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6`, `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0` | yes | n/a | 19,089 → 22,039 tok/s ([e02](results/experiments/e02-qsa.json) → [e11](results/experiments/e11-qsa-qrint6.json)) | GSM8K-1319 0.9629 (d02), 0.9651 (d01) |
| tuned AITER fused-MoE tables ¹ | mount [`tuned/aiter/`](tuned/aiter/), set `AITER_CONFIG_FMOE` | `ep2` table | `tp1` table | kernel −3.7% to −13% at the tuned shapes; TP1 8×32K prefill +0.8% / +0.7% ([t14](results/experiments/t14-tp1-gqa32.json) → [c01](results/experiments/c01-tp1-combo.json), [c01r](results/experiments/c01r-tp1-combo.json)) | GSM8K-200 0.965 / 0.97 |
| GDN FlyDSL prefill ¹ | `--additional-config={"gdn_prefill_backend":"aiter_flydsl"}` | not measured | yes | agent 96K × 4 TTFT p95 19.9 → 17.6 s ([c01](results/experiments/c01-tp1-combo.json) → [c02](results/experiments/c02-tp1-combo-gdnfly.json)); repeat 18.2 → 17.7 s ([c01r](results/experiments/c01r-tp1-combo.json) → [c02r](results/experiments/c02r-tp1-combo-gdnfly.json)) | GSM8K-200 0.975 |

GSM8K-200 has a standard error of about 1.2 points; the full 1319-question runs are the accuracy gate.
¹ measured on top of the previous overlay revision; d02 re-validates the full TP2 combination with the shipped overlay.
² both PRs cover the TP1 shapes in their microbenchmarks; the shipped overlay was not run end to end on TP1.

These are not optimizations, but this model on this image needs them:

| setting | reason |
|---|---|
| `--enable-expert-parallel` (TP2) | pure TP2 splits the 640-wide experts and AITER FP8 MoE rejects the resulting 64×64 scales (vllm#60839) |
| `VLLM_ROCM_USE_AITER_CUSTOM_AR=0` (TP>1) | the int8 PLE all-reduce is not supported by AITER custom AR (vllm#58688) |
| `--compilation-config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | PIECEWISE replay reads stale n-gram offsets on AMD (vllm#61486) |
| `--engram-config={"cpu_offload":false}` | keeps the n-gram table in HBM (AMD default is CPU offload) |
| no `--kv-cache-dtype fp8` | QSA requires a BF16 main KV cache |
| `--tool-call-parser=qwen3_coder --reasoning-parser=qwen3` | otherwise tool calls come back as raw `<tool_call>` text |

Other flags (MTP 3, `--max-num-batched-tokens 32768`, `--max-num-seqs 64`, util 0.93) were swept; see
[Tried and dropped](#tried-and-dropped).

### TP2 (one engine on 2 GPUs)

Run from the repo root. Check your HIP device ids first; on our host amd-smi index and HIP id differ.

```bash
IMAGE=vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905
MODEL_DIR=/path/to/qwen3.8-flash-next-fp8
VLLM_PKG=/usr/local/lib/python3.12/dist-packages

docker run -d --name qwen38-tp2 --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -p 127.0.0.1:8001:8000 \
  -v "$MODEL_DIR":/models/qwen3.8-flash-next-fp8:ro \
  -v "$PWD/overlay/qsa.py":$VLLM_PKG/vllm/models/qwen4_exp/amd/ops/qsa.py:ro \
  -v "$PWD/tuned/aiter":/aiter-tuned:ro \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e AITER_CONFIG_FMOE=$VLLM_PKG/aiter/configs/tuned_fmoe.csv:/aiter-tuned/tuned_fmoe_qwen38_ep2.csv \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6 -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 -e VLLM_ROCM_USE_AITER_CUSTOM_AR=0 \
  -e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0 -e SAFETENSORS_FAST_GPU=1 -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e OMP_NUM_THREADS=8 -e HF_HUB_OFFLINE=1 \
  "$IMAGE" /models/qwen3.8-flash-next-fp8 --served-model-name=qwen3.8-flash-next \
  --tensor-parallel-size=2 --enable-expert-parallel \
  --max-model-len=262144 --max-num-seqs=64 --gpu-memory-utilization=0.93 \
  --max-num-batched-tokens=32768 --enable-chunked-prefill --enable-prefix-caching \
  --enable-prompt-tokens-details --async-scheduling \
  --speculative-config='{"method":"mtp","num_speculative_tokens":3}' \
  --engram-config='{"cpu_offload":false}' \
  --compilation-config='{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3
```

### TP1 (one engine per GPU)

Same as TP2 except:

```bash
  -e HIP_VISIBLE_DEVICES=0 \
  -e AITER_CONFIG_FMOE=$VLLM_PKG/aiter/configs/tuned_fmoe.csv:/aiter-tuned/tuned_fmoe_qwen38_tp1.csv \
  # drop the two QUICK_REDUCE variables
  ... --tensor-parallel-size=1 \
  # drop --enable-expert-parallel, add:
  --additional-config='{"gdn_prefill_backend":"aiter_flydsl"}'
```

A cold start takes about 4-5 minutes (weights from page cache, graph capture, one JIT kernel). At util 0.93 each
GPU needs about 240 GiB free.

### Which layout

- **2 × TP2**: many concurrent long-context sessions (above ~50 at 128K), sessions that stay idle and should keep
  their prefix cache, or lowest single-request TTFT. This is what serves ~100 users here.
- **4 × TP1**: up to ~40 active agent sessions with contexts mostly ≤128K. 1.4-1.7× faster per GPU, lower tail latency.

## Full deploy stack

[`deploy/`](deploy/) runs 2 × TP2 or 4 × TP1 replicas plus an optional LiteLLM gateway (session affinity and
failover), with docker compose.

```bash
cd deploy
cp .env.example .env && chmod 600 .env    # set MODEL_DIR, LITELLM_MASTER_KEY, LAYOUT, GPU map
scripts/download_model.sh                 # pinned revision, resumable
scripts/verify_model.sh                   # sha256 / blob sha1 of all 144 files vs the Hub manifest
scripts/up.sh                             # replicas of LAYOUT in parallel -> wait healthy -> LiteLLM
scripts/status.sh                         # health, prefix-cache hit %, running/waiting, KV %, HBM
scripts/down.sh                           # stops only this stack's containers
```

Replica ports bind to 127.0.0.1 (vLLM has no auth); clients use LiteLLM on :4000. Without the gateway, start only
the replicas: `docker compose --profile tp2x2 up -d` and point clients at them directly.
Client configs for OpenCode and Pi Agent are in `deploy/config/agents/` (replace `LAB_HOST`).

LiteLLM v1.105.1 settings (`deploy/config/litellm.*.yaml`, `deploy/compose.yaml`) and why:

| setting | why | raw |
|---|---|---|
| `optional_pre_call_checks: [session_affinity]` | prefix caches are per replica. With `x-session-id`: 16/16 sessions sticky, later-turn cache hit 0.899, warm TTFT p50 0.525 s. Without: 1/16 sticky, 0.489, 1.601 s | [affinity](results/gateway/affinity-affinity-tp1x4-shuffle.json), [no header](results/gateway/affinity-noheader-tp1x4-shuffle.json) |
| `allowed_fails: 0` | with 1, the in-request retries of a pinned session went back to the stopped replica and the client got a 500 | [failover-tp1x4.json](results/gateway/failover-tp1x4.json) |
| `DISABLE_AIOHTTP_TRANSPORT=True` | with the default aiohttp transport the first request after a replica was stopped took 71.6-72.1 s before failing over (3 runs); with httpx 0.64 s, no client error | [v2-v4](results/gateway/failover-tp1x4-v2.json), [v5-httpx](results/gateway/failover-tp1x4-v5-httpx.json) |
| no per-deployment `timeout`, global `request_timeout: 1800` | a per-deployment timeout replaces the 5 s connect timeout | |
| `routing_strategy: least-busy` (tp1x4) | simple-shuffle placed 16 sessions 2/4/7/3; least-busy with staggered starts 5/4/4/3 and 4/4/4/4. Sessions that start in the same instant all land on one replica | [s3](results/gateway/affinity-tp1x4-leastbusy-stagger-s3.json), [s4](results/gateway/affinity-tp1x4-leastbusy-stagger-s4.json), [s1](results/gateway/affinity-affinity-tp1x4-leastbusy-s1.json) |

Header support: OpenCode 1.19.3 sends `x-opencode-session-id`, recognised as is. Pi Agent 1.3.0 needs
`compat.sessionAffinityFormat: "openrouter"` to send `x-session-id`. Any client can send `x-litellm-session-id`.
LiteLLM pins are claim-once: after failover the session returns to its original replica, cold, once that is back.

Gateway checks: functional suite 9/9 on tp2x2 with the shipped overlay
([file](results/gateway/functional-gateway-tp2x2-upstream-overlay.json)). With the previous overlay: 9/9 on tp1x4
([file](results/gateway/functional-gateway-tp1x4.json)), 8/9 on tp2x2
([file](results/gateway/functional-gateway-tp2x2.json); the short-prompt determinism check returned two variants for
1 of 4 prompts). Real agent tasks (fix 4 bugs until pytest passes): OpenCode 2/2, Pi Agent 2/2, tests untouched,
each session on one replica ([results/gateway/](results/gateway/)).

## Upstream

| PR | change | evidence | replaces here |
|---|---|---|---|
| [vllm-project/vllm#62589](https://github.com/vllm-project/vllm/pull/62589) | tiled QSA indexer scoring, per-request skip of invisible columns. TP2 8×32K prefill 11,486 → 20,987 tok/s (+83%), decode c64 2,167 → 3,023 tok/s (+39%) | [upstream/vllm-62589-qsa-indexer-tiled](upstream/vllm-62589-qsa-indexer-tiled/) | indexer part of `overlay/qsa.py` |
| [vllm-project/vllm#62590](https://github.com/vllm-project/vllm/pull/62590) | sparse GQA prefill BLOCK_N=32 on gfx942. TP2 8×32K prefill 11,486 → 12,272 tok/s (+6.8%) | [upstream/vllm-62590-sparse-gqa-gfx942](upstream/vllm-62590-sparse-gqa-gfx942/) | sparse-GQA part of `overlay/qsa.py` |
| [ROCm/aiter#6692](https://github.com/ROCm/aiter/pull/6692) | fused-MoE tuner uses all sources for shapes FlyDSL v2 cannot tune (with `TUNE_ONLY` unset it found 0 candidates) | [upstream/aiter-6692-moe-tuner-fallback](upstream/aiter-6692-moe-tuner-fallback/) | the `TUNE_ONLY=` workaround when re-tuning |
| [ROCm/aiter#6693](https://github.com/ROCm/aiter/pull/6693) | gfx942 / 304-CU fused-MoE configs for this model | [upstream/aiter-6693-gfx942-moe-configs](upstream/aiter-6693-gfx942-moe-configs/) | `tuned/aiter/` and `AITER_CONFIG_FMOE` |

The PR end-to-end numbers compare each PR's `qsa.py` alone against a stock run
([pr-stock-tp2](results/experiments/pr-stock-tp2.json), [fin2-tiled-tp2](results/experiments/fin2-tiled-tp2.json),
[fin-pr3-tp2](results/experiments/fin-pr3-tp2.json)), with no QuickReduce or tuned tables. `overlay/qsa.py` is the
v0.32.0 file with both PRs applied. When both are in a released vLLM ROCm image, drop the overlay mount.
When an image ships an aiter with #6693, drop `AITER_CONFIG_FMOE`.

## What was tuned and why

Profiling a 32K prefill on stock TP2 ([lab notebook](docs/lab-notebook.md), e01): the QSA indexer kernel took 43%
of GPU time, the TP all-reduce 15%, BF16 GEMM 10%, AITER FP8 MoE 10%, sparse GQA 7.5%.

- **QSA indexer (vllm#62589).** The stock scorer sizes its work to max_model_len: 65,600 columns per query row,
  most of them masked for a 32K prompt, one row per program, no MFMA. The overlay scores 16 rows × 64 keys per
  program with `tl.dot` and skips column blocks past each request's visible range. `qsa_select_paged_tokens`:
  32K prefill 90.7 → 4.58 ms, decode 64 × 4 0.925 → 0.261 ms; selection differs from main only at exact score ties.
- **Sparse-GQA prefill (vllm#62590).** The >512-program launch profile was tuned on GB300. On gfx942,
  BLOCK_N 32 instead of 64 runs the kernel 1.5-2.3× faster for the TP1 and TP2 shapes
  ([sweep](results/experiments/sweep_gqa2.txt)). (64,1,4,1) and (128,1,4,1) fault the GPU.
- **QuickReduce INT6.** After the indexer fix, the TP2 all-reduce (~156 MB per 30K-token chunk per layer over one XGMI
  link) was the largest cost. INT8 gave +13% prefill, INT6 about the same or slightly better, FP gave nothing (link-bound).
  INT6 passed full GSM8K.
- **Tuned AITER fMoE tables.** AITER ships no gfx942/304-CU rows for these shapes. The tuner found gains at 5 of 16
  shapes for E=256 and 3 for E=512; decode-size shapes stayed on the default because the best candidate was slower.
- **GDN FlyDSL prefill (TP1).** Opt-in AITER FlyDSL kernels for the gated delta-net layers. Small gain on long-context
  agent tail latency; not measured on TP2.
- **LiteLLM.** See [Full deploy stack](#full-deploy-stack).

## Tried and dropped

All within noise or worse ([results/experiments/](results/experiments/), details in the [lab notebook](docs/lab-notebook.md)):

| change | result | run |
|---|---|---|
| MTP 2 | throughput within ±3% of MTP 3 | e07, t03 |
| MTP 4 | acceptance 0.66, decode c64 −22% (TP1) | e08, t02 |
| Triton MoE instead of AITER | decode c64 −7%, c16 −8% | t04 |
| `--max-num-batched-tokens` 65536 / 16384 | decode c64 −19% / prefill −8% | t05, t12 |
| `--max-num-seqs 128` | no change | t06 |
| `-O3`, `--performance-mode throughput` | no change | t08, t09 |
| `--long-prefill-token-threshold 8192` | worse 96K p95 | t13 |
| AITER linear / Triton GEMM off | no change (aiter tgemm is gfx950-only) | t10, t11 |
| QuickReduce FP | same as no QuickReduce | e04 |
| QuickReduce INT4 | +6-7% prefill over INT8, GSM8K-200 0.965; not taken (highest numeric risk) | e12 |
| vllm#62381 GDN overlay | no effect on this model | t19 |
| offline TunableOp, 75 hot BF16 GEMMs | decode c64 +2-3% only | t20, t21, t21r |

Online TunableOp was stopped: about 1 minute per shape across hundreds of shapes.

## Verify or re-tune

- [`tools/`](tools/): experiment harness (`exp.sh`, `run.sh`, `perfsuite.py`, `compare.py`, `gate.sh`,
  `queue.sh`, `profile.py`), tests and the agent benchmark. See [tools/README.md](tools/README.md).
- [`docs/lab-notebook.md`](docs/lab-notebook.md): the full log of experiments and decisions.
- [`results/README.md`](results/README.md): what each raw file is.

Reproduce a table from the shipped data:

```bash
python3 tools/experiments/compare.py e00-base d01-tp2-combo d02-tp2-combo-upstream
```

## History

The first overlay revision (not shipped) had its own indexer kernel and a `VLLM_QSA_GQA_PREFILL` env var for the
sparse-GQA launch profile; the lab notebook describes it. Most experiments ran on it, including the TP2 final d01:
8×32K prefill 24,615 tok/s, decode c64 3,203 tok/s, agent 32K wall 32.1 s, GSM8K-1319 0.9651
([d01](results/experiments/d01-tp2-combo.json), [gate](results/experiments/gate-tp2-d01.gsm8k1319.json)).
d02 with the shipped overlay is within 4% of d01 on prefill and decode throughput and agent wall time; its
single-request 32K cold TTFT is 1.51 s vs 1.33 s.
The TP1 runs, the 4 × TP1 grid, sim100 and the gateway tests except
[functional-gateway-tp2x2-upstream-overlay.json](results/gateway/functional-gateway-tp2x2-upstream-overlay.json)
(9/9) used the first revision.

## License

Apache-2.0 for this repository. `overlay/qsa.py` and the `upstream/*/qsa.py` files are modified vLLM source
(Apache-2.0). The model weights are under the Qwen Community License and are not included.
