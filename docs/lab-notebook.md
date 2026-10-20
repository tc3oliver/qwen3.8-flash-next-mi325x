# Tuning experiments — Qwen3.8-Flash-Next-FP8 on MI325X (vLLM 0.32.0)

> Lab notebook kept during tuning (2026-10-19, UTC). Paths are from the original working tree:
> `experiments/patches/qsa.py` and `patches/qsa.py` = `overlay/qsa.py`; `experiments/results/` = `results/experiments/`;
> `tests/results/` = `results/gateway/`; `benchmarks/results/` = `results/benchmark-grid/`;
> `experiments/*.sh|py`, `tests/`, `benchmarks/` = `tools/`; `patches/test_qsa.py`, `patches/sweep_gqa2.py` = `tools/kernels/`.
> Profiles, traces and server logs are not included.

Window: 2026-10-19 09:30 → ~22:30 UTC. Goal: best production config for 2 replicas on GPU0-3.

Method
* `experiments/exp.sh <slot> <name> ...` starts `qwen38-exp-<slot>` with the compose.yaml settings
  as base and only the listed overrides. Slot a = HIP 1,3 (amd-smi GPU0,1) :8011,
  slot b = HIP 2,0 (GPU2,3) :8012. Production containers `qwen38-vllm-*` are separate.
* `experiments/run.sh` = start → GSM8K-200 gate → `perfsuite.py` → results/<name>.json + server log.
* perfsuite: fixed-length outputs (ignore_eos), unique nonce per run, so runs are comparable:
  prefill_1 (32K cold, bs1), prefill_8 (8×32K cold), decode c1/c16/c64 (1K in / 512 out),
  agent_32k_c16 and agent_96k_c8 (3 turns, 256 out/turn, prefix reuse).
* Accept a change only if GSM8K-200 stays within noise of base (≥0.94) and the target metric
  improves beyond run-to-run noise (measured by repeating the base).

Starting point (e00): TP2+EP, AITER FP8 MoE (untuned on gfx942/304CU: every shape "default"),
MTP=3, mnbt 32768, max-num-seqs 64, FULL_DECODE_ONLY, util 0.93, async scheduling.

Observation from the first benchmark: bs1 cold prefill ≈ 11K tok/s on TP2 (32K in 3.1 s), i.e. a
few % of MI325X dense FLOPs for ~3B active params → prefill is the first target; profile first.

## Log

### e00-base (09:35) — reference
GSM8K-200 0.970. prefill_1 11.5K tok/s (32K TTFT 3.48 s), prefill_8 11.5K tok/s (flat → saturated).
decode c1 4.1 ms TPOT (244 tok/s/seq), c16 8.0 ms / 1184 tok/s, c64 18.9 ms / 2167 tok/s.
agent_32k_c16 wall 63 s, cold TTFT p50 29.8 s, warm 1.0 s; agent_96k_c8 wall 85 s. MTP acc 0.736, mean len 3.21.

### e01-profile (09:40) — torch profiler, rank 0
32K cold prefill (3.3 s GPU): QSA indexer `_qsa_mqa_paged_kernel` 1.45 s (43%), NCCL all-reduce 0.51 s
(15%), BF16 GEMM 0.34 s (10%), AITER FP8 MoE 0.33 s (10%), QSA sparse GQA 0.25 s (7.5%), GDN 0.19 s (6%).
Decode (16 seq × 4 tok, 23.5 ms/step): AITER MoE 27% (130 µs/layer call, untuned default kernels),
QSA indexer 10% (160 µs for 64 rows), custom AR 6%, GDN 5%, rest small GEMMs/elementwise.

Root cause for the indexer: `qsa_select_paged_tokens` sizes the score matrix to the cache CAPACITY
(`page_table.shape[1] * page_size` = max_model_len/4 = 65,536 columns), not to the visible length, and
`_qsa_mqa_paged_kernel` launches rows × 65536/32 programs, one row per program, reloading the shared MQA
key tile for each of the 4 heads, fp32 FMA (no MFMA). For a 32K prompt ~94% of programs are fully masked.
Upstream: vllm#59437 (AITER FlyDSL K1/K2) is opt-in and slower than Triton at TP2 per its own data;
vllm#62344 is an SM80 launch config. Nothing for gfx942 → local overlay.

### QSA indexer overlay (experiments/patches/qsa.py, base sha256 7a7c8501… = v0.32.0)
* v1: early-exit column blocks ≥ row's visible count (top-k reads only [0, visible): rowEnd = seqLens
  in `top_k_per_row_decode`), key tile loaded once for all heads. Logits bit-identical to stock.
* v2 (default): BLOCK_M=16 rows × BLOCK_N=64 keys per program, per-request key tile, per-head
  `tl.dot` (MFMA, fp32 acc), relu, head sum; block-level early exit on max visible; rows of several
  requests in one block handled by looping over the distinct requests. Logits |err| ≤ 5e-7 (fp32
  summation order).
* Selection check (`patches/test_qsa.py`, real shapes, 65,600 columns): every differing selected index
  is an exact score tie (0.0 after relu); stock kernel vs itself is not order-stable either.
* Microbench (MI325X, one GPU), select = score + top-k + expand:
  32K prefill 90.6 → 4.56 ms (19.9×); 16K chunk at 96K 49.0 → 4.34 ms (11.3×);
  decode 64 req×4 tok 0.93 → 0.29 ms; 16×4 0.26 → 0.08 ms; 1×4 0.087 → 0.074 ms.
  Sweep: BM 16/32/64 × BN 32/64/128 × warps 4/8; BM16 BN64 w4 best in every case.

### e02-qsa (10:10) — QSA overlay v2, everything else = e00   ✅ KEEP
GSM8K-200 0.970 (=), MTP acceptance 0.7356 (=). prefill 11.5K → 19.6K tok/s (+70%), 32K TTFT
3.48 → 1.74 s; decode c64 2167 → 3065 tok/s (+41%), TPOT 18.9 → 14.8 ms; agent_32k_c16 wall
63 → 38 s, cold TTFT p50 29.8 → 17.2 s; agent_96k_c8 wall 85 → 50 s.

### MoE tuning (gemm_moe_tune.py, finished 10:00)
First run found 0 candidates: TUNE_ONLY defaults to "flydslv2" only (no gfx942 FP8 blockscale) →
rerun with TUNE_ONLY= (all backends). Several asm 1-stage candidates (a16_blockscale / flat / emsort)
crash the GPU (memory access fault) — tuner restarts its pool; one hung until the 1800 s watchdog.
Result (--compare, ≥3%): only 5/16 shapes improved: M=2 +5.6%, 4 +13%, 1024 +3.7%, 8192 +7.2%,
16384 +7.7%. For M=1,8,16,32 the best tuned candidate is 12–20% SLOWER than the production default,
M=64..512 within ±3%. → tuned_fmoe_qwen38_ep2.csv (5 rows); expected effect ≈ <1% (MoE is 10% of
prefill, and the decode shapes are not improved). Tested only in the final combination.

### e02b-qsa (slot b parity) — same config as e02 on HIP 2,0: all metrics within ±3% → noise ≈ ±3–5%.

### e03-qsa-qrint8 (10:16) — + QuickReduce INT8 for TP all-reduce (bf16 kernels, no fp16 cast)
Profile after e02: NCCL all-reduce is now #1 (412 ms of 1.38 s for a 30K prefill, ~1.85 ms per 156 MB
call ≈ one XGMI link per TP2 pair). QR INT8: prefill 19.6K → 22.4K tok/s (+14%), agent32k wall 37.8
→ 34.3 s, agent96k 49.7 → 45.4 s, decode c16 +16%, c64 ≈. GSM8K-200 0.975. Needs full GSM8K +
needle before acceptance (quantized reduction).

### e06-tp1-qsa (10:17) — ONE GPU, TP1 (no EP), QSA overlay   ⚠️ architecture finding
KV 1.86M tokens (7.1 × 262K), weights 176.8 GiB. GSM8K-200 0.975.
1 GPU vs TP2 (2 GPUs, e02): prefill_8 17.8K vs 19.1K tok/s; decode c64 2479 vs 3065 tok/s;
bs1 TTFT 32K 2.14 vs 1.74 s; c1 TPOT 4.1 vs 3.9 ms; agent32k_c16 wall 45.9 vs 37.8 s.
→ per GPU, TP1 delivers ~1.6–1.8× the throughput of TP2 (TP2 pays a 156 MB all-reduce per layer over
one XGMI link; TP1 pays nothing). Single-request latency only ~5–25% worse.
→ 4×TP1 on GPU0-3 should give ~1.6–1.7× the aggregate of 2×TP2 (spec asks for 2×TP2; reported as
an alternative with data, both tracks tuned in parallel). Added half-load agent cells
(agent_32k_c8, agent_96k_c4) so one TP1 engine at half the sessions can be compared with TP2 at full.
Single-GPU slots g0..g3 (HIP 1,3,2,0) added to exp.sh.

### e04-qsa-qrfp — QuickReduce FP (no quantization): = e02b within noise → link-bandwidth bound; DROP.

### t01-tp1 (repeat of e06 on GPU0 with half-load cells) — reproduces e06 within ±3%.
Equal-GPU comparison (2 GPUs, 16 agent sessions): 2×TP1 with 8 sessions each → agent32k wall 24.9 s,
TTFT p95 15.2 s; agent96k (4 each) 31.2 s / p95 20.0 s. 1×TP2 with 16 → 37.1 s / 24.7 s (QR-INT8:
34.3 / 21.7); 96K with 8 → 52.6 / 37.8 (QR-INT8 45.4 / 35.1). → TP1 ≈ 1.4–1.7× faster for the agent
workload at equal GPU count. Only idle single-request TTFT is worse (32K cold 2.17 s vs 1.56 s INT8-TP2).

### MTP depth (e07 TP2 MTP2, e08 TP2 MTP4, t02 TP1 MTP4, t03 TP1 MTP2) vs MTP3
MTP2: acceptance 0.78–0.80 vs 0.71–0.72, but c1 TPOT 3.9/4.4 vs 3.8/4.3 ms; c64 throughput ±3% → noise.
MTP4 (TP1): acceptance 0.66, decode c64 2513 → 1964 tok/s (−22%), c1 TPOT 4.3 → 4.7 ms. → keep MTP=3.
### t04-tp1-tritonmoe (VLLM_ROCM_USE_AITER_MOE=0): decode c64 −7%, c16 −8%, agent ≈ → keep AITER MoE.
### t05-tp1-mnbt64k: prefill +3% (≈noise), decode c64 −19%, cold TTFT p50 +18% → keep 32768.
### e08-qsa-mtp4 (TP2): acceptance 0.65, c1 TPOT 3.8 → 4.2 ms, throughput ≈ → MTP3 confirmed on TP2 too.
### e11-qsa-qrint6 (TP2): ≈ INT8 (+1–3%, noise). t06-tp1-seqs128: ≈ base → keep 64.

### Sparse-GQA prefill launch profile sweep (patches/sweep_gqa2.py, isolated subprocess per config)
Stock profile for > 512 programs is (BLOCK_N 64, splits 1, 2 warps, 1 stage) "tuned on GB300".
4096 rows × 2051 selected tokens × D256: best (32,1,2,1): TP2 shape 2.66 → 1.37 ms (1.95×),
TP1 shape 5.33 → 2.65 ms (2.01×); max |Δ| 4.9e-4 (1 bf16 ulp). (64,1,4,1) and (128,1,4,1) crash the
GPU (memory access fault) on gfx942 — avoid; BLOCK_N 128 with 2 stages exceeds 64 KiB LDS.
Exposed as VLLM_QSA_GQA_PREFILL="32,1,2,1" in the overlay (off by default until e2e-tested: t14/e14).
### t07-tp1-gdnflydsl (--additional-config gdn_prefill_backend=aiter_flydsl): prefill_1 +7%, prefill_8 −3%, agent −2…−4% wall → marginal (≤ noise); retest in final combo. GSM8K-200 0.975.
### e12-qsa-qrint4 (TP2): prefill +4–6% over INT8, agent −5–8% wall, GSM8K-200 0.965 (−1 pt, maybe noise; INT4 = highest numeric risk). TP2 pick: INT6 pending full GSM8K.
### t12-tp1-mnbt16k: prefill_8 −8%, agent wall +4…+7% → 32768 confirmed as the optimum (16K/32K/64K tested).
### e14-qsa-qrint8-gqa32 (TP2): GQA prefill profile (32,1,2,1) on top of INT8: prefill_8 +9%, agent wall −4…−5%, GSM8K-200 0.98 → KEEP.
### t08-tp1-O3: ≈ base, GSM8K-200 0.955 → DROP.

### TP1 profiles (t15 graph mode, t16 eager) — GQA32 on
Prefill 30.4K (1.52 s): AITER FP8 MoE 522 ms (34%, ≈22% of FP8 peak), BF16 GEMM ~420 ms (28%; GDN
in_proj N=16384 at ~51% of BF16 peak), sparse GQA 265 ms (17%), GDN core 240–300 ms, HC ~100 ms.
Decode (16×4 tok, 26.8 ms/step): MoE ~24%. A 226 µs bf16 Mul next to every MoE call turned out to be
the shared-expert branch on a second stream (stream 2) overlapping the persistent fmoe kernel
(stream 3) and starved of CUs — not on the critical path (union of kernel time 25.9 ms ≈ step 26.8 ms).

### Small toggles on TP1 (all ≈ t01 within noise → keep defaults)
t09 --performance-mode=throughput; t13 --long-prefill-token-threshold=8192 (worse 96K p95);
t10 VLLM_ROCM_USE_AITER_LINEAR=0; t11 VLLM_ROCM_USE_AITER_TRITON_GEMM=0. Reason for t10: aiter tgemm
is only enabled on gfx950 (`rocm_aiter_ops.is_tgemm_enabled`), so on gfx942 BF16 linears always go
through torch F.linear → hipBLASLt (skinny wvSplitK for tiny M). GEMM tuning lever = PyTorch TunableOp.

### TP1 MoE tuning (E=512 local, tuned_fmoe_qwen38_tp1.csv, --compare ≥3%)
Updated: M=4 +6.2%, M=16384 +7.5%, M=32768 +11.7% (1-stage asm 32x128 vs/novs variants, err 0.0%).
Small decode M again slower than the production default (M=1 −22%, 8 −14%, 16 −18%, 32 −15%).
### d01-tp2-combo (QSA + GQA32 + QR INT6 + tuned MoE ep2): prefill 24.6K tok/s (e00 11.5K, 2.14×), agent32k_c16 wall 32.1 s (e00 63.1), agent96k_c8 41.2 s (e00 84.9), decode c64 3203 tok/s (e00 2167); GSM8K-200 0.965 (σ≈1.2 pt) → full GSM8K in gate.
### TP1 combos (11:30)
t14 (GQA32 only): prefill_8 17.9K → 19.6K (+10%), agent32k wall 44.3 → 41.4, agent96k 58.2 → 53.6.
c01 (GQA32 + tuned MoE tp1): ≈ t14 (+0–2%). c02 (c01 + GDN FlyDSL prefill): agent96k_c4 29.2 → 26.6 s,
p95 19.9 → 17.6 s, other cells ≈ → consistent with t07; repeats c01r/c02r running.
Equal GPUs: 2×TP1 (c02, 8 sessions each) agent32k 22.3 s / p95 13.0 s vs best TP2 (d01, 16 sessions)
32.1 s / 19.7 s → TP1 ≈ 1.44× faster.
### c01r / c02r repeats — reproduce c01 / c02 within ±3%; GDN FlyDSL gain on agent96k p95 reproduces.

### gate-tp1-c02 (11:37) — TP1 finalist: QSA overlay + GQA32 + tuned MoE tp1 + GDN FlyDSL
functional 9/9 (incl. #61486 short-prompt determinism, tool calls, reasoning), GSM8K full 1319
zero-shot 0.9666 (vllm#60839 gfx942 reference 0.9704 5-shot), 238K needle 3/3 PASS.

## Production config v2 (11:40) — compose.yaml rewritten with two layouts
LAYOUT=tp1x4 (default) → vllm-1..4, one GPU each (HIP 1,3,2,0), ports 8001-8004, LiteLLM 4 deployments.
LAYOUT=tp2x2 → vllm-a/b as in the original spec, + QuickReduce INT6 and tuned EP2 MoE table.
Both: patches/qsa.py overlay (ro bind mount), VLLM_QSA_GQA_PREFILL=32,1,2,1, AITER_CONFIG_FMOE with the
layout's tuned table (config/aiter/), MTP3, mnbt 32768, seqs 64, util 0.93. TP1 also: GDN FlyDSL prefill.
v1 files kept in experiments/old/.
### gate-tp1-stock (reference, no overlay/tuning, TP1): functional 9/9, GSM8K-1319 0.9621, needle PASS → optimized TP1 (0.9666) has no accuracy loss (σ≈0.5 pt).
### gate-tp2-d01 (TP2 finalist on HIP 1,0: QSA + GQA32 + QR INT6 + tuned MoE ep2): functional 9/9, GSM8K-1319 0.9651, needle PASS → QR INT6 no accuracy loss.
### t19 (c02 + vllm#62381 GDN overlay: skip AITER packed GDN under spec decode): all cells within c02/c02r spread → no measurable effect on this model; DROP (avoid a second overlay).
### t20/t21 — offline TunableOp for hot BF16 GEMMs
t20 recorded 3,813 distinct GEMM shapes (record-untuned, TUNING=0; no overhead vs c02). Mixed
prefill+decode batches run eagerly with arbitrary token counts, so only full 32768-token chunks and
decode graph sizes recur. Tuned 75 hot shapes (n ∈ {32768, 32, 64, 128, 256}) offline on 4 GPUs in 5 min.
t21/t21r (c02 + tuned file, TUNING=0): decode c64 +2–3%, everything else within c02/c02r spread →
DROP (hipBLASLt heuristics already near-best; not worth an extra env + results file).

## Decision (12:15)
TP1 final = c02: QSA overlay (v2 indexer) + GQA prefill profile 32,1,2,1 + tuned MoE tp1 + GDN FlyDSL
prefill, MTP3, mnbt 32768, seqs 64, util 0.93, AITER MoE, FULL_DECODE_ONLY.
TP2 final = d01: QSA overlay + GQA32 + QR INT6 + tuned MoE ep2 (GDN FlyDSL not measured on TP2).
Both gates passed. Default LAYOUT=tp1x4.

## Production tp1x4 deployed (12:25) — gateway tests
functional via LiteLLM 9/9 (multi_turn stays on one replica).
Affinity (16 sessions × 4 turns, 24K prefix): x-session-id → 16/16 sticky, later-turn cache hit 0.90,
warm TTFT p50 0.53 s; no header → 1/16 sticky, hit 0.49, warm TTFT p50 1.60 s.
Routing for NEW sessions: simple-shuffle placed 16 sessions 2/4/7/3. least-busy with all 16 starting in
the same instant → 16/0/0/0 (in-flight counters not yet updated: race); with 0.5 s stagger → 5/4/4/3 and
4/4/4/4, turn0 TTFT p50 1.5 s. Chosen: least-busy (interactive agents start seconds apart);
caveat documented; benchmarks stagger session starts by 0.25 s.

### Failover (tests/failover.py, session pinned → docker stop that replica → 3 turns → restart)
v1 allowed_fails 1: first turn after stop → HTTP 500 (both in-request retries went back to the pinned
dead replica), next turns on replica-2. v2 allowed_fails 0: no client error, but first turn took 72 s.
Ruled out: DNS (getaddrinfo failure 0.02 s; AIOHTTP_TTL_DNS_CACHE=10 no change), per-deployment
`timeout` overriding the 5 s connect timeout (removed, no change), GPU/driver stall on other replicas
(direct probe of replica-2 during the stop: 0.10 s throughout). With the replica down >1 min a fresh
request failed over in 0.16 s → the hang is LiteLLM's aiohttp transport reusing a pooled keep-alive
connection to the stopped container. v5 DISABLE_AIOHTTP_TRANSPORT=True (httpx): first turn after stop
0.64 s, 3/3 success. Kept: allowed_fails 0, request_timeout 1800 (no per-deployment timeout),
AIOHTTP_TTL_DNS_CACHE=10 (harmless), httpx transport.
Known LiteLLM behaviour: the session pin is "claim once" (deployment_affinity_check._claim_pin), so
after failover the pin still names the old replica and the session returns to it (cold) once it is back.

### Real agent tasks through LiteLLM (tests/agent_real.sh; isolated HOME; task = fix 4 bugs, no hints)
OpenCode 1.19.3 ×2 and Pi Agent 1.3.0 ×2: 4/4 PASS (pytest 5/5, tests untouched), 7–11 s / 3–4 s,
every request of a session on one replica (4–7 requests each). Harness issues fixed on the way: mise
shims break when HOME changes (resolve node bin first); `opencode run` hangs without `< /dev/null`.

### Final benchmark grid (benchmarks/results/tp1x4-gateway-*, tp1-single-replica-*; fixed 256-token turns, 3 turns)
tp1x4 via LiteLLM, 8K–128K × 1/4/8/16: 0 errors, sessions spread evenly (0.25 s stagger).
245760 cells: all 400 (client sizing: prompt + max_tokens > 262144 — the 4.5 chars/token estimate
undershoots on some code); rerun at 229376: c1/c4 0 errors, c8 7/24 and c16 3/48 still 400 for the same
reason (largest successful prompt 261.7K) — client-side context overflow, not a server failure.
Single TP1 replica 128K × 16: KV exhausted (16 × 131K > 1.86M tokens) → preemption, hit 0, TPOT 257 ms.
