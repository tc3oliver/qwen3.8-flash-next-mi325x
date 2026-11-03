# Phase 1 HIP 3: vllm#62590 sparse-GQA correctness on MI300X (v0.32.0)

Scope: correctness only. No timings were taken. Device: HIP 3 = /dev/dri/renderD152 (+ /dev/kfd), seen as HIP 0 in the container. No other GPU, no container stop/restart.
Image: vllm/vllm-openai-rocm:v0.32.0, digest sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905. Overlay (qsa.py, indexer_qsa.py) mounted read-only; stock/overlay hashes not rechecked by me (lead verified).

## Commands
`results/mi300-2026-11/phase1-hip3/run.sh` = docker run with `--device /dev/kfd --device /dev/dri/renderD152 --group-add video --group-add render --ipc host --security-opt seccomp=unconfined`, overlay mounts (per overlay/README.md), scripts at /s.

    run.sh python3 /s/probe.py                                  -> raw/probe_v0.32.0.txt
    run.sh bash -c 'pip install -q pytest (if missing); python3 -m pytest -v -rA -p no:cacheprovider /t/test_qsa_kernels.py'   -> raw/pytest_v0.32.0.txt
    run.sh python3 /s/dispatch_ref.py                           -> raw/dispatch_ref_v0.32.0.txt

## 0. Arch detection
```
device_count 1  gfx942:sramecc+:xnack- CUs 304
is_rocm True on_gfx942 True
```

## 1. test_qsa_kernels.py: 19 passed, 0 failed (single run, 7.87 s)
All 19 cases PASSED (full per-case list in raw/pytest_v0.32.0.txt), including test_qsa_select_decode_rows_replays_in_cuda_graph (known flaky on MI325X at ~10% on v0.32.0; passed in this single run, which says nothing about its flake rate here: unverified).

## 2. Dispatch (real code, gfx942 detected: arch gfx942, 304 CUs, on_gfx942=True)
Real `qsa_sparse_paged_attention` was run; the launch of `_qsa_sparse_paged_gqa_splitk_kernel` was wrapped by a spy that records the kwargs actually passed (BLOCK_M/BLOCK_N/splits/warps/stages). Rule in overlay/qsa.py: BLOCK_M = next_pow2(q_heads/kv_heads) (line 1053); programs = rows*kv_heads (line 1054); `base_programs <= 512` branch at line 1065 (BLOCK_N=64, splits 4; the else branch = >512 programs, splits 1); gfx942 override `if on_gfx942() and block_m >= 16: block_n = 32` at lines 1070-1075 (condition at 1074).

| shape | q/kv heads | BLOCK_M | programs <= 512 | programs > 512 (gfx942) |
|---|---|---:|---|---|
| TP1 | 24/2 | 16 | BLOCK_N=64, splits 4 | BLOCK_N=32, splits 1 |
| TP2 | 12/1 | 16 | BLOCK_N=64, splits 4 | BLOCK_N=32, splits 1 |
| TP4 | 6/1 | 8 | BLOCK_N=64, splits 4 | BLOCK_N=64, splits 1 (unchanged by #62590) |

Measured boundary: TP1 256 rows (512 programs) -> 64; 257 rows (514) -> 32. TP2 512 -> 64; 513 -> 32. TP4 512/513/1024 -> always 64. Warps 2, stages 1 everywhere.
=> BLOCK_M=8 TP4 is unaffected on gfx942: confirmed by executing the dispatch.

## 3. Kernel vs fp32 torch reference (max abs diff)
Data: seq 32000, page 1600, topk 2051 (512 local + 1539 random far), head_dim 256, identity block table (PR microbench layout). Reference: fp32 softmax(q.k * 256^-0.5) @ v over gathered indices. Pass tol 2e-2 (loose; actual diffs ~3-5e-4). "force64" = same real dispatch with `vllm.platforms.rocm.on_gfx942` patched to return False, i.e. the stock BLOCK_N=64 path on the same inputs.
Column "diff vs 64" is the max abs diff between the real-dispatch output and the force64 output.

| dtype | shape | rows | programs | mode | BLOCK_M | BLOCK_N | splits | max diff vs ref | diff vs BLOCK_N=64 | result |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| bf16 | TP1 | 256 | 512 | real | 16 | 64 | 4 | 3.309e-04 | 0.000e+00 | PASS |
| bf16 | TP1 | 256 | 512 | force64 | 16 | 64 | 4 | 3.309e-04 | 0.000e+00 | PASS |
| bf16 | TP1 | 257 | 514 | real | 16 | 32 | 1 | 3.329e-04 | 4.883e-04 | PASS |
| bf16 | TP1 | 257 | 514 | force64 | 16 | 64 | 1 | 3.329e-04 | 0.000e+00 | PASS |
| bf16 | TP1 | 1024 | 2048 | real | 16 | 32 | 1 | 3.656e-04 | 4.883e-04 | PASS |
| bf16 | TP1 | 1024 | 2048 | force64 | 16 | 64 | 1 | 3.616e-04 | 0.000e+00 | PASS |
| bf16 | TP2 | 512 | 512 | real | 16 | 64 | 4 | 3.692e-04 | 0.000e+00 | PASS |
| bf16 | TP2 | 512 | 512 | force64 | 16 | 64 | 4 | 3.692e-04 | 0.000e+00 | PASS |
| bf16 | TP2 | 513 | 513 | real | 16 | 32 | 1 | 3.813e-04 | 4.883e-04 | PASS |
| bf16 | TP2 | 513 | 513 | force64 | 16 | 64 | 1 | 3.813e-04 | 0.000e+00 | PASS |
| bf16 | TP2 | 1024 | 1024 | real | 16 | 32 | 1 | 3.674e-04 | 4.883e-04 | PASS |
| bf16 | TP2 | 1024 | 1024 | force64 | 16 | 64 | 1 | 3.432e-04 | 0.000e+00 | PASS |
| bf16 | TP4 | 512 | 512 | real | 8 | 64 | 4 | 4.272e-04 | 0.000e+00 | PASS |
| bf16 | TP4 | 512 | 512 | force64 | 8 | 64 | 4 | 4.272e-04 | 0.000e+00 | PASS |
| bf16 | TP4 | 513 | 513 | real | 8 | 64 | 1 | 3.226e-04 | 0.000e+00 | PASS |
| bf16 | TP4 | 513 | 513 | force64 | 8 | 64 | 1 | 3.226e-04 | 0.000e+00 | PASS |
| bf16 | TP4 | 1024 | 1024 | real | 8 | 64 | 1 | 3.407e-04 | 0.000e+00 | PASS |
| bf16 | TP4 | 1024 | 1024 | force64 | 8 | 64 | 1 | 3.407e-04 | 0.000e+00 | PASS |
| fp8 | TP1 | 256 | 512 | real | 16 | 64 | 4 | 3.368e-04 | 0.000e+00 | PASS |
| fp8 | TP1 | 256 | 512 | force64 | 16 | 64 | 4 | 3.368e-04 | 0.000e+00 | PASS |
| fp8 | TP1 | 257 | 514 | real | 16 | 32 | 1 | 3.418e-04 | 4.883e-04 | PASS |
| fp8 | TP1 | 257 | 514 | force64 | 16 | 64 | 1 | 3.323e-04 | 0.000e+00 | PASS |
| fp8 | TP1 | 1024 | 2048 | real | 16 | 32 | 1 | 3.517e-04 | 4.883e-04 | PASS |
| fp8 | TP1 | 1024 | 2048 | force64 | 16 | 64 | 1 | 3.345e-04 | 0.000e+00 | PASS |
| fp8 | TP2 | 512 | 512 | real | 16 | 64 | 4 | 3.646e-04 | 0.000e+00 | PASS |
| fp8 | TP2 | 512 | 512 | force64 | 16 | 64 | 4 | 3.646e-04 | 0.000e+00 | PASS |
| fp8 | TP2 | 513 | 513 | real | 16 | 32 | 1 | 3.387e-04 | 4.883e-04 | PASS |
| fp8 | TP2 | 513 | 513 | force64 | 16 | 64 | 1 | 3.387e-04 | 0.000e+00 | PASS |
| fp8 | TP2 | 1024 | 1024 | real | 16 | 32 | 1 | 3.481e-04 | 4.883e-04 | PASS |
| fp8 | TP2 | 1024 | 1024 | force64 | 16 | 64 | 1 | 3.508e-04 | 0.000e+00 | PASS |
| fp8 | TP4 | 512 | 512 | real | 8 | 64 | 4 | 5.070e-04 | 0.000e+00 | PASS |
| fp8 | TP4 | 512 | 512 | force64 | 8 | 64 | 4 | 5.070e-04 | 0.000e+00 | PASS |
| fp8 | TP4 | 513 | 513 | real | 8 | 64 | 1 | 3.434e-04 | 0.000e+00 | PASS |
| fp8 | TP4 | 513 | 513 | force64 | 8 | 64 | 1 | 3.434e-04 | 0.000e+00 | PASS |
| fp8 | TP4 | 1024 | 1024 | real | 8 | 64 | 1 | 3.509e-04 | 0.000e+00 | PASS |
| fp8 | TP4 | 1024 | 1024 | force64 | 8 | 64 | 1 | 3.509e-04 | 0.000e+00 | PASS |

All 36 rows (9 shapes x 2 dtypes x real/force64) PASS; max diff vs reference 5.07e-4 (fp8 TP4 512); real-vs-64 diff at most 4.883e-4 (one bf16 ulp, matches the PR's claim), and exactly 0 for every TP4 row and every <=512-program row (same kernel config).

## Notes / caveats
- "FP8 inputs": `qsa_sparse_paged_attention` asserts q/k/v are bfloat16 (overlay/qsa.py, `assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16`), so a native FP8 sparse-GQA call is not possible. The fp8 rows above are values quantized to fp8 (current_platform.fp8_dtype()) then stored as bf16; this is NOT an fp8 kernel path. FP8 inputs to the MQA indexer kernels are covered by the 19-case pytest (fp8 params passed).
- BLOCK_N=32/64 is chosen by the dispatch, not a public parameter; I did not hand-force BLOCK_N on the kernel (no test of BLOCK_N=32 on TP4 shape; PR says it is 5x slower there, not re-measured).
- Tests covered the first 32000-token layout with a single seed; no multi-request / block_table-with-holes / negative-index case for the GQA kernel (only the existing pytest masks cases).

## Unverified / not run
- v0.33.0 (non-rocm72 tag): not run (not pulled).
- Flakiness rate of the CUDA-graph replay test on MI300X: unverified (1 run only).
- Performance on MI300X: not measured by design (other jobs on other GPUs).
- Native FP8 sparse-GQA: not supported by the code; see caveat above.
- upstream/ PR qsa.py not tested separately (overlay contains #62590; not diffed against upstream/.../qsa.py by me).
- No proposed diffs to overlay/ or upstream/.

## Raw files
- results/mi300-2026-11/phase1-hip3/raw/probe_v0.32.0.txt
- results/mi300-2026-11/phase1-hip3/raw/pytest_v0.32.0.txt
- results/mi300-2026-11/phase1-hip3/raw/dispatch_ref_v0.32.0.txt
- results/mi300-2026-11/phase1-hip3/run.sh, scripts/probe.py, scripts/dispatch_ref.py

---
# Addendum (revised scope, 2026-11-03 07:18-07:42): main nightly + v0.33.0-rocm72, microbench

## Images / SHAs
| image | digest | vllm | note |
|---|---|---|---|
| vllm/vllm-openai-rocm:nightly (re-pulled) | sha256:5800b3b4cd79defc1cbcc66adb83b3c49e574b49ca5c6ba556e17388515dc84f (id 65812e58df91) | 0.33.0rc1.dev573+g40941f2c4.rocm100 (torch 2.13.0+rocm10.0.0, triton 3.8.0) | image Created field is 1970 (unusable as a date); `ops/qsa.py` has NO gfx942 branch => #62590 not merged in this build |
| vllm/vllm-openai-rocm:v0.33.0-rocm72 | sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182 (id ac429221c28f) | 0.33.0 | stock ops/qsa.py sha 7a7c8501... (= v0.32.0 stock) |
| vllm/vllm-openai-rocm:v0.32.0 | sha256:14f6a575... | | used earlier for the 19-case test |

PR #62590: state OPEN, head `qsa-amd-sparse-gqa-gfx942-prefill` @ f87fca3760c55e5f610ce0db76bd002d253fcde5 (base 4e304b4d, 2026-10-20; main is now 437288d1). Only change to qsa.py is a 7-line hunk (pr62590/pr62590-qsa.hunk, sha a43bde34...). PR-head file `pr62590/qsa.pr62590.f87fca37.py` sha 3a5488d4... is byte-identical to upstream/vllm-62590-sparse-gqa-gfx942/qsa.py and to (v0.33.0-rocm72 stock + hunk).
Main layout differs from the PR base, so for nightly I applied the hunk to the nightly stock file with `patch` (succeeded, offset 14 lines): `pr62590/qsa.nightly+pr62590.py` sha e5aec2c7... (stock: `stock/qsa.nightly.py`, sha 49720339...). The overlay (#62589 + backport) was NOT used on nightly (layout `qsa_indexer.py`, does not apply).
Benchmark pairs are "stock vs stock+#62590 hunk only" (cleaner than the overlay, which also carries #62589).

## Dispatch / correctness
- v0.33.0-rocm72 + overlay: arch gfx942, 304 CUs, on_gfx942=True; dispatch_ref.py output identical to v0.32.0 (36 rows PASS, 0 failures; raw/dispatch_ref_v0.33.0-rocm72.txt). TP4 stays BLOCK_N=64 at 512/513/1024 rows; TP1/TP2 switch 64 -> 32 above 512 programs.
- nightly + PR hunk: gfx942 / 304 CUs / on_gfx942=True; 36 rows PASS, 0 failures, identical dispatch table (one value differs in the 4th digit: 4.271e-04 vs 4.272e-04; raw/dispatch_ref_nightly+pr62590.txt).
- 19-case overlay test on v0.33.0-rocm72: **1 failed, 18 passed** (raw/pytest_v0.33.0-rocm72.txt). The failure is `test_qsa_select_decode_rows_replays_in_cuda_graph`, `Mismatched elements: 8 / 152 (5.3%)` = the same flaky signature documented in overlay/README.md. Retries of that test alone (2): pass, fail (raw/pytest_cudagraph_retry{1,2}_v0.33.0-rocm72.txt). So it is flaky on MI300X too (v0.32.0: 1 run, pass; rocm72: full run fail, retry1 pass, retry2 fail). It tests the #62589 per-row decode path, not #62590. NOT reported as pass.
- 19-case test not run on nightly (overlay does not apply).

## Microbenchmark (PR's bench_gqa.py, one invocation per image; stock vs stock+#62590 hunk)
Conditions: HIP 3 only (rocm-smi GPU[1]); rocm-smi --showuse: all 8 GPUs 0-2% before and after every run (the "other job" GPU HIP 0 = rocm-smi GPU[3] showed 2% in one pre-snapshot, 0% otherwise); host load average 7.4-22 (CPU contention from other jobs, kernels are ms-scale GPU-bound); snapshots in raw/bench_*.txt. GPU was quiet => not marked noisy; the host-load caveat is stated. Two runs on rocm72 agree within ~1%.

| shape | rows | programs | rocm72 run1 stock/PR ms | rocm72 run2 stock/PR ms | nightly stock/PR ms | nightly speedup | max abs diff |
|---|---:|---:|---|---|---|---:|---:|
| TP2 12q/1kv | 520 | 520 | 0.408 / 0.275 | 0.409 / 0.278 | 0.409 / 0.274 | 1.49x | 4.9e-4 |
| TP2 | 1024 | 1024 | 0.786 / 0.359 | 0.786 / 0.361 | 0.787 / 0.359 | 2.19x | 4.9e-4 |
| TP2 | 4096 | 4096 | 2.667 / 1.382 | 2.663 / 1.379 | 2.666 / 1.373 | 1.94x | 4.9e-4 |
| TP1 24q/2kv | 260 | 520 | 0.414 / 0.279 | 0.411 / 0.278 | 0.413 / 0.277 | 1.49x | 4.9e-4 |
| TP1 | 1024 | 2048 | 1.542 / 0.735 | 1.540 / 0.738 | 1.538 / 0.727 | 2.11x | 4.9e-4 |
| TP1 | 4096 | 8192 | 5.362 / 2.809 | 5.358 / 2.809 | 5.350 / 2.794 | 1.92x | 4.9e-4 |
| TP4 6q/1kv | 520 | 520 | 0.366 / 0.353 | 0.370 / 0.355 | 0.365 / 0.355 | 1.03x | 0 |
| TP4 | 4096 | 4096 | 2.231 / 2.214 | 2.223 / 2.218 | 2.215 / 2.205 | 1.00x | 0 |

Reading: on MI300X the BLOCK_N=32 effect reproduces the PR's MI325X claim (TP1/TP2 ~1.5x at ~520 programs, ~1.9-2.2x at 1K-8K programs; PR claims 1.5x and 2.1-2.3x) on both rocm72 and main-nightly. TP4 (BLOCK_M=8) output is bit-identical (diff 0) and time within ~1-3% of stock (0.365 vs 0.355 ms at 520 rows is a small gain even though the same BLOCK_N=64 config; likely launch-path/noise, not claimed as a speedup).
Not run: the PR's sweep_gqa2.py (needs a `_QSA_GQA_PREFILL` hook that the overlay/PR file does not have, and was excluded as a new parameter search), bench_tp4.py (BLOCK_N=32-on-TP4 5x slowdown not re-measured).

## Extra raw files
raw/probe_v0.33.0-rocm72.txt, raw/pytest_v0.33.0-rocm72.txt, raw/pytest_cudagraph_retry{1,2}_v0.33.0-rocm72.txt, raw/dispatch_ref_v0.33.0-rocm72.txt, raw/dispatch_ref_nightly+pr62590.txt, raw/bench_v0.33.0-rocm72_run{1,2}.txt, raw/bench_nightly_run1.txt, raw/nightly_version.txt (contains an earlier failed attempt: no render group in nightly; fixed by using the numeric gid), bench.sh, pr62590/, stock/.
Unverified: nightly end-to-end; v0.33.0 (non-rocm72); whether the flaky CUDA-graph failure also occurs on nightly; no pushes/PR comments; overlay/ and upstream/ untouched.
