# MI300X verification, 2026-11 (index)

Index of the MI300X verification of vllm#62589 and vllm#62590 (2026-11-03). Every number below is copied from the linked REPORT.md, log or JSON file. The MI300X work is finished; nothing here is a stability sign-off (see "Unresolved stability risk").

## Environment (source: mi300-verify/reports/env.md, in the vllm-services repo)
- AMD Instinct MI300X x8 (not MI300A), gfx942, 304 CU per GPU, 192 GiB HBM per GPU (206141652992 B).
- Host: ROCm 7.0.2, amdgpu driver 6.14.14, kernel 6.8.0-142-generic, amd-smi 26.0.2. Container stacks differ: nightly torch 2.13.0+rocm10.0.0 / triton 3.8.0 (phase1-hip3/REPORT.md); v0.33.0-rocm72 HIP 7.2.53211 (deploy/.env.example); v0.32.0 ROCm 7.2.3 (deploy/.env.example).
- XGMI: all-to-all, weight 15, 1 hop; HIP 0-3 are mutually direct.
- GPUs used: phase 1 = HIP 0 (renderD128), HIP 1 (renderD136), HIP 3 (renderD152) and HIP 5 (renderD168, tests of the final #62589 commit); phase 2 e2e = Pair A HIP 6,7 (renderD176+184) and Pair B HIP 2,3 (renderD144+152), BDFs checked per run in `phase2-e2e/*.gpu.txt`. HIP 4 is production and was not touched.
- Container stack next to host ROCm 7.0.2: nightly = ROCm 10.0 (torch 2.13.0+rocm10.0.0, triton 3.8.0); v0.33.0-rocm72 = ROCm 7.2 (HIP 7.2.53211); v0.32.0 = ROCm 7.2.3 (deploy/.env.example, phase1-hip3/REPORT.md).
- With only some render nodes mounted, HIP ids inside the container are renumbered 0..k-1 (env.md).

## Images and PR heads
| item | value | source |
|---|---|---|
| vllm/vllm-openai-rocm:v0.32.0 | sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905 (id 03f01e2ac1e9) | phase1-hip1/REPORT.md, phase1-hip0/REPORT.md |
| vllm/vllm-openai-rocm:v0.33.0-rocm72 | sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182 | phase1-hip3/REPORT.md (addendum) |
| vllm/vllm-openai-rocm:nightly | sha256:5800b3b4cd79defc1cbcc66adb83b3c49e574b49ca5c6ba556e17388515dc84f, vllm 0.33.0rc1.dev573+g40941f2c4 | phase1-hip0/REPORT.md, phase1-hip3/REPORT.md |
| vllm#62589 head | all MI300X measurements use 404857c80123438104534eec919cb95900f5562e; the PR head is now 348ca71d (one test-only commit on top, +5/-1 in `tests/models/qwen4_exp/test_qsa_amd.py`, no kernel change) | phase1-hip0/REPORT.md, GitHub |
| vllm#62590 head | f87fca3760c55e5f610ce0db76bd002d253fcde5 (branch qsa-amd-sparse-gqa-gfx942-prefill, OPEN, base 4e304b4d) | phase1-hip3/REPORT.md |

## Phase 1 results
| phase | what | result | report |
|---|---|---|---|
| hip0 | flaky `test_qsa_select_decode_rows_replays_in_cuda_graph` (#62589 path) | v0.32.0 + repo overlay: 45 pass / 55 fail of 100; v0.33.0-rocm72 (overlay): 19 / 21 of 40; nightly + #62589 head 404857c8: 50 / 50 of 100. All failures `Mismatched elements: 8 / 152`. Not a kernel bug: the selected set is identical in every repeat, only the output order changes. Test-only fix (compare sorted): 100/100 pass on v0.32.0 and on nightly; the final commit 348ca71 was run 10 times on nightly (HIP 5) with 10/10 passing, but that console output was not saved, so it has no raw file here. See "CUDA Graph investigation: closed" below. | [phase1-hip0/REPORT.md](phase1-hip0/REPORT.md) |
| hip1 | #62589 tiled indexer microbench, nightly image, stock vs PR head 404857c8 (select, GPU ms median) | 32K 4 rows c64: 0.8937 -> 0.1338; 98K 1 row c64: 0.3456 -> 0.2417; 262K 4 rows c64: 1.3318 -> 0.5043. 262K 1 row c16: 0.1824 -> 0.1886 (full pass), 0.1871 -> 0.1895 (262K-only second pass, ranges overlap). Top-k sets identical in 23/24 cases (nightly); the 24th (98K 1 row c64 row 55) is a 1-ulp boundary near-tie (row55.log). Earlier v0.32.0 runs used an OLDER PR revision (1159-line file, SHA not recorded), ratios within 3%. Source: nightly-full.log, nightly-262k-pass2.log. | [phase1-hip1/REPORT.md](phase1-hip1/REPORT.md) |
| hip3 | #62590 sparse-GQA, dispatch + correctness + microbench | gfx942 / 304 CUs / on_gfx942=True on all three images; TP1/TP2 switch BLOCK_N 64 -> 32 above 512 programs, TP4 (BLOCK_M=8) stays 64. fp32-reference check: 9 shapes x (bf16, fp8-quantised-as-bf16) x (default dispatch, forced BLOCK_N=64) = 36 rows, all within the 2e-2 tolerance; max diff 3.8e-4 on the BLOCK_N=32 path (bf16 TP2 513), 5.07e-4 overall (fp8-as-bf16 TP4 512, BLOCK_N=64). nightly microbench speedup: TP2 1.49x (520) / 2.19x (1024) / 1.94x (4096); TP1 1.49x (520) / 2.11x (2048) / 1.92x (8192); TP4 1.03x / 1.00x with output diff 0. The flaky CUDA-graph test is unrelated to #62590 (see hip0). | [phase1-hip3/REPORT.md](phase1-hip3/REPORT.md) |

## CUDA Graph investigation: closed
- The order of the selected blocks produced by `top_k_per_row_decode` is not defined. `csrc/sampler.cu:281` (`atomicAdd(&smemFoundTopKValues[0], 1)`, permalink in the #62589 comment) hands out output slots in thread arrival order. The file is byte-identical in v0.32.0, nightly and the #62589 head.
- Measured (phase1-hip0 Addendum 6, `exp/`): the order differs between consecutive launches inside one process, and a hipcc test kernel with the same grid (8 blocks x 512 threads) shows 9-10 different wave arrival orders across launches. `HW_REG_XCC_ID` shows block b on XCC b in every process, so XCC placement does not explain it.
- Selected set: identical in 500/500 repeats on the failing inputs. Downstream attention output for the actual flaky order is bitwise identical to the tiled order; random permutations differ by about one bf16 ulp (synthetic q/K/V).
- Test fix: compare the selection per row after sorting (a multiset check: it still catches missing, extra, wrong and duplicated indices; it does not check order). Pushed to vllm#62589 as commit 348ca71 (test only, +5/-1).
- The earlier "pass/fail alternates with run number" pattern (nightly: 47 of 50 failures on even run numbers; rocm72: 20 of 21) did NOT reproduce in later runs (Addendum 6, E9: `PFFFFFFPFFPF`, `PFFFFFFFPP`, `FFPPFFPFPP`). It is an unexplained observation from one batch of runs and must not be read as an established pattern.
- Excluded as a function of the order: number of earlier unrelated kernels (E2), idle gap of 0-2 s (E5), L2/MALL eviction with a 2 GiB fill (E8). Observation, cause unknown: after a 4096^3 matmul the graph launch put "620 first" in 15/120 launches instead of 0/120 (E8), so GPU state before the launch can change the order.
- Not investigated by instruction: the production `monitoring-agent` / `metrics-exporter` monitoring was neither stopped nor tested. It is only recorded as an untested hypothesis in Addendum 6.
- Why MI300X failed more often than MI325X (v0.33.0-rocm72: 21/40 here vs 1/40 on MI325X, overlay/README.md:45) is not known.

## Phase 2: TP2 end-to-end, stock vs overlay
Setup: `vllm/vllm-openai-rocm:v0.33.0-rocm72` (sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182), model `Qwen3.8-Flash-Next-FP8` (revision 236dfdf2, matches HF main on 2026-11-03), TP2 + EP2, MTP 3, AITER FP8 MoE, QuickReduce INT6, tuned fMoE table (gfx942, 304 CUs), `FULL_DECODE_ONLY`, `--max-num-seqs 64`, `--max-num-batched-tokens 32768`, 262K context. Arm = stock (image files) or overlay (`overlay/qsa.py` + `overlay/indexer_qsa.py`); the only difference is the two mounts. `arm.sh` checks the sha256 of the files python actually imports (stock `7a7c8501…` / `ca8ede9f…`, overlay `f59670bf…` / `dbb72baa…`; all 12 hash checks OK) and the PCI BDFs of the pair (Pair A = `e6:00.0`+`96:00.0`, Pair B = `66:00.0`+`16:00.0`). Perfsuite subset only: prefill 8x32K, decode C64, agent 32K C16; 3 passes per arm per round. GSM8K, 238K needle and the functional gate were NOT run.

**Do not attribute the delta to one PR.** The overlay contains #62589 + #62590 + the `indexer_qsa.py` backport; e2e cannot separate them. For reference, the MI325X lab notebook (`docs/lab-notebook.md:256-257`, r1, 3 passes) reports about +100% prefill, +46% decode C64 and -43% agent wall for the same overlay.

Results (median [min-max] over 9 passes per arm; raw: `phase2-e2e/*.perf.json`, aggregation: `phase2-e2e/agg-output-final.txt`):

| metric | stock | overlay | overlay vs stock |
|---|---|---|---|
| prefill 8x32K tok/s | 11370 [9162-11930] | 22000 [15680-24650] | +93.5% |
| decode C64 tok/s | 1924 [1544-1944] | 2786 [2133-2851] | +44.8% |
| decode C64 TPOT ms | 22.5 [22-25.3] | 17.4 [17.2-20.7] | -22.7% |
| agent 32K C16 wall s | 60.7 [58.3-66.8] | 35.3 [33.6-39.1] | -41.8% |

Per-round overlay vs stock (all passes): prefill R1 +94.0%, R2 +87.6%, R3 +115.9%; decode C64 R1 +47.7%, R2 +44.8%, R3 +43.0%; agent wall R1 -43.4%, R2 -41.2%, R3 -40.9%. Dropping pass 1 (cold warm-up) of every run: prefill +100.5% / +96.5% / +111.9%, decode C64 +47.5% / +45.8% / +43.3%, agent wall -42.9% / -42.7% / -42.1%. Swapped round vs mean of R1 and R3: prefill -8.5%, decode C64 -0.3%, agent +1.6%. This is NOT a clean pair-bias estimate: it is confounded with host load and with which neighbour engines were running (R1 both arms concurrent; R2 stock ran about 5 min alone after the first overlay attempt died at 08:05:57; R3 stock ran next to one overlay engine on the other pair; the R2/R3 overlay retries ran concurrently with each other). Differences below about 5% are not resolvable (pass 2 vs pass 3 differ by up to about 5%).

**Test conditions differ by round. The nine passes are not interference-free.** Host load average (1 min, from `*.noise.txt`) includes the two engines themselves and other jobs on the host.

| round | stock | overlay | how it ran |
|---|---|---|---|
| R1 | Pair A, 07:48-07:53, load 24-25 | Pair B, 07:48-07:52, load 24-28 | The only round where both arms ran at the same time with no crash and no retry. Not quiet: load 24-28, and another agent's pytest loops (HIP 0/1) ran during part of it. |
| R2 | Pair B, 08:05-08:11, load 12-18 | Pair A, 08:27-08:31, load 22-27 (retry; first attempt crashed) | Not concurrent. The overlay retry overlapped root-cause experiments on HIP 0/1 (08:22-08:36, at most 12 short processes at a time) and a verification test on HIP 5 (08:29-08:32). None touched the pair GPUs, but they add host CPU load. |
| R3 | Pair A, 08:15-08:20, load 18-19 | Pair B, 08:27-08:31, load 22-24 (retry; first attempt crashed) | Same as R2. The stock arm (08:15-08:20) ran before those experiments started. |

R2 and R3 are therefore sequential comparisons under different host load, not concurrent pairs. The direction agrees in all three rounds; the prefill magnitude spans +88% to +116%. R1 is the only round where both arms ran at the same time, but it also has the highest host load (24-28). Treat the median over nine passes as indicative only. The two crashed first attempts are kept in `phase2-e2e/invalid-crash-attempt1/` and are not in any number above.

## Unresolved stability risk (MI300X): not excluded
Facts only:
- First attempts: overlay 2 of 3 crashed (R2 on Pair A during decode C64 after prefill_8 had finished, 08:05; R3 on Pair B in the prefill_8 of pass 2, 08:21); stock 0 of 3 crashed. R1 overlay (Pair B) did not crash. A rerun of both overlay arms (same flags) completed all passes.
- Signature in both server logs: `HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources` on HSA queues, then `hipErrorUnknown` / `EngineDeadError`. Both HSA abort lines carry `Available Free mem : 0 MB` (overlay-r2-A.server.log:573-574, overlay-r3-B.server.log:650-651). In R2 an aiter line `Error in moe_sorting: CUDA error: unknown error` (:575) comes right after the abort, before the Python traceback, which ends at `ops/qsa.py:819` (`_qsa_mqa_paged_kernel[grid]`, during `torch.cuda.current_device`); the first error is the HSA queue abort, so the frame is where it surfaced.
- Host load average (1 min) was about 18 around the first crash and 14-19 around the second.
- Logs kept: `phase2-e2e/invalid-crash-attempt1/` (server logs, consoles, gpu/hash/noise files), `*.server.log` of the other arms, `system-snapshot-after-crashes.txt` (`rocm-smi` RAS counters: all 0 on 8 GPUs; `dmesg` and `journalctl -k` are not readable without `adm` group or sudo, so no kernel log could be saved).
- Not done, by decision: stress test, GPU reset, any action that could affect production.
- Cause unknown. The data do not show that the overlay, ROCm, the host or the hardware is responsible. Triton "JIT compilation during inference" warnings appear in all arms (9 lines in each completed arm, 8 and 9 in the crashed R2/R3 attempts) and are not specific to the overlay. 3 vs 3 first attempts is far too few to separate overlay from chance.
- Status: **"unresolved stability risk on MI300X"**. This run is NOT a stability acceptance. It says nothing about #62589 or #62590 on MI325X.

## GitHub status (2026-11-03)
- vllm#62589: test-only commit 348ca71 pushed (fast-forward from 404857c8); MI300X kernel-test comment posted: https://github.com/vllm-project/vllm/pull/62589#issuecomment-6599007249
- vllm#62590: MI300X kernel-test comment posted: https://github.com/vllm-project/vllm/pull/62590#issuecomment-6599007801
- PR descriptions of #62589 and #62590 are being updated by the MI325X side; not touched here. The e2e and crash results above were not posted upstream.
- Posted comment texts: `upstream-drafts/62589-comment.md`, `upstream-drafts/62590-comment.md` (`*.v1.md` are earlier drafts).

## Not verified / open
From hip0:
- Cause of the MI300X flake rate vs MI325X not identified: v0.32.0 55/100 (MI300X) vs 4/40 (MI325X), v0.33.0-rocm72 21/40 vs 1/40 (MI325X numbers: overlay/README.md:45). The run-number alternation seen in one batch did not reproduce (phase1-hip0 Addendum 6).
- v0.33.0 non-rocm72 not run (hip0 and hip3).
- Bitwise order-independence of downstream `qsa_sparse_paged_attention` not tested beyond synthetic cases (random permutations differ by 3.9e-3 to 7.8e-3, one bf16 ulp; failing-row order gave 0 diff).
- Sensitivity test used synthetic q/K/V, not model activations.
- Sorted-test loop on nightly ran on HIP 1 instead of the assigned HIP 0.
From hip1:
- Single GPU, one seed per case, no MI325X comparison; GPU load sampled only at start/end of each run.
- Host load average 6-13 during runs (main-run1: host load 13, results consistent with the quiet runs, within 3%; nightly runs 6-11).
- Index-level order within a row not compared (sets only); only row 55 examined for the near-tie.
- #62589 `qsa_indexer.py` hunk not benchmarked (microbench calls ops/qsa.py only); end-to-end 262K regression not measured.
- upstream/.../qsa.py in the repo differs from PR head 404857c8 (1159 vs 1326 lines).
From hip3:
- "FP8" rows are fp8-quantised values stored as bf16: `qsa_sparse_paged_attention` asserts bf16 inputs, there is no fp8 kernel path.
- Synthetic activations, single seed, identity block table.
- Host load average 7.4-22 during the timed microbench (GPU use 0-2%).
- Flake rate of the CUDA-graph test: see phase1-hip0 for the measured rates (hip3 ran it once per image); BLOCK_N=32 on TP4 (5x slower per PR) not re-measured; sweep_gqa2.py and bench_tp4.py not run.
- PR #62590 base is old (4e304b4d); on nightly the 7-line hunk was applied with `patch` (offset 14), not the PR branch itself.

## Reproduce
Scripts live next to each report; all need a Docker host with the MI300X and the images above. Triton caches were kept in `mi300-verify/cache/` (in the vllm-services repo); scripts mount those paths via `-v`, so adjust the `-v` cache paths (and `TRITON_CACHE_DIR`) for your machine. Images without a `render` group name (nightly) need the numeric gid: `--group-add $(getent group render | cut -d: -f3)`.
- hip0: `phase1-hip0/scripts/repeat.sh <image> <tag> <N>`, `repeat_main.sh`, `repeat_main_sorted.sh`; diagnostics `dbg.py`, `order.py`, `sens.py`; test-only fix `phase1-hip0/proposed/test_qsa_amd_sorted.diff`.
- hip1: `phase1-hip1/run_v032.sh`, `run_main_on_v032img.sh`, `run_nightly.sh full|262k-pass2 262k`; row-55 check `row55.py`.
- hip3: `phase1-hip3/run.sh <cmd...>` (IMG env selects image; overlay mounts), `phase1-hip3/bench.sh <image> <stock.py> <pr.py> <tag>`; scripts in `phase1-hip3/scripts/`.
- Phase 2: `phase2-e2e/RUNBOOK.md`, `arm.sh <stock|overlay> <A|B> <round>`, `round.sh <n> <stockPair>`, `agg.py`. They use `tools/experiments/exp.sh`; set `RENDER_MAP` in `deploy/.env` (sample: `deploy/.env.mi300x.example`) so only the chosen GPUs' render nodes are mounted and the production GPU is refused. The e2e runs were made with the first version of this change (commit 0b1c69a, hard-coded table); the follow-up commit makes the same mapping opt-in and restores the MI325X defaults. A corpus is built by `tools/benchmarks/make_corpus.sh`.
