# Phase 1 / HIP 0: flaky `test_qsa_select_decode_rows_replays_in_cuda_graph` on MI300X

結論: no kernel bug found. The test's exact-equality assumption is unsound: the top-k step returns the same selected blocks in an UNDEFINED ORDER (the kernel gives no order contract for them; the order observed in a process is fixed or nearly fixed, and differs between processes, see 'Failure pattern' below).
Not a tie (scores are distinct), not CUDA-graph specific.

## Setup
- GPU: HIP 0 only (`--device /dev/kfd --device /dev/dri/renderD128`), image `vllm/vllm-openai-rocm:v0.32.0` (03f01e2ac1e9), overlay mounted per overlay/README.md.
- Triton cache persisted in `triton-cache/` (TRITON_CACHE_DIR=/cache).

## 1. Reproduction (fresh pytest process per run, test alone)
`scripts/repeat.sh <image> <tag> <N>`
| image | runs | pass | fail |
|---|---|---|---|
| v0.32.0 | 100 | 45 | 55 |
| v0.33.0-rocm72 | 40 | 19 | 21 |
| v0.32.0 with sorted-compare test (proposed) | 100 | 100 | 0 |
v0.33.0 (non-rocm72): not run (image not present locally).
All 55+21 failures: `Mismatched elements: 8 / 152`. Per-run logs: `runs/<tag>/run<N>.log`, `runs/<tag>/summary.txt`.
Contrast with MI325X (overlay/README.md:45, kernel-test-repeats.txt): v0.32.0 4/40 failures there vs 55/100 here; v0.33.0-rocm72 1/40 there vs 21/40 here. Cause unknown (not investigated).

## 2/3. Evidence (`scripts/dbg.py`, `scripts/order.py`, outputs in `dbg/`)
Test inputs are fixed-seed, so every failure hits the same row (row 2, columns 4..11 = two 4-wide blocks swapped).
Failing row 2 (visible_blocks=352, length 1411), selected block starts:
- graph:  [896, 620, 1388, 652, tail 1408..1410]
- tiled:  [896, 1388, 620, 652, tail 1408..1410]
- eager rows: same as tiled
Same four blocks, two swapped. Scores (fp32 reference / rows kernel / tiled kernel agree to 2.4e-7):
block 224 -> 2.9505, 155 -> 3.4795, 347 -> 3.5249, 163 -> 2.7535; next best 83 -> 2.6718. No ties at the cut line (gap 0.08), so tie-ordering is ruled out.
- fp8/bf16 rounding: rows vs tiled logits max abs diff 2.4e-7 over visible columns; selection identical as a set. Ruled out.
- Unwritten slots: set equality holds, tail slots equal; also covered by `test_qsa_select_decode_rows_ignores_unwritten_logits` (passes). No evidence.
- Sync/graph memory reuse: the same order flips happen in plain eager calls (below), so replay is not the cause.

500 repeats on identical inputs (`dbg/order_run1.txt`):
```
graph_replay(rows kernel): exact_equal=2/500   sorted_equal=500/500 distinct_row2_orders=2
eager rows kernel:         exact_equal=0/500   sorted_equal=500/500 distinct_row2_orders=1
eager tiled kernel:        exact_equal=4/500   sorted_equal=500/500 distinct_row2_orders=2
```
Even tiled vs tiled (same kernel, same inputs) is not bit-identical in order; the selected set is identical in 100% of calls.
`dbg/dbg_run1.txt`, `dbg/fail{1,2,3}.json`: 200 replays, 5 with the 8-element swap, eager vs tiled 0 mismatches in that process.
Source of the unordered output: `ops.top_k_per_row_decode` (ROCm branch of `qsa_select_paged_tokens`, overlay/qsa.py:986); kernel source read later, see Addendum 2 (corrected in Addendum 4).

## 4. Fix (test only; overlay/ not edited)
`proposed/test_sorted.diff` (full file: `proposed/test_qsa_kernels.py`): compare `captured.sort(dim=1).values` with `expected.sort(dim=1).values`.
Before: 55/100 fail. After: 100/100 pass (`runs/v0.32.0-sorted-test/summary.txt`). The assertion still fails if a different block is selected.
Unverified: that downstream `qsa_sparse_paged_attention` is order-insensitive (mathematically a set-sum; bitwise equality of its output across orders not tested).

## Files
scripts/{repeat.sh,dbg.py,order.py}, runs/*, dbg/*, proposed/*, (triton caches moved to mi300-verify/cache/phase1-hip0/; scripts mount $D/triton-cache* so edit the -v path to rerun).

# Addendum: vLLM main (nightly) + PR #62589 head (priority target)
- Image: `vllm/vllm-openai-rocm:nightly`, digest `sha256:5800b3b4cd79defc1cbcc66adb83b3c49e574b49ca5c6ba556e17388515dc84f` (id 65812e58df91), `vllm.__version__` = 0.33.0rc1.dev573+g40941f2c4.
- PR #62589 head `404857c80123438104534eec919cb95900f5562e` (branch `qsa-amd-indexer-tiled`). Files fetched from that SHA into `main/` (sha256 in `main/SHA256SUMS`) and mounted read-only over the image's `ops/qsa.py`, `qsa_indexer.py`, plus `tests/models/qwen4_exp/test_qsa_amd.py` as `/t/test_qsa_amd.py`. No rebuild, no model download.
- Note: the nightly image has no `render` group name; the run script uses numeric gid 993 (`scripts/repeat_main.sh`).
- Test: `test_qsa_amd.py::test_qsa_select_decode_rows_replays_in_cuda_graph`, fresh process per run, HIP 0.

| run | pass | fail |
|---|---|---|
| first 20 (`runs/nightly-pr62589-first20`) | 10 | 10 |
| extended 100 (`runs/nightly-pr62589`, new 100 runs, includes the same cold start) | 50 | 50 |
| same test with set-compare (sorted) (`runs/nightly-pr62589-sortedtest`, run on HIP 1) | 100 | 0 |
All failures: `Mismatched elements: 8 / 152 (5.3%)` at test line 645, identical signature to v0.32.0.
Order check on main (`dbg/order_main.txt`, 500 repeats each, same inputs): graph replay exact 497/500, eager rows 500/500, eager tiled 493/500, sorted-equal 500/500 for all. Same conclusion: the selected set is identical in 500/500; the order is mostly stable within a process (see Addendum 4 for the exact counts), so this is NOT an i.i.d. per-launch race.
Fix = test-only: `proposed/test_qsa_amd_sorted.diff` (applies to `tests/models/qwen4_exp/test_qsa_amd.py` in the PR). overlay/ and upstream untouched, nothing pushed/commented.
v0.33.0 non-rocm72: not run. Positive control on v0.32.0 + overlay already done above (45/100 pass), so no extra control needed.
Unverified: kernel source of `top_k_per_row_decode`; bitwise order-independence of downstream sparse attention.
No experiment containers left running.

# Addendum 2: source evidence and downstream sensitivity
## Is the output order defined? No (kernel source, unordered by construction)
`top_k_per_row_decode` is `csrc/sampler.cu` (ROCm branch of `qsa_select_paged_tokens`, qsa.py `ops.top_k_per_row_decode`). Fetched `src/sampler.v0.32.0.cu` (tag v0.32.0) and `src/sampler.404857c8….cu` (PR head): byte-identical (`cmp`), so #62589 does not touch it. Hashes: `src/SHA256SUMS`.
- sampler.cu:281 `int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);` elements whose bin is below the threshold bin are written to output slot `dstIdx` as threads reach them: slot = arrival order of a shared-memory atomic across the block's threads (racy by construction, nothing sorts them afterwards).
- sampler.cu:295 / :308 same pattern (`atomicAdd` on `smemFinalDstIdx` / histogram) for the threshold-bin items.
- Only the threshold-bin items are sorted (insertion sort lambda :479, `baseIdx` :481 and :534, `FinalSort` used :530, type defined :358), and they are placed after the items already written by the atomics.
- (Corrected) sampler.cu:403-405 "Indices are not sorted by their corresponding logit" is inside the `if (rowLen <= topK)` short-row shortcut only. The failing row has 352 visible blocks vs k=4, so it does NOT take that path; that comment is not evidence for the failing case.
- Caveat: atomicAdd arrival order explains why the kernel's output order is UNDEFINED by contract; it does NOT by itself explain the alternating per-process pattern (Addendum 4).
Conclusion: the selected SET is deterministic, the order of the "clearly above threshold" part depends on atomic arrival; not a bug, so the sorted-compare test fix is correct. The failing row 2 swaps blocks 155 and 347 (scores 3.48 / 3.52, both above the cut), plausibly produced by this path (not proven).
(ROCm uses this kernel, not `persistent_topk`/`cooperative_topk`, which are CUDA-only branches, qsa.py:969-995 in the PR head.)

## Downstream sensitivity (`scripts/sens.py`, `dbg/sens_main.txt`, nightly image + PR head, HIP 0)
`qsa_sparse_paged_attention`, bf16, head_dim 256, 6 q heads / 1 kv head, 8 rows, selection width 19, same indices in different orders:
- real flaky case (graph order for row 2 vs tiled order): bitwise identical, max abs diff 0.
- 5 random column permutations of all rows: not bitwise identical, max abs diff 3.9e-3 to 7.8e-3 (one bf16 ulp at output magnitude ~1; max |out| = 1.47).
- repo tolerance (`test_qsa_sparse_paged_attention_matches_reference`): rtol=2e-2, atol=2e-2, so permutation noise is about 2.5x-5x below it.
Caveat: synthetic q/K/V (random), not model activations.
Containers: none left running.

# Addendum 3: confirmation for lead's re-request (no new GPU runs)
Items (a) and (b) were already done in Addendum 2 (sens.py / source reading); this addendum only adds what was missing.
- Nightly image source check: `csrc/sampler.cu` at commit 40941f2c4 (the nightly's `g40941f2c4`) is byte-identical to v0.32.0 and PR head (`src/sampler.nightly-40941f2c4.cu`, `cmp`). Order mechanism: sampler.cu:281 atomicAdd slot assignment (the :403 comment is short-row only, see Addendum 4). Order is NOT defined by the kernel contract.
- Does anything rely on order? Upstream `tests/kernels/test_top_k_per_row.py` (nightly, saved in `src/`) compares top-k results as sets/sorted (`indices.sort(1).values` at lines 505, 574, 780, 1136; set compare at ~line 180). The PR's other QSA tests (test_qsa_amd.py) compare logits or sorted indices (`torch.equal(indices.sort(1).values, expected)`); only the CUDA-graph test uses exact positional equality. No doc/comment promises sorted output.
- Downstream (a/b already): same set, graph vs tiled order of the failing row -> bitwise identical attention output; random permutations -> max abs diff 3.9e-3..7.8e-3 (1 bf16 ulp) vs repo tolerance 2e-2.
- Scope note: the sorted-test loop was run on HIP 1 (renderD136) by mistake (HIP 0 was assigned); no further GPU runs now.

# Addendum 4: corrections after review (no new runs; numbers recomputed from runs/*.log and dbg/order_*.txt)
## Failure pattern (replaces any i.i.d./random-race wording above)
- Fresh-process failure rate is about half: v0.32.0 55/100, nightly+PR 50/100, v0.33.0-rocm72 21/40.
- On nightly and rocm72 failures alternate with run number: nightly (100 runs) even-numbered 47/50 fail vs odd-numbered 3/50; rocm72 (40 runs) even 20/20 vs odd 1/20 (20 of 21 failures even). v0.32.0 shows no such pattern (even 29/50, odd 26/50).
- Within one process the order is mostly fixed (`dbg/order_run1.txt` v0.32.0, `dbg/order_main.txt` nightly, 500 repeats on identical inputs):
  - nightly: graph replay equal to tiled base in 497/500 (3 differ), eager rows 500/500 equal (1 distinct order), eager tiled 493/500 equal (7 differ, 2 distinct orders).
  - v0.32.0: eager rows kernel: always 1 order, but never equal to the tiled base (0/500); graph replay equal only 2/500 (2 distinct orders); eager tiled equal 4/500 (2 distinct orders). So there the base process order was the rare one; again few distinct orders per process.
  - selected SET identical 500/500 in all modes; sorted-compare test passes 100/100 (v0.32.0 and nightly).
- So the order is essentially fixed per process and differs between processes. The process-level source of that difference is NOT identified (open question; no mechanism claimed, e.g. no XCD/CU attribution; the even/odd alternation is unexplained).
## Which top-k variant runs in the failing case (by reading only, not measured)
`qsa_select_paged_tokens` on ROCm calls `ops.top_k_per_row_decode` (overlay/qsa.py:986; PR head qsa.py:995) with `logits` of width `columns = page_table.shape[1]*k_cache.shape[1]` (qsa.py:936; test: 24*16 = 384 if the logits buffer has that width; the width of the buffer returned by `qsa_mqa_paged` was not printed, so unmeasured), topK = block_topk = 4. In sampler.cu:876-877 `numColumns < kSortingAlgorithmThreshold (12288)` selects `topKPerRowDecode<512, false>`, the insertion-sort variant (the gfx950-only branches starting at :823 do not apply on gfx942). Any width < 12288 gives this variant. Not confirmed by a profiler or kernel-name trace: "likely, not measured".

# Addendum 5: nits
- Which diff goes where: ONLY `proposed/test_qsa_amd_sorted.diff` is meant for upstream (headers `a/tests/models/qwen4_exp/test_qsa_amd.py`; `patch -p1 --batch --dry-run` against the PR head file `main/test_qsa_amd.py` in a tests/models/qwen4_exp/ tree: applies cleanly, rc=0 above). `proposed/test_sorted.diff` (and `proposed/test_qsa_kernels.py`) are repo-local only, for overlay/test_qsa_kernels.py; its first line says so.
- What the sorted comparison checks: `sort(dim=1)` on both outputs compares, per row, the multiset of selected token indices (including the tail slots), which is what the downstream attention consumes. It does NOT check the order of indices, nor within-block order. Attention output was shown order-insensitive only up to about 1 bf16 ulp (Addendum 2, synthetic inputs).

# Addendum 6: search for the cause of the per-process alternation, not found (bisection, nightly image + PR head, HIP 0 only, stopped 08:37 per host clock; start not recorded)
Outcome: **the alternation itself is NOT explained ("not found")**. What was measured: (i) the order of the selected blocks is a per-launch hardware race (atomic arrival), (ii) the alternation seen earlier did not reproduce when re-tested now. Everything marked HYPOTHESIS is unmeasured.
All outputs in `exp/`, scripts in `scripts/` (fp.py, delay.py, ptr.py, bias.py, cold.py, arrive.hip/.py, exp_*.sh). Loops were <=12 fresh processes; no pytest 100-loops.

- E1a (`exp/E1a_single.txt`): `scripts/exp_run.sh single renderD128` = 10 fresh `python3 fp.py` in one container, host load 14-18 (e2e running). Order of row 2 blocks {620,1388} for tiled / rows / graph: no P F P F pattern; e.g. tiled = [620,1388],[1388,620],[1388,620],[620,1388],... and the three launch kinds disagree with each other inside one process in 4 of 10 processes. E1b (new container per run) and E1c (5 s sleeps) were NOT run (E1a already showed no alternation, so there was nothing to localise; skipped for load reasons).
- E2 (`exp/E2_k.txt`): k = 0..9 trivial kernels before the top-k call, fresh process each: the tiled order is [1388,620] for k=0,3,9 and [620,1388] for k=1,2,4,5,6,7,8; not a function of k mod 2 or mod 8 (HYPOTHESIS "dispatch phase" not supported).
- E5 (`exp/E5_delay.txt`, one process, 30 calls per gap): after gaps of 0 / 10 ms / 100 ms / 500 ms / 2 s the "620 first" outcome count is 2/30, 0/30, 0/30, 0/30, 1/12. Idle gap length does not change the order.
- E6 (`exp/E6_ptr.txt`): the logits/out pointers are identical across calls within a process (logits width 384, topK 4: confirms the insertion-sort launch `topKPerRowDecode<512,false>` is the one used, by shapes only, kernel name not traced) and the order of row 2 flips between consecutive identical calls in the same process (1 to 3 of 8 calls differ from the others in each of the 6 processes). So it is a per-launch race with a skewed probability, not fixed per process.
- E7 (`exp/E7_bias.txt`, 10 fresh processes x 200 calls): "620 first" fraction for tiled 1-19/200, rows 0-3/200, graph 0/200 in every process. Every process is biased the same way (mostly 1388 first); no process has the opposite bias. With this bias the pytest outcome "graph != tiled" would be rare, i.e. the 50% pytest failure rate is not reproduced by back-to-back calls.
- E8 (`exp/E8_cold.txt`): bias after evicting L2/MALL with a 2 GiB fill: tiled 1+3+2 of 120, graph 0/120 (same as back-to-back). After a 4096^3 matmul: tiled 5+4+3, graph 3+7+5 of 40 each (graph 0/120 back-to-back vs 15/120 after the matmul). So what ran immediately BEFORE the launch changes the order of the graph launch; cache coldness did not. (Measured; mechanism behind it is a HYPOTHESIS: CU/wave residency left by the previous kernel.)
- E10 (`exp/E10_arrive.txt`, `scripts/arrive.hip`): hipcc-built kernel with the same grid shape (8 blocks x 512 threads, load then `atomicAdd` on a shared counter), fresh process x6, 300 launches each, reading HW_REG_XCC_ID: block b always runs on XCC b (1 distinct mapping in 6x300 launches, identical in all processes); the wave arrival order inside a block takes 9-10 different permutations roughly uniformly in every process (wave 3 and 7 fixed, waves 0-2 and 4-6 shuffle). So the XCC placement is NOT the per-process variable, and the intra-block atomic arrival order is genuinely random per launch. This is the measured cause of why the kernel's order is undefined. It is a synthetic kernel, not the vLLM kernel, and it does not itself differ between processes.
- E9 (`exp/E9_phase.txt`, pytest alone, nightly+PR files, same harness): A 12 runs = `PFFFFFFPFFPF`; B (dummy GPU process before each pytest) = `PFFFFFFFPP`; C = `FFPPFFPFPP`. The P F P F alternation is NOT reproduced now (and inserting a dummy GPU process did not shift a phase), although it was clean in `runs/nightly-pr62589-first20` (`PFPF...` x20), `runs/v0.33.0-rocm72` (`PFPF...`), and most of `runs/nightly-pr62589`, with disturbances in the 100-run log at the times other work ran on the host. v0.32.0 never alternated.
- E4 (partial): host load average 14-18 during E1a (other agents' e2e job on HIP 2,3,6,7). `gpuagent` (pid 2095152, root, since Oct 16) and container `device-metrics-exporter-otto` (rocm/device-metrics-exporter:v1.4.1, logging "Unable to open queues directory for process N") are running on this host and the containers print `[queue-intercept] ... profiling requires system-memory InterceptQueue`. HYPOTHESIS (unmeasured, not testable without touching the production exporter, which was not touched): queue interception / metrics polling changes dispatch timing and could create the regular period seen earlier. Not tested: stopping the exporter (forbidden), per-run correlation with GPU activity on other GPUs.

Conclusion for the report's original question: the pass/fail of this test is decided by a hardware race on atomic arrival (measured: order differs between consecutive launches, and the intra-block arrival order is random in a minimal kernel); the earlier strict alternation across processes was a property of the host state at that time (regular cadence of solo runs) and could not be reproduced, so it has NO identified cause. The test-only sorted-compare fix does not depend on it.
