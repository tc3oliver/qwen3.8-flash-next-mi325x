# Phase 1 / HIP 1: vllm#62589 tiled QSA indexer, MI300X (gfx942)

Device: HIP 1 = /dev/dri/renderD136 only (container sees it as HIP 0). No production GPU touched.
Image: vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905 (v0.32.0).
Script: bench_qsa.py (loads make_case/run_select/run_mqa from harness.py, which is a copy of upstream/vllm-62589-qsa-indexer-tiled/evidence/review3/harness.py (the helper script from this repo's own overlay tests, not code from the vLLM PR itself); 65,600 columns, 4 heads, D128, top-k 2048, bf16).
Method: 5 warmup, 50 iterations, each iteration = CUDA events around the call + sync. Values = median [min-max] ms.
"score" = qsa_mqa_paged (scoring kernel). "select" = qsa_select_paged_tokens (score + top-k + expand) = full selection.
Wall = perf_counter around call+sync. Raw: *-run*.log, load/GPU snapshots: *-pre*.txt / *-post*.txt (uptime + rocm-smi --showuse).
No cache-size reduction was needed (64 req x 164 pages fits easily).

## Runs
| id | stock file | PR file | log | noise |
|---|---|---|---|---|
| v032-run1 | v0.32.0 image qsa.py | upstream/.../qsa.py (OLDER PR revision, 1159 lines, SHA not recorded) | v032-run1.log | clean: load avg ~7, GPU[0] 0-1% before/after |
| v032-run2 | same | same (repeat, nontie diagnostics added) | v032-run2.log | clean: load avg 9, GPU[0] 0% |
| main-run1 | main@67791dd9 `amd/ops/qsa.py` (main/main_qsa.py) | PR head 404857c8 (main/pr_head_qsa.py) | main-run1.log | host load avg 13 (other agents), GPU[0] 0% before/after (sampled only at start/end); results consistent with v032-run1/2 within 3% |

Main/PR files were run inside the v0.32.0 image (runtime = v0.32.0 torch/triton/ops), not the nightly image: the local
`:nightly` tag is the stale 2026-09-03 image (vllm 0.28.1rc1), the new pull had not finished, so it was NOT used.
Main-vs-PR-head numbers agree with v032 runs within ~1-3% on every row despite the higher host load.
Commands: `./run_v032.sh` (run1/run2, edit log names), `./run_main_on_v032img.sh`.

## Results (select = full selection, GPU ms, v032-run2; main-run1 within 3%)
Full tables (all 24 cases, score + select + wall) are in the logs. Key rows:

| ctx | rows/req | conc | stock select | PR select | PR/stock |
|---|---|---|---|---|---|
| 32K | 1 | 1 | 0.0908 | 0.0937 | 1.03 (run1) |
| 32K | 1 | 64 | 0.2722 | 0.1241 | 0.46 (run1) |
| 32K | 4 (MTP) | 64 | 0.8967 | 0.1341 | 0.15 (run1) |
| 98K | 1 | 16 | 0.1462 | 0.1335 | 0.91 |
| 98K | 1 | 64 | 0.3510 | 0.2446 | 0.70 |
| 98K | 4 | 64 | 1.0353 | 0.2559 | 0.25 |
| 262K | 1 | 1 | 0.1078 | 0.1100 | 1.02 |
| 262K | 1 | 16 | 0.1848 | 0.1911 | 1.03 |
| 262K | 1 | 64 | 0.4792 | 0.4755 | 0.99 |
| 262K | 4 | 16 | 0.4109 | 0.2032 | 0.49 |
| 262K | 4 | 64 | 1.3255 | 0.5125 | 0.39 |

## 262K high-concurrency (item 2)
- 1 row/request, c16: stock 0.1848 [0.1802-0.1939] vs PR 0.1911 [0.1888-0.1963] (+3%, score kernel 0.1215 -> 0.1310, +8%).
  Select median PR/stock over 5 runs: v032-run1 1.039, v032-run2 1.034, main-run1 1.044, nightly-full 1.034, nightly-262k-pass2 1.013, i.e. 1-4% slower in all 5 runs; min-max ranges overlap in each run; no confidence interval computed. Scoring kernel +7-11%, same direction in every run.
- 1 row/request, c64: stock 0.4792 vs PR 0.4755 (-1%), no regression; score 0.4047 vs 0.4074 (flat).
- Also flat/slightly slower at 262K 1 row c4: select 0.1204 -> 0.1263 (+5%).
- No large regression found. With full-context requests everything is visible, so the early exit has nothing to skip; PR gains nothing at 1 row and pays a few us: the 1-row, c1 cases at 32K/98K/262K are consistently ~2-3 us (1.02-1.03x) slower (fixed overhead; nightly-full 0.0900/0.0930, 0.0891/0.0906, 0.1048/0.1067 ms).
- 262K with 4 rows/request (MTP): PR is 2.0-2.6x faster at c16 (0.41 -> 0.20) and c64 (1.33 -> 0.51).
- Not measured: 262K regression in end-to-end serving (out of scope, no model).

## Top-k agreement (item 1)
Compare block sets (indices//4) of stock vs PR per row. 1 of 24 cases (98K, 1 row, c64, row 55) shows a 1-block difference (1-ulp boundary flip, see addendum); 23 of 24 cases identical.
- 98K, 1 row, c64: 63/64 rows identical; row 55 differs by 2 blocks. Seen in all 4 runs that include this case: v032-run1 (63/64 identical; the row-55 diagnostic line was added only from run2), v032-run2, main-run1, nightly-full (nightly-262k-pass2 does not cover 98K). The stock logits of the two
  swapped blocks are equal to 6 decimals (1.095196) but not bit-equal, so this is a near-tie at the 2048 boundary (fp rounding of tl.dot vs
  stock accumulation), not an exact tie. My "tie" test required exact equality, so the log counts it as diff_nontie.
- All other 23 cases: all rows identical. Not verified: index-level order within a row (only sets compared).

## Open items / unverified
- Nightly image (`vllm/vllm-openai-rocm:nightly`) not benchmarked (pull unfinished at time of writing); runtime differences unverified.
- The upstream/.../qsa.py in the repo differs from PR head 404857c8 (1159 vs 1326 lines, rebased on newer main); both give the same numbers here.
- Single GPU, one seed per case, no cross-GPU or MI325X comparison. Other agents' jobs ran on other GPUs; GPU[0] was idle in every sample, but the sampling is only at start/end.
- No diffs to propose for overlay/ or upstream/.

## main nightly (added 2026-11-03)
Image: vllm/vllm-openai-rocm@sha256:5800b3b4cd79defc1cbcc66adb83b3c49e574b49ca5c6ba556e17388515dc84f (vllm 0.33.0rc1.dev573+g40941f2c4, torch 2.13+rocm10.0), renderD136 only, group-add video + numeric render gid.
Stock arm = the image's own `amd/ops/qsa.py` (nightly/stock_qsa.py, sha256 49720339c385...; 1137 lines, identical to main@67791dd9 file).
PR arm = #62589 head 404857c8 (state OPEN, unchanged): `patch` of the PR's amd/ops/qsa.py hunk onto the stock file applied cleanly (no offsets/fuzz reported,
nightly/patch.log), result nightly/pr_qsa.py sha256 88dad9056... == main/pr_head_qsa.py byte-identical. The PR's `amd/qsa_indexer.py` hunk was not applied (the microbench only calls ops/qsa.py).
Same bench_qsa.py (5 warmup, 50 it, CUDA events). Commands: `./run_nightly.sh full` and `./run_nightly.sh 262k-pass2 262k`. Logs: nightly-full.log, nightly-262k-pass2.log (+ -pre/-post.txt).
Noise: host load avg 6-11 (other agents), all GPU[0..7] use 0% at pre/post (post of pass2: GPU[0] 2%). No concurrent GPU job observed, but sampled only at start/end; not labeled noisy-GPU, host CPU load noted.

select GPU ms, median (PR/stock), full pass:
| ctx | rows | c1 | c16 | c64 |
|---|---|---|---|---|
| 32K | 1 | 0.0900/0.0930 (1.03) | 0.1296/0.1184 (0.91) | 0.2714/0.1247 (0.46) |
| 32K | 4 | 0.0905/0.0936 (1.03) | 0.2716/0.0913 (0.34) | 0.8937/0.1338 (0.15) |
| 98K | 1 | 0.0891/0.0906 (1.02) | 0.1463/0.1323 (0.90) | 0.3456/0.2417 (0.70) |
| 98K | 4 | 0.0968/0.0916 (0.95) | 0.3101/0.1218 (0.39) | 1.0448/0.2539 (0.24) |
| 262K | 1 | 0.1048/0.1067 (1.02) | 0.1824/0.1886 (1.03) | 0.4727/0.4730 (1.00) |
| 262K | 4 | 0.1186/0.1083 (0.91) | 0.4080/0.2018 (0.49) | 1.3318/0.5043 (0.38) |

262K 1-row c16 (the suspected regression): full pass 0.1824 [0.1779-0.1916] vs 0.1886 [0.1861-0.1965] = 1.03 (score kernel 0.1173 -> 0.1300, +11%);
262K-only pass 0.1871 [0.1787-0.2035] vs 0.1895 [0.1870-0.2097] = 1.01 (score 0.1227 -> 0.1315, +7%). c64: 1.00 in both passes.
=> Same statement as above: select median PR/stock 1.013-1.044 over 5 runs (1-4% slower, ranges overlap in each run, no CI computed); scoring kernel +7-11%, same direction in every run. Matches v0.32.0 (+3-4%).
Top-k: same as before, 23/24 cases all rows identical; 98K 1-row c64 row 55 differs by 2 blocks (1-ulp boundary flip, see addendum). Results on nightly match v0.32.0 within ~3% everywhere.

## Addendum: 98K, 1-row, c64, row 55 top-k difference (nightly image, no timing)
Script row55.py (log row55.log), same case (seed 64, ctx 98304, 64 reqs, row 55; 24,576 visible blocks, k = 512 blocks), nightly stock vs PR head.
- Selected sets differ by exactly one block: stock-only 21072, PR-only 11651.
- Exact fp32 scores (hex bits): block 21072 stock 1.0951964855194092 (0x3f8c2f66), PR same (0x3f8c2f66), 0 ulp.
  Block 11651 stock 1.0951963663101196 (0x3f8c2f65), PR 1.0951964855194092 (0x3f8c2f66): 1 ulp apart (PR is 1 ulp higher).
- Rank k boundary (evidence: row55.log lines 7-9; the last two lines of row55.log, rank lookup {11651: 511, 21072: 511}, are an artifact of the script (strict-greater count under ties) and are not evidence): stock sorted: rank 511 = blk 21072 (…4855), rank 512 = blk 11651 (…3663) -> stock picks 21072 because it is 1 ulp higher, no tie.
  PR sorted: ranks 511 and 512 are blk 11651 and 21072 with identical value 1.0951964855194092 -> exact tie in the PR's own scores; PR's top-k kept 11651 over 21072 (tie-break order).
- Logits row over visible columns: not bitwise equal (9,551 of 24,576 columns differ, max abs diff 3.6e-07 = ~1-3 ulp), as expected from a different accumulation order (tl.dot tiles vs per-row sum).
- (c) The tie rule is in my bench_qsa.py (not harness.py): it requires exact equality of the stock scores of the differing blocks. Those stock scores differ by 1 ulp, so it flagged nontie; it is stricter than the data's meaning (differences are accumulation-order noise).
Conclusion: "scores differ by 1 ulp (accumulation order), selection flips at the boundary" (plus a tie-break in the PR's own scores). Not a real discrepancy. Confidence high for this row (exact bits shown); only this one row/case was examined, and other rows' agreement was by set equality only.
