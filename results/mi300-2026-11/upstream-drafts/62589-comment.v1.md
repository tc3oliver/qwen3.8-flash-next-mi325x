DRAFT, not posted.

**MI300X (gfx942) check of #62589 at 404857c8**

Hardware: AMD Instinct MI300X, gfx942, 304 CUs. Host ROCm 7.0.2 / amdgpu 6.14.14; the container brings its own stack (nightly: torch 2.13.0+rocm10.0.0, triton 3.8.0). Image: `vllm/vllm-openai-rocm:nightly`, sha256:5800b3b4cd79defc1cbcc66adb83b3c49e574b49ca5c6ba556e17388515dc84f (0.33.0rc1.dev573+g40941f2c4). Stock = the image's own `amd/ops/qsa.py`; PR = the 404857c8 `ops/qsa.py` hunk applied to it. One GPU, logits width 65,600 (262144/4; visible columns are set by the context length), 4 heads, D128, top-k 2048, bf16, 5 warmup + 50 iterations, CUDA events.

`qsa_select_paged_tokens` GPU ms, median, stock -> PR @ 404857c8:

| ctx | rows/req | conc | stock | PR |
|---|---|---|---|---|
| 32K | 4 | 64 | 0.8937 | 0.1338 |
| 98K | 1 | 64 | 0.3456 | 0.2417 |
| 262K | 4 | 64 | 1.3318 | 0.5043 |
| 262K | 1 | 16 | 0.1824 | 0.1886 |

"rows/req = 4" means MTP verify rows.

262K 1-row c16 is about 1-3% slower in select (0.1824 -> 0.1886; a second 262K-only pass: 0.1871 -> 0.1895, ranges overlap, so within run-to-run range). The scoring kernel alone is +7-11% there (0.1173 -> 0.1300; 0.1227 -> 0.1315).

Top-k: selected block sets equal stock in 23 of 24 cases. The 24th (98K, 1 row, c64, row 55) differs by one block (stock picks 21072, PR picks 11651). Stock fp32 scores: 21072 = 1.0951964855194092 (0x3f8c2f66, rank 511), 11651 = 1.0951963663101196 (0x3f8c2f65, rank 512). In the PR both blocks score 0x3f8c2f66 (exact tie at the rank 511/512 boundary) and its top-k kept 11651. In stock the two scores are 1 ulp apart (no tie), so stock picks the higher one; in the PR's own scores they are exactly equal fp32, so the pick is decided by tie-break. A 1-ulp accumulation-order difference thus flips a boundary pick.

**Flaky test** `test_qsa_select_decode_rows_replays_in_cuda_graph` (exact equality of selected indices). Fresh process per run, `Mismatched elements: 8 / 152`, same row, two selected blocks swapped:
- nightly + 404857c8 files, the PR's own `tests/models/qwen4_exp/test_qsa_amd.py`: 50/100 fail (the reproducible one).
- v0.32.0 55/100 and v0.33.0-rocm72 21/40 fail with a downstream overlay test file (`test_qsa_kernels.py`, same test body) and overlay qsa.py/indexer_qsa.py. For contrast, the same test on MI325X: 1/40 on v0.33.0-rocm72 (the overlay author's README).
- The selected set is identical in 500/500 repeats; only the order differs. The kernel defines no order (`csrc/sampler.cu:281` `atomicAdd(&smemFoundTopKValues[0], 1)` is the known unordered write; file byte-identical in v0.32.0, nightly and the PR head, so the PR does not change it).
- Pattern: about 50% in fresh processes; on nightly and rocm72 failures alternate with run number (nightly 47/50 even-numbered runs fail vs 3/50 odd; rocm72 20/20 even vs 1/20 odd), while repeats inside one process rarely change the order. The process-level source of this is not identified, so I am not claiming a root cause.
- Comparing sorted indices passes 100/100 (v0.32.0 and nightly). Not a bug in the selected set; I would make the test order-insensitive. Test-only diff on request.

Caveats: synthetic inputs, one seed per case, one GPU; host load average 6-11 during timing (GPU use 0-2% at start/end only); the `qsa_indexer.py` hunk was not benchmarked; v0.33.0 (non-rocm72) not run.

Attachments to upload as a gist after the author approves: phase1-hip1/nightly-full.log, phase1-hip1/nightly-262k-pass2.log, phase1-hip1/row55.log, phase1-hip0/proposed/test_qsa_amd_sorted.diff, phase1-hip0/dbg/order_main.txt.
