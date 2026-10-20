[ROCm][Perf][Qwen4Exp] Tile QSA indexer scoring and skip invisible columns

<!-- markdownlint-disable -->

## Overview

The ROCm QSA indexer scoring kernel (`_qsa_mqa_paged_kernel`) scored every row against the full page-table capacity (max_model_len / compress_ratio columns). In a 32K prefill on MI325X it took 43% of GPU time. It now skips column blocks past each row's visible range and scores 16 rows x 64 keys per program with `tl.dot`.

## Claims

- `qsa_select_paged_tokens` on MI325X: 32K prefill 91.1 -> 4.55 ms (20x), 16K chunk at 96K context 48.7 -> 4.37 ms (11x), decode 64x4 0.93 -> 0.33 ms (2.8x).
- End-to-end, Qwen3.8-Flash-Next-FP8 at TP2 on 2x MI325X: 8 x 32K prefill 11,486 -> 19,355 tok/s (+69%), decode at 64 concurrent 2,167 -> 2,997 tok/s (+38%), 32K-context agent cold TTFT p50 30.0 -> 16.9 s.
- No accuracy change: GSM8K-200 0.970 -> 0.975. Visible logits differ from main by at most 6e-7 (fp32 summation order), and selection differs only at exact ties.
- `qsa_mqa_paged` no longer writes logits past `visible_blocks[row]`. Its only caller passes `visible_blocks` as the row length to `top_k_per_row_decode`, which reads `[0, rowEnd)` on every path, including the gfx950 k=512 path.

## Validation

**Unit test.** New `test_qsa_mqa_paged_matches_reference_on_visible_columns` covers 3 requests (a prefill chunk, a decode request, and rows with nothing visible), with one row block spanning all three, a partial page, and several column blocks. It compares against a torch reference and fails under each of three mutations:
- scoring only the first request of a row block;
- exiting on the block's minimum visible count instead of the maximum;
- dropping the partial last column block.

The test checks logits only. The top-k bound comes from reading `csrc/sampler.cu`; the ROCm selection test mocks top-k.

    pytest tests/models/qwen4_exp/test_qsa_amd.py   # MI325X (gfx942): 17 passed

**Microbenchmark**, `qsa_select_paged_tokens` on one MI325X. Indexer shapes: 4 heads, head_dim 128, compress ratio 4, page size 400, 65,600 columns (max_model_len 262,144 / 4), top-k 2048. Synchronized wall-clock, 5 (prefill) or 50 (decode) iterations after warmup. The middle column is an ablation from an earlier run: early exit plus one key load for all heads, on the original one-row-per-program kernel with BLOCK_N 128. main varied by less than 1% between runs.

| case | rows | main (ms) | early exit (ms) | this PR (ms) | speedup |
|---|---:|---:|---:|---:|---:|
| prefill 32K | 32,000 | 91.1 | 14.6 | 4.55 | 20.0x |
| 16K chunk at 96K context | 16,384 | 48.7 | 20.4 | 4.37 | 11.1x |
| decode 64 req x 4 tok | 256 | 0.929 | 0.533 | 0.327 | 2.8x |
| decode 16 req x 4 tok | 64 | 0.226 | 0.090 | 0.070 | 3.2x |
| decode 1 req x 4 tok | 4 | 0.064 | 0.063 | 0.065 | 1.0x |

Tile sweep: BLOCK_M 16/32/64 x BLOCK_N 32/64/128 x 4/8 warps. 16 x 64 with 4 warps was fastest on every case.

**Selection check** at real shapes (65,600 columns, top-k 2048), comparing the selected index sets from main and this kernel:

| case | rows | identical rows | index overlap (first 256 rows) |
|---|---:|---:|---:|
| prefill 32K | 32,000 | 0.9983 | 1.00000 |
| 16K chunk at 96K | 16,384 | 0.9999 | 1.00000 |
| mixed (4 requests, prefill + decode) | 13,005 | 0.9965 | 1.00000 |
| decode 64 x 4 | 256 | 1.0000 | 1.00000 |

The differing indices are exact score ties (0.0 after ReLU); main's top-k is not order-stable under ties either.

**End-to-end.** Qwen3.8-Flash-Next-FP8, TP2 + EP on 2x MI325X: MTP 3, max-num-batched-tokens 32768, max-num-seqs 64. vLLM v0.32.0 image with this PR's `qsa.py` mounted (v0.32.0's `qsa.py` is identical to main's). Run-to-run noise is ±3-5%.

| metric | main | this PR |
|---|---:|---:|
| prefill 1 x 32K (tok/s) | 11,726 | 19,824 |
| prefill 8 x 32K (tok/s) | 11,486 | 19,355 |
| decode c64 (tok/s) | 2,167 | 2,997 |
| 16 agents, 32K context: cold TTFT p50 (s) | 30.0 | 16.9 |
| 8 agents, 96K context: wall (s) | 81.6 | 49.8 |
| GSM8K-200 | 0.970 | 0.975 |

## Details

- Rows are request-major, so a 16-row block almost always shares one page table. The [64, D] key tile is gathered once per block, and each head is a [16, D] x [D, 64] `tl.dot` with fp32 accumulation. A block that spans requests loops over the requests it contains.
- The block exits when its first column is at or past the largest visible count among its rows.
- Both `tl.dot` operands are cast to bf16: a no-op for today's bf16 q and cache, and exact for the fp8 e4m3 q and cache from #61529 (checked on gfx942: compiles, matches the bf16 run exactly). An fp32 or fp16 query would lose precision versus main's fp32 FMA; the model uses neither today (#60650 would add fp16).
- Not reusing the NVIDIA indexer (`nvidia/ops/qsa_indexer.py`): it needs split prefill/decode metadata (request-major decode rows, `query_start_loc`, host-side `max_seq_len`) and the CUDA-only persistent/cooperative top-k. The AMD path runs mixed batches with unified per-token metadata and `top_k_per_row_decode`.
- Validated on MI325X (gfx942) only. The optional MI355 CI step runs the new test, but gfx950 performance is unmeasured.

Not a duplicate: I searched open PRs on 2026-10-20 ("qsa rocm", "qwen4_exp qsa", "qsa indexer", "_qsa_mqa_paged_kernel", "qsa_sparse_paged_attention", "qwen4_exp amd", "QSA gfx942"). #59437 adds an opt-in AITER path in a new file. #57969, #60650 and #61529 touch only sparse GQA in this file.
