[ROCm][Perf][Qwen4Exp] Tile QSA indexer scoring and skip invisible columns

<!-- markdownlint-disable -->

## Overview

`_qsa_mqa_paged_kernel` scores every row against the full page-table width (max_model_len / 4 = 65,600 columns), regardless of how many are visible. On MI325X it is 43% of GPU time in a 32K prefill. This PR skips columns past each request's visible range and scores 16 rows x 64 keys per program with `tl.dot`.

## Claims

- `qsa_select_paged_tokens`, MI325X: 32K prefill 90.7 -> 4.58 ms, decode 64x4 0.93 -> 0.26 ms. Decode without MTP is 1.05-3.2x faster at 16+ requests; a single request is 2-7% slower (+1-4 us).
- Qwen3.8-Flash-Next-FP8, TP2: 8x32K prefill 11.5K -> 21.0K tok/s (+83%), decode at 64 concurrent 2,167 -> 3,023 tok/s (+39%). GSM8K-200 0.970 -> 0.965 (within noise for n=200).
- Logits past `visible_blocks[row]` are no longer written. The only consumer, `top_k_per_row_decode`, reads `[0, visible_blocks[row])`.

## Validation

    pytest tests/models/qwen4_exp/test_qsa_amd.py   # MI325X: 18 passed

The new test compares logits on visible columns against a torch reference (bf16 and fp16; 3 requests, a row block spanning requests, partial page, empty rows). It fails if the dot runs in bf16 for fp16 inputs, and under three mutations: scoring only the first request in a block, exiting on the block's min visible count, dropping the partial last column block.

Selected indices match main except at exact score ties (0.0 after ReLU): identical rows 0.9965-1.0 across prefill, mixed and decode shapes, index overlap 1.00000. Feeding NaN/inf past `visible_blocks` into the ROCm `top_k_per_row_decode` leaves selection unchanged.

<details>
<summary>Microbenchmarks (MI325X, 65,600 columns, top-k 2048)</summary>

4 heads, head_dim 128, compress ratio 4, page size 400. Synchronized wall-clock (prefill, x4 decode) or CUDA events (x1 decode) after warmup.

| case | main (ms) | this PR (ms) |
|---|---:|---:|
| prefill 32K | 90.7 | 4.58 |
| 16K chunk at 96K | 48.4 | 4.42 |
| decode 64 x 4 | 0.925 | 0.261 |
| decode 1 x 4 | 0.065 | 0.067 |
| decode 1 x 1, 8K-200K | 0.0616 | 0.0659 |
| decode 64 x 1, 8K-200K | 0.265 | 0.172 |
| decode 128 x 1, 8K-200K | 0.530 | 0.344 |
| decode 64 x 1, one 250K among 4K | 0.226 | 0.099 |
| decode 128 x 1, one 250K among 4K | 0.411 | 0.129 |

Early exit plus a shared key load on the old one-row kernel gives 14.6 ms for prefill 32K. Tile sweep over BLOCK_M 16/32/64, BLOCK_N 32/64/128, 4/8 warps: 16 x 64, 4 warps was fastest everywhere.
</details>

<details>
<summary>End-to-end (TP2 + EP, 2x MI325X, MTP 3)</summary>

vLLM v0.32.0 image with this `qsa.py` mounted; v0.32.0's `qsa.py` matches main before this PR. Noise ±3-5%.

| metric | main | this PR |
|---|---:|---:|
| prefill 1 x 32K (tok/s) | 11,726 | 19,762 |
| prefill 8 x 32K (tok/s) | 11,486 | 20,987 |
| decode c64 (tok/s) | 2,167 | 3,023 |
| 16 agents, 32K context: cold TTFT p50 (s) | 30.0 | 17.0 |
| 8 agents, 96K context: wall (s) | 81.6 | 50.4 |
| GSM8K-200 | 0.970 | 0.965 |
</details>

## Details

- Rows are request-major, so a 16-row block usually shares one page table and loads the key tile once for all rows and heads. A block that spans requests loops over them; each request skips column blocks past its own visible range.
- The dot runs in fp16 for fp16 inputs (#60650) and bf16 otherwise, which is exact for the fp8 cache in #61529. fp32 inputs, which no caller passes, would also use bf16.
- The NVIDIA indexer in `nvidia/ops/qsa_indexer.py` is not reusable here: it needs split prefill/decode metadata and the CUDA-only top-k.
- Tested on gfx942 only. MI355 CI runs the test; gfx950 perf is unmeasured.

Not a duplicate: open PRs on this file (#57969, #59437, #60650, #61529) don't touch indexer scoring.
