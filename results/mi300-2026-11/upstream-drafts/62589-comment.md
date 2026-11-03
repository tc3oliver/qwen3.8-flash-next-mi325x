<!-- POSTED 2026-11-03 (see comment-6599007249 / 6599007801). Original draft note: Re-check PR head (404857c8) and state before posting. -->
**MI300X (gfx942) numbers for #62589 @ 404857c8**

Setup: MI300X, 304 CUs, one GPU. `vllm/vllm-openai-rocm:nightly` (sha256:5800b3b4cd79dc…c84f, 0.33.0rc1.dev573; torch 2.13+rocm10.0, triton 3.8.0; host ROCm 7.0.2). Stock is the image's `amd/ops/qsa.py`; PR is the 404857c8 `ops/qsa.py` hunk applied on top. Logits width 65,600, 4 heads, D128, top-k 2048, bf16, 5 warmup + 50 iterations, CUDA events.

`qsa_select_paged_tokens`, GPU ms (median), stock -> PR:

| ctx | rows/req | conc | stock | PR |
|---|---|---|---|---|
| 32K | 4 | 64 | 0.8937 | 0.1338 |
| 98K | 1 | 64 | 0.3456 | 0.2417 |
| 262K | 4 | 64 | 1.3318 | 0.5043 |
| 262K | 1 | 16 | 0.1824 | 0.1886 |

rows/req = 4 is MTP verify.

262K, 1 row, c16 is 1-3.4% slower in select (0.1824 -> 0.1886, range 0.1779-0.1916 vs 0.1861-0.1965; second pass 0.1871 -> 0.1895, range 0.1787-0.2035 vs 0.1870-0.2097). The ranges overlap, so I can't call it a stable regression. The scoring kernel alone is +7-11% there.

Top-k: block sets match stock in 23 of 24 cases. The odd one is 98K, 1 row, c64, row 55, one block differs. At the rank 511/512 boundary, stock's two fp32 scores are 1 ulp apart (0x3f8c2f66 vs 0x3f8c2f65); with the PR they are exactly equal, so tie-break decides. A 1-ulp accumulation-order difference flips a boundary pick.

**Flaky test.** `test_qsa_select_decode_rows_replays_in_cuda_graph` asserts exact equality of the selected indices. On MI300X it fails in about half of fresh-process runs (nightly + 404857c8, the PR's own test file: 50/100, always `Mismatched elements: 8 / 152`, same row, two selected blocks swapped). Over 500 repeated launches on the failing inputs the selected set is identical every time; only the order changes. `top_k_per_row_decode` doesn't define an order: [`csrc/sampler.cu#L281`](https://github.com/vllm-project/vllm/blob/404857c80123438104534eec919cb95900f5562e/csrc/sampler.cu#L281) hands out output slots with `atomicAdd(&smemFoundTopKValues[0], 1)`, so the slot is thread arrival order (the PR doesn't touch this file). Sorting before the compare passes 100/100. Suggested change:

```diff
-    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
+    # top-k emits the selected blocks in unspecified order, so compare
+    # the selection as a set per row.
+    torch.testing.assert_close(
+        captured.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0
+    )
```

Caveats: synthetic inputs, one seed per case, one GPU, host load 6-11 while timing, `qsa_indexer.py` hunk not benchmarked.
