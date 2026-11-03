# overlay/

Two files that replace the stock ones in `vllm/vllm-openai-rocm` **v0.32.0 and v0.33.0** (the AMD QSA files are
byte-identical in both tags). Mount them read-only, no environment variables:

```
PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
-v $PWD/overlay/qsa.py:$PKG/ops/qsa.py:ro
-v $PWD/overlay/indexer_qsa.py:$PKG/indexer_qsa.py:ro
```

`deploy/compose.yaml` already mounts both. **Only valid for those two tags**: it replaces whole files, and the file names
differ on newer main (`qsa_indexer.py`). Check the stock hashes before use:

| file | stock sha256 (v0.32.0 = v0.33.0) | overlay sha256 |
|---|---|---|
| `ops/qsa.py` | `7a7c8501e0b92a4236e382c6ef494670d684ee1dfeb2d673f942a4f38fccf69b` | `f59670bf046ad58b051b74d80ba23883477b845824278704eff6ee67116ac524` |
| `indexer_qsa.py` | `ca8ede9f27c933b85a3a3e45e8e1b4f7693c8f0271ff3cab3fa6fb64de8c2476` | `dbb72baa9b578553ecb58e0cb334e6f9f96d172f216d0ed4267e0ce536caa46a` |

## Content

* `qsa.py`: the stock file plus
  * [vllm-project/vllm#62589](https://github.com/vllm-project/vllm/pull/62589) (branch head `404857c80`, 2026-10-30): tiled QSA indexer scoring
    (16 rows × 64 keys per program, `tl.dot`), per-request skip of column blocks past the visible range, fp16 dot for fp16 inputs, and the
    gfx942 per-row kernel for single-row decode;
  * [vllm-project/vllm#62590](https://github.com/vllm-project/vllm/pull/62590): sparse-GQA prefill BLOCK_N=32 on gfx942 when BLOCK_M >= 16.
  Both PR diffs were applied to the v0.32.0 file with `patch` (line offsets only, no conflicts). The PRs themselves are written against main.
* `indexer_qsa.py`: the stock file plus the 3 lines that pass `single_row_decode=(num_prefills == 0 and decode_query_len == 1)` to
  `qsa_select_paged_tokens`. **This is a hand backport**: #62589 makes this change in `qsa_indexer.py`, a file that does not exist in
  v0.32.0 or v0.33.0 (the module is still called `indexer_qsa.py` there). Without it the per-row kernel is never called. It is not part of any PR and
  was not run through the PR's own tests, only through the kernel tests below.

## The per-row decode path is off with MTP

`single_row_decode` needs one query row per request. With `--speculative-config` MTP and 3 draft tokens, `decode_query_len` is 4, so the
shipped deployment keeps using the tiled kernel for decode and gets no effect from the per-row path. It only applies without speculative decoding.
No end-to-end measurement of the per-row path is included here.

## Test

`test_qsa_kernels.py`: the four kernel-vs-reference tests from the #62589 test file that need only `ops/qsa.py` (19 cases: bf16/fp16/fp8 inputs,
visible-column masking, decode rows 1/5/32/33 including the 32→tiled switch, NaN in unwritten logits, CUDA-graph replay). The PR's full test file
imports modules that do not exist in the releases. With the overlay (one MI325X) the whole file passed 5 of 5 runs on each of v0.32.0, v0.33.0 and
v0.33.0-rocm72. **One test is flaky when run alone:** `test_qsa_select_decode_rows_replays_in_cuda_graph` (exact equality of the selected indices between the
per-row kernel replayed in a CUDA graph and the tiled kernel) failed 4/40 on v0.32.0, 2/40 on v0.33.0 and 1/40 on v0.33.0-rocm72, always with the same
mismatch (8 of 152 elements). Cause not found; see [results/overlay-refresh-2026-11/kernel-test-repeats.txt](../results/overlay-refresh-2026-11/kernel-test-repeats.txt).
The path is inactive with MTP 3. Run command:

```
docker run --rm --device /dev/kfd --device /dev/dri/renderD<one GPU> --group-add video --ipc host --security-opt seccomp=unconfined \
  -v $PWD/overlay/qsa.py:$PKG/ops/qsa.py:ro -v $PWD/overlay/indexer_qsa.py:$PKG/indexer_qsa.py:ro \
  -v $PWD/overlay/test_qsa_kernels.py:/t/test_qsa_kernels.py:ro --entrypoint python3 \
  vllm/vllm-openai-rocm:v0.32.0 -m pytest -q /t/test_qsa_kernels.py
```

## Results

End-to-end on 2 × TP2 settings of the shipped compose: [results/overlay-refresh-2026-11/](../results/overlay-refresh-2026-11/) and the
[README](../README.md).

## history/

`history/qsa.2026-10-20-d02.py`: the previous overlay (#62589 first revision + #62590, `ops/qsa.py` only, sha256 `6c84b3f9…`). Run d02 and the
results published before 2026-11-03 used it.
