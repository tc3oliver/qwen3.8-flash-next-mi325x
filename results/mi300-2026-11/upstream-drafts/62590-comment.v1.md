DRAFT, not posted.

**MI300X (gfx942) check of #62590 at f87fca37**

Hardware: AMD Instinct MI300X, gfx942, 304 CUs; the code sees `gfx942:sramecc+:xnack-`, `on_gfx942()=True`. Host ROCm 7.0.2 / amdgpu 6.14.14; container stacks differ: nightly torch 2.13.0+rocm10.0.0 / triton 3.8.0; v0.33.0-rocm72 torch 2.13, HIP 7.2.53211. Images: `vllm/vllm-openai-rocm:nightly` sha256:5800b3b4cd79defc1cbcc66adb83b3c49e574b49ca5c6ba556e17388515dc84f (0.33.0rc1.dev573; the PR's 7-line hunk applied to the image's `ops/qsa.py` with `patch`, offset 14) and `:v0.33.0-rocm72` sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182 (stock + hunk is byte-identical to the PR head file). One GPU, the PR's own `bench_gqa.py`, stock vs stock+hunk.

Dispatch: TP1 (24q/2kv) and TP2 (12q/1kv), BLOCK_M=16, use BLOCK_N=64 up to 512 programs and BLOCK_N=32 above (boundary seen at 512/514 and 512/513 programs). TP4 (6q/1kv), BLOCK_M=8, stays at BLOCK_N=64 at 512, 513 and 1024 rows.

Speedup, nightly (rocm72 within about 1%):

| shape | rows | stock ms | PR ms | speedup |
|---|---|---|---|---|
| TP2 | 520 | 0.409 | 0.274 | 1.49x |
| TP2 | 1024 | 0.787 | 0.359 | 2.19x |
| TP2 | 4096 | 2.666 | 1.373 | 1.94x |
| TP1 | 260 | 0.413 | 0.277 | 1.49x |
| TP1 | 1024 | 1.538 | 0.727 | 2.11x |
| TP1 | 4096 | 5.350 | 2.794 | 1.92x |
| TP4 | 520 | 0.365 | 0.355 | 1.03x |
| TP4 | 4096 | 2.215 | 2.205 | 1.00x |

Max abs diff vs stock 4.9e-4 for TP1/TP2, 0 for TP4.
Correctness vs an fp32 torch reference: 9 shapes (TP1/TP2/TP4, rows 256/257/1024 and 512/513/1024) x (bf16, fp8-quantised values stored as bf16; the function asserts bf16) x (default dispatch, forced BLOCK_N=64). On the BLOCK_N=32 path the max diff is 3.8e-4 (bf16 TP2 513 rows); the overall max is 5.07e-4 (fp8-quantised TP4 512 rows, BLOCK_N=64, not touched by the PR). Everything is within the 2e-2 tolerance used.

The flaky CUDA-graph test seen in the QSA kernel tests on MI300X is unrelated to this PR; see the #62589 comment.

Caveats: no native fp8 path exists for this kernel; synthetic activations, single seed; host load average 7.4-22 during timing while GPU use was 0-2%; v0.33.0 (non-rocm72) not run; the PR's TP4 BLOCK_N=32 slowdown (5x) was not re-measured; PR base is 4e304b4d, the PR branch itself was not built on nightly.

Attachments to upload as a gist after the author approves: phase1-hip3/raw/dispatch_ref_v0.32.0.txt, phase1-hip3/raw/dispatch_ref_nightly+pr62590.txt, phase1-hip3/raw/bench_nightly_run1.txt, phase1-hip3/raw/bench_v0.33.0-rocm72_run1.txt, phase1-hip3/raw/bench_v0.33.0-rocm72_run2.txt.
