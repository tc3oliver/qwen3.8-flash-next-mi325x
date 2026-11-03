<!-- POSTED 2026-11-03. Original draft note: Re-check PR head (f87fca37) and state before posting. -->
**MI300X (gfx942) numbers for #62590 @ f87fca37**

Setup: MI300X, 304 CUs, one GPU (`on_gfx942()` is True). Images: `vllm/vllm-openai-rocm:nightly` sha256:5800b3b4cd79dc…c84f (0.33.0rc1.dev573, torch 2.13+rocm10.0, triton 3.8.0; the 7-line hunk applied to `ops/qsa.py` with `patch`, offset 14) and `:v0.33.0-rocm72` sha256:4fe7bc121cbe2f…a182 (torch 2.13, HIP 7.2.53211). Host ROCm 7.0.2. The PR's `bench_gqa.py`, stock vs stock+hunk.

Dispatch: TP1 (24q/2kv) and TP2 (12q/1kv) have BLOCK_M=16 and use BLOCK_N=64 up to 512 programs, 32 above. TP4 (6q/1kv, BLOCK_M=8) stays at 64 at 512, 513 and 1024 rows.

Speedup on nightly (rocm72 within ~1%):

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

Max abs diff vs stock: 4.9e-4 for TP1/TP2, 0 for TP4. Against an fp32 torch reference I ran 9 shapes x (bf16, fp8-quantised values stored as bf16) x (default dispatch, forced BLOCK_N=64). The BLOCK_N=32 path is off by at most 3.8e-4; the overall max is 5.07e-4 (TP4, which the PR doesn't touch). All within the 2e-2 tolerance.

The flaky CUDA-graph test I hit in the QSA tests on MI300X is unrelated to this PR; see the #62589 comment.

Caveats: the kernel has no native fp8 path (it asserts bf16), synthetic activations, single seed, host load 7-22 while timing (GPU use 0-2%), the PR's 5x TP4 slowdown at BLOCK_N=32 not re-measured, the PR branch itself not built on nightly.
