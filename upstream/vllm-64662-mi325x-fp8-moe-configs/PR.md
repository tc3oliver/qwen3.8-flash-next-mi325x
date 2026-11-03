<!-- markdownlint-disable -->

## Overview

Adds tuned Triton fused-MoE configs for Qwen3.8-Flash-Next-FP8 (E=512, hidden 2560, MoE intermediate 640) on MI325X at TP2 and TP4.

At these TP sizes `refine_fp8_moe_block_shape` refines the 128x128 FP8 weight blocks to 64x64 (N=320) and 32x32 (N=160). MI325X has no configs for these shapes, so the Triton kernel falls back to `get_default_config()`.

| TP | file |
|---|---|
| 2 | `E=512,N=320,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[64,64].json` |
| 4 | `E=512,N=160,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[32,32].json` |

## Claims

- TP4 (main nightly, 2026-10-29): c64 output throughput +20.4%, TPOT -20.2%, prefill throughput +15.7%.
- TP2 (v0.32.0): c64 output throughput +4.9%, prefill throughput +5.1%.
- The configs are only used when the Triton FP8 MoE backend is selected and the layer has N=160 or N=320 with the refined block shape on MI325X. They do not apply with expert parallelism, where the local expert count changes `E`.

## Validation

### End to end

```bash
vllm serve Qwen/Qwen3.8-Flash-Next-FP8 --tensor-parallel-size 4 \
  --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.93 --enable-chunked-prefill --enable-prefix-caching \
  --async-scheduling --engram-config '{"cpu_offload":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

TP2 used the same command with `--tensor-parallel-size 2`. The baseline is the same server without the new file; the tuned run passes it through `VLLM_TUNED_CONFIG_FOLDER`. `VLLM_ROCM_USE_AITER` was unset; both arms set `HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 TORCH_BLAS_PREFER_HIPBLASLT=1 SAFETENSORS_FAST_GPU=1 OMP_NUM_THREADS=8`. Logs showed `Using TRITON Fp8 MoE backend`, the refined block shape, and `Using default MoE config` (baseline) or `Using configuration from .../E=512,N=...json` (tuned).

Client: a local streaming-chat script (can share on request) with `ignore_eos`, `temperature=0.6`, `seed=1` and a unique prefix per prompt (no prefix-cache hits). `cN`: N concurrent requests, ~1K-token prompts, 512 output tokens (c1 was run at TP2 only). `prefill`: 8 concurrent 32K-token prompts, 1 output token. Medians [min-max] of 6 warm passes per arm (2 server boots x 3 passes).

TP4, `vllm/vllm-openai-rocm:nightly-3966924d` (main, 2026-10-29), boots alternated baseline/tuned:

| workload | metric | baseline | tuned | change |
|---|---|---|---|---|
| c64 | output tok/s | 2126 [2033-2180] | 2560 [2440-2600] | +20.4% |
| c64 | TPOT p50 (ms) | 17.55 [17.1-18.7] | 14.00 [13.8-14.2] | -20.2% |
| c16 | output tok/s | 1416 [1400-1462] | 1646 [1472-1679] | +16.2% |
| prefill | prefill tok/s | 11439 [11328-11528] | 13238 [12955-13276] | +15.7% |
| prefill | TTFT p50 (s) | 19.89 [19.74-20.16] | 17.29 [17.11-17.51] | -13.1% |

TP2, `vllm/vllm-openai-rocm:v0.32.0`, both arms in parallel on separate GPU pairs, swapped between boots:

| workload | metric | baseline | tuned | change |
|---|---|---|---|---|
| c64 | output tok/s | 1981 [1935-2021] | 2077 [2028-2096] | +4.9% |
| prefill | prefill tok/s | 11048 [10869-11091] | 11614 [11519-11689] | +5.1% |

TP2 c1 and c16 were within noise. The v0.32.0 run used an earlier copy of the TP2 file whose BLOCK_SIZE_K values above 64 are clamped to 64 by the kernel, so it runs the same configs.

Accuracy, GSM8K (n=200, zero-shot, greedy, thinking off), baseline and tuned under the same setup: TP4 nightly 0.970 / 0.970, TP2 v0.32.0 0.975 / 0.970.

### Lookup, correctness, kernel

On one MI325X with the nightly image above and the two files installed:
- Lookup (using the `fused_moe.py` lookup code from main 335dd77): both files resolve and every M picks the nearest key. EP local expert counts (E=256/128), bf16, unrefined `[128,128]`, other N, and other device names fall back to the default config.
- `fused_experts` output with the files is bit-identical to the default config at 17 M values from 1 to 32768 (all entries keep the default BLOCK_SIZE_K).
- Kernel time vs `get_default_config()` (CUDA graph) at every key and the geometric midpoint between keys: 1.01x-1.54x at N=320, 1.36x-2.32x at N=160. No point is slower.
- `pre-commit run --files <2 files>` passes.

CI has no MI325X runner; the checks above were run locally.

## Details

- Not a duplicate: main has no E=512 configs for any AMD GPU and no open PR adds them. #61688 (opt-in runtime autotune) skips shapes with bundled configs.
- Related PRs: #62092 (AITER on by default for CDNA3+) leaves TP2/TP4 on Triton because AITER FP8 MoE does not take refined block scales; #60839 makes `VLLM_ROCM_USE_AITER=1` (without `VLLM_ROCM_USE_AITER_MOE=1`) fall back to Triton with the refined blocks at TP2/TP4 instead of raising, so these files would apply there too; #58152 (auto EP for this model) would bypass them.
- Tuning: `benchmark_moe.py` on Triton 3.8.0 with local patches for the gfx942 FP8 dtype and refined block shapes (the latter is #63484). BLOCK_SIZE_K is capped at the quant block size, which the kernel enforces anyway.
- Cold start: new kernel variants compile on first use; at TP4 the first c16 pass after boot had 0.81 s and 1.05 s higher TTFT in the two tuned boots. #55268 adds Triton fused-MoE warmup.
- AI assistance was used to write the tuning and benchmark scripts, run the benchmarks, and draft this description.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
