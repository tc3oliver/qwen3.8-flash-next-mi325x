[Config] Qwen3.8-Flash-Next FP8 fused-MoE tuned configs for gfx942 (MI325X, 304 CUs)

## Motivation

No fused-MoE rows exist on main for Qwen3.8-Flash-Next-FP8 on gfx942 with `cu_num=304` (MI325X). Routed experts: FP8 `per_1x128` block scale, model_dim 2560, inter_dim 640, top-10, SiLU, g1u1. Naming follows the open gfx950 PR #5870. #5379 (MI308X, `cu_num=80`) has no overlapping keys.

## Changes

- `a8w8_blockscale_tuned_fmoe_qwen3_8_flash_next_gfx942.csv`: 8 rows, E=512 (TP1: M=4/16384/32768) and E=256 (TP2 + EP: M=2/4/1024/8192/16384). Unmodified tuner output. All are asm 1-stage `fmoe_bf16_blockscaleFp8_g1u1_{vs,novs}_silu_1tg_32x128`, err 0.0%.
- `a8w8_blockscale_untuned_fmoe_qwen3_8_flash_next_gfx942.csv`: M=1..32768 for both E.

Only shapes at least 3% faster than the default were written (`--compare --update_improved`). Every other shape was within 3% of the default, except M=1, 8, 16, 32, where the best candidate was 12-22% slower. `op_tests/tuning_tests/test_config_shape_collision.py` passes with the new file.

## Performance

MI325X, aiter 0.1.24.post1 (vLLM v0.32.0 ROCm image). Times are from the tuner's `--compare` report, measured on the fused_moe path, in us. The CSV `us` column is kernel time from the sweep, so it does not match this table.

| E | tokens | before | after | delta |
|---|---|---|---|---|
| 512 | 4 | 67.53 | 63.34 | -6.2% |
| 512 | 16384 | 6937.81 | 6420.35 | -7.5% |
| 512 | 32768 | 11224.80 | 9909.30 | -11.7% |
| 256 | 2 | 51.58 | 48.71 | -5.6% |
| 256 | 4 | 71.11 | 61.90 | -13.0% |
| 256 | 1024 | 621.04 | 598.25 | -3.7% |
| 256 | 8192 | 3448.78 | 3201.96 | -7.2% |
| 256 | 16384 | 5198.83 | 4800.71 | -7.7% |

End-to-end (vLLM 0.32.0, TP1, MTP 3, 2 runs): 8x32K cold prefill +0.8% / +0.7%. Decode at c16 and c64 (64 and 256 tokens/step) hits none of the new rows. c1 (4 tokens/step) hits the M=4 row: TPOT 4.0 ms without, 4.2 / 4.0 ms with. GSM8K-200: 0.98 without, 0.965 / 0.97 with (SE ~1.2 pt).

## Reproduce

```
# E=256 (EP2), 2 GPUs
TUNE_ONLY= python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i untuned_fmoe_qwen38_ep2.csv -o tuned_fmoe_qwen38_ep2.csv \
  --mp 2 --compare --update_improved --min_improvement_pct 3

# E=512 (TP1), 1 GPU
TUNE_ONLY= python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i untuned_fmoe_qwen38_tp1.csv -o tuned_fmoe_qwen38_tp1.csv \
  --mp 1 --timeout 300 --compare --update_improved --min_improvement_pct 3
```

`TUNE_ONLY=` is needed until #6692 lands. Some gfx942 asm FLAT candidates fault; the tuner restarts its pool (#6013).
