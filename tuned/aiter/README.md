# Tuned AITER fused-MoE tables

Append to AITER's default table with `AITER_CONFIG_FMOE` (paths are colon-separated):

```
-v $PWD/tuned/aiter:/aiter-tuned:ro
-e AITER_CONFIG_FMOE=/usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv:/aiter-tuned/tuned_fmoe_qwen38_ep2.csv   # TP2 + EP
-e AITER_CONFIG_FMOE=/usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv:/aiter-tuned/tuned_fmoe_qwen38_tp1.csv   # TP1
```

| file | layout | rows (tokens) | gain at those shapes (tuner `--compare`) |
|---|---|---|---|
| `tuned_fmoe_qwen38_ep2.csv` | TP2 + `--enable-expert-parallel` (E=256 local) | 2, 4, 1024, 8192, 16384 | -5.6%, -13.0%, -3.7%, -7.2%, -7.7% time |
| `tuned_fmoe_qwen38_tp1.csv` | TP1 (E=512) | 4, 16384, 32768 | -6.2%, -7.5%, -11.7% time |

Scope: gfx942 with 304 CUs (MI325X), aiter 0.1.24.post1 (in the vLLM v0.32.0 ROCm image),
Qwen3.8-Flash-Next-FP8 routed experts only (FP8 `per_1x128` block scale, model_dim 2560, inter_dim 640, top-10,
SiLU, g1u1). Rows are keyed on these values; other models or CU counts do not match them and fall back to AITER's
defaults. Only shapes at least 3% faster than the default were kept.

End-to-end effect is small (TP1 8x32K prefill +0.8% / +0.7%, see `upstream/aiter-6693-gfx942-moe-configs/PR.md`).
Upstreamed in [ROCm/aiter#6693](https://github.com/ROCm/aiter/pull/6693); once a vLLM image ships an aiter with
those rows, `AITER_CONFIG_FMOE` is not needed. Tuning logs: `upstream/aiter-6693-gfx942-moe-configs/evidence/`.
