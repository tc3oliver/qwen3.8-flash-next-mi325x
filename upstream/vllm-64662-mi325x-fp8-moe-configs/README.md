# vllm-project/vllm#64662

[ROCm][Perf] Add MI325X FP8 fused MoE configs for Qwen3.8-Flash-Next: Triton fused-MoE configs for `E=512,N=320,[64,64]` (TP2) and
`E=512,N=160,[32,32]` (TP4). `PR.md` is the PR text, `TITLE.txt` its title; the two JSON files are in the PR.

**Does not apply to this repo's deployment.** The shipped compose uses AITER FP8 MoE (TP2 with expert parallel, local E=256; TP1 with E=512, N=640).
These files are read only by the Triton FP8 MoE backend under pure TP, i.e. vLLM's default (`VLLM_ROCM_USE_AITER` unset) at TP2/TP4. TP4 on the
2026-10-29 nightly: c64 output throughput +20.4%, TPOT -20.2%, prefill +15.7%; TP2 on v0.32.0: +4.9% / +5.1%. Status: open, waiting for a maintainer
`ready` label.
