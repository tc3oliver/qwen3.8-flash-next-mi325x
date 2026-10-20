# Upstream PRs

One folder per PR: `PR.md` is the final PR text (GitHub checklist template removed), plus the evidence behind it.

| folder | PR |
|---|---|
| `vllm-62589-qsa-indexer-tiled/` | [vllm-project/vllm#62589](https://github.com/vllm-project/vllm/pull/62589): `qsa.py` (the branch file used for the end-to-end run), `evidence/` (microbenchmarks, selection check, fp8 check, pytest output, the three mutants the new test catches, scripts) |
| `vllm-62590-sparse-gqa-gfx942/` | [vllm-project/vllm#62590](https://github.com/vllm-project/vllm/pull/62590): `qsa.py`, `evidence/` (microbenchmark, launch-profile sweep, TP4 point, pytest output, scripts) |
| `aiter-6692-moe-tuner-fallback/` | [ROCm/aiter#6692](https://github.com/ROCm/aiter/pull/6692): `evidence/tune_fmoe.run1-no-candidates.log`, the tuner run with `TUNE_ONLY` unset that found 0 candidates |
| `aiter-6693-gfx942-moe-configs/` | [ROCm/aiter#6693](https://github.com/ROCm/aiter/pull/6693): untuned shape lists and per-shape tuner profiles in `configs/`, full tuner logs in `evidence/`. The tuned tables are in `../tuned/aiter/` |

End-to-end runs for the vLLM PRs: `results/experiments/pr-stock-tp2`, `fin-tiled-tp2`, `fin-pr3-tp2`.
The evidence scripts expect files mounted at `/work` or `/patch` inside the vLLM image; `repotest.sh` shows the mount layout.
