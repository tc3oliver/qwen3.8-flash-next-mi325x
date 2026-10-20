# tools

For verifying the results or re-tuning. Not needed to use the overlay or the tuned tables.
Run everything from `tools/`. Scripts read `../deploy/.env` (copy `deploy/.env.example` first).
Python scripts run with `uv run --with httpx ...` (GSM8K also needs `datasets`).

## Corpus

The prompts are real source code: the vLLM Python package from the pinned image. It is not committed. Build it once:

```bash
benchmarks/make_corpus.sh      # -> benchmarks/corpus/ (about 20 MB)
```

## experiments/

| script | use |
|---|---|
| `exp.sh <slot> <name> [VAR=val ...] [BASE_<flag>=x ...] [PATCH_QSA=1] [-- extra args]` | start one engine with the deploy settings as base plus overrides. Slots `a`/`b` = TP2 on `REPLICA_A_HIP`/`REPLICA_B_HIP`, `g0`-`g3` = TP1. `PATCH_QSA=1` mounts `../overlay/qsa.py`; `../tuned/aiter` is mounted at `/tuned` |
| `run.sh <slot> <name> [exp.sh args]` | start -> GSM8K-200 -> perfsuite -> `experiments/results/<name>.json` |
| `queue.sh <slot> <queue-file>` | run queued `run.sh` lines one after another |
| `gate.sh <port> <name>` | functional, full GSM8K, 238K needle on a running engine |
| `perfsuite.py` | the benchmark behind `results/experiments/*.json` |
| `compare.py <run> ...` | side-by-side table; reads `../results/experiments/` (set `RESULTS=experiments/results` for new runs) |
| `profile.py --phase prefill|decode` | torch profile of one phase (engine started with `--profiler-config`) |

Example, d01:

```bash
experiments/run.sh b d01-tp2-combo PATCH_QSA=1 VLLM_QSA_GQA_PREFILL=32,1,2,1 \
  VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6 VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  AITER_CONFIG_FMOE=/usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv:/tuned/tuned_fmoe_qwen38_ep2.csv
```

The exact flags and env of every shipped run are in `results/experiments/<run>.config.txt`.

## kernels/

Kernel checks for `overlay/qsa.py`, run inside the vLLM image with the overlay and the scripts mounted at `/patch`:

```bash
mkdir -p /tmp/patch && cp ../overlay/qsa.py kernels/*.py /tmp/patch/
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render --ipc host \
  -e HIP_VISIBLE_DEVICES=0 -v /tmp/patch:/patch:ro --entrypoint python3 "$VLLM_IMAGE" /patch/test_qsa.py
```

`test_qsa.py`: correctness (logits, selection) and speed vs the stock kernel. `test_qsa_ties.py`: shows differing
selections are exact ties. `sweep_qsa.py`, `sweep2.py`: indexer tile sweep. `sweep_gqa.py`, `sweep_gqa2.py`:
sparse-GQA launch-profile sweep (`sweep_gqa2.py` runs each config in its own process; some configs fault the GPU).

## tests/

| script | use |
|---|---|
| `functional.py` | 9 checks incl. tool calls, reasoning, #61486 short-prompt determinism (`--gateway` for LiteLLM) |
| `gsm8k_quick.py` | GSM8K zero-shot, `--n 200` or `--n 1319` |
| `long_needle.py` | needle in a long code prompt |
| `burst_prefill.py` | N concurrent long prompts, checks the server stays healthy |
| `affinity_cache.py` | session stickiness and prefix-cache hit through the gateway |
| `failover.py` | stops the pinned replica (`docker stop`) mid-session, then restarts it with `deploy/scripts/up.sh` |
| `sim_users.py` | N users that never clear context, fixed wall time |
| `header_capture.py` | shows which session headers a client sends |
| `agent_real.sh opencode|pi <label>` | real agent task (`agenttask/`) through the gateway in an isolated HOME |

## benchmarks/

`agent_bench.py`: the multi-turn agent benchmark behind `results/benchmark-grid/`:

```bash
uv run --with httpx benchmarks/agent_bench.py --base http://127.0.0.1:4000 --key $LITELLM_MASTER_KEY \
  --label tp1x4-gateway --contexts 8192,32768,65536,131072 --concurrency 1,4,8,16 --fixed-output
```
