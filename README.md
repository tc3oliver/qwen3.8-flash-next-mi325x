# Qwen3.8-Flash-Next-FP8 on AMD MI325X

Serving settings and a two-file vLLM overlay that roughly double prefill and halve agent wall time for Qwen3.8-Flash-Next-FP8 on MI325X (gfx942), behind a LiteLLM gateway. Tuned AITER MoE tables are included; their only isolated end-to-end measurement moved 8 × 32K prefill by under 1%, on TP1.

## Results

Plain vLLM v0.32.0 (same flags; no overlay, no tuned tables, no QuickReduce) against this repo, TP2 + expert parallel, run d02 (October).
Run-to-run noise is about 3-5%, more on cold TTFT: stock measured 3.48 s and 2.92 s in two runs with the same flags.

| TP2, one engine on 2 GPUs | stock | this repo |
|---|---:|---:|
| Prefill, 8 x 32K concurrent | 11,469 tok/s | **24,266 tok/s** |
| 32K cold TTFT, single request | 3.48 s | **1.51 s** |
| Decode, 64 concurrent | 2,167 tok/s | **3,208 tok/s** |
| Agent 32K, 16 sessions x 3 turns, wall | 63.1 s | **31.9 s** |
| Agent 96K, 8 sessions x 3 turns, wall | 84.9 s | **41.7 s** |
| GSM8K, 1319 questions zero-shot | | 0.9629 |

Accuracy gate for the same run: 238K needle 3/3, functional 9/9.

At 100 simulated users on 2 x TP2 through LiteLLM, with the first overlay revision:
0 errors in 2,312 requests, warm TTFT p50 0.58 s, p99 49.8 s, 94% warm-turn cache hit.

Re-measured 2026-11-03 with the refreshed overlay. Here QuickReduce INT6 and the tuned MoE table are on in **both** columns,
so the difference is the overlay (and, in the last two columns, the newer vLLM). Medians of 3 passes.

| TP2 + EP2, MTP 3 | v0.32.0 stock | v0.32.0 + overlay | v0.33.0 (ROCm 10.0) + overlay | v0.33.0-rocm72 + overlay |
|---|---:|---:|---:|---:|
| Prefill, 8 x 32K concurrent (tok/s) | 12,966 | 25,917 | 27,271 | 26,540 |
| 32K cold TTFT (s) | 2.94 | 1.54 | 1.48 | 1.48 |
| Decode, 64 concurrent (tok/s) | 2,262 | 3,292 | 3,552 | 3,417 |
| Agent 32K, 16 sessions, wall (s) | 56.5 | 32.1 | 29.1 | 30.8 |
| Agent 96K, 8 sessions, wall (s) | 77.4 | 42.4 | 39.1 | 39.4 |
| GSM8K-200 | 0.975 | 0.980 | 0.965 | 0.975 |

Caveats:
- The newest numbers are one boot per arm, 3 passes, medians; ranges are in [`results/overlay-refresh-2026-11/`](results/overlay-refresh-2026-11/).
- vLLM v0.33.0 gains 2-7% over v0.32.0 on the same ROCm 7.2 (`v0.33.0-rocm72`) and 5-9% on the ROCm 10.0 build (`v0.33.0` tag). Both were tested with the overlay; neither is the compose default (v0.32.0, ROCm 7.2.3).
- GSM8K-200 has a standard error of about 1.2 points; the 1319-question run is the accuracy gate. It has not been run on v0.33.0, where the four n=200 runs (0.960-0.975) sit slightly below the v0.32.0 ones (0.975, 0.980).

## Quick start

```bash
git clone https://github.com/tc3oliver/qwen3.8-flash-next-mi325x && cd qwen3.8-flash-next-mi325x/deploy
cp .env.example .env && chmod 600 .env   # set MODEL_DIR, HIP ids, LITELLM_MASTER_KEY, LAYOUT (tp2x2 or tp1x4)
./scripts/up.sh                          # docker compose; overlay/ and tuned/ are mounted by compose.yaml
./scripts/status.sh
```

Pick the HIP device ids for your host first; the mapping in `.env.example` is from our machine.

## What is included

- [`overlay/`](overlay/): `qsa.py` and `indexer_qsa.py` for vLLM v0.32.0 / v0.33.0 (tiled QSA indexer, sparse-GQA BLOCK_N=32), kernel tests, stock hashes.
- [`tuned/aiter/`](tuned/aiter/): tuned AITER fused-MoE tables for TP1 and TP2 + EP2 (304 CUs).
- [`deploy/`](deploy/): compose for 2 x TP2 or 4 x TP1 plus LiteLLM with session affinity, and operating scripts.
- [`results/`](results/): raw data for every number above.
- [`upstream/`](upstream/): the upstream PRs with their evidence.
- [`tools/`](tools/): benchmark and test scripts.

Caveat: per-row decode (part of #62589) is inactive with MTP 3; the shipped deployment uses the tiled kernel for decode.

## Validated on

`vllm/vllm-openai-rocm:v0.32.0` @ `sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905`
(ROCm 7.2.3, torch 2.13, aiter 0.1.24.post1), MI325X (gfx942, 304 CUs, 256 GiB HBM),
model `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13`.

## Upstream status (2026-11-03)

| PR | content | status |
|---|---|---|
| [vllm#62589](https://github.com/vllm-project/vllm/pull/62589) | tiled QSA indexer | open, no maintainer review |
| [vllm#62590](https://github.com/vllm-project/vllm/pull/62590) | sparse-GQA launch profile on gfx942 and gfx950 | open, no maintainer review |
| [vllm#64662](https://github.com/vllm-project/vllm/pull/64662) | MI325X Triton FP8 MoE configs | open, no maintainer review |
| [aiter#6692](https://github.com/ROCm/aiter/pull/6692) | MoE tuner fallback | merged 2026-11-03 |
| [aiter#6693](https://github.com/ROCm/aiter/pull/6693) | gfx942 MoE configs | merged 2026-10-22; not yet in the vLLM v0.32.0 / v0.33.0 images, so `tuned/aiter/` is still needed |

CI on the vLLM PRs waits for a maintainer `ready` label. The #64662 configs do not apply to the shipped AITER-MoE deployment.

## Details

- [`docs/full-report.md`](docs/full-report.md): every table, setting, caveat and the manual `docker run` commands.
- [`docs/lab-notebook.md`](docs/lab-notebook.md): chronological experiment log.
- [`results/`](results/): raw data and the 2026-11 refresh.
- [`overlay/`](overlay/): what the overlay changes and how it is tested.
- [`upstream/`](upstream/): PR texts and evidence.

## License

Apache-2.0 for this repository. `overlay/*.py` and the `upstream/*/qsa.py` files are modified vLLM source (Apache-2.0).
The model weights are under the Qwen Community License and are not included.
