# overlay-refresh-2026-11

2026-11-03. Re-measures the refreshed overlay ([`../../overlay/`](../../overlay/)) on the shipped TP2 + EP2 settings, and the same overlay on newer images.
MI325X, 2 GPUs per engine, two engines at a time (HIP 4+5 and HIP 6+7 of the test host; each container saw only its own two render nodes).
Settings = [`deploy/compose.yaml`](../../deploy/compose.yaml) TP2: expert parallel, MTP 3, AITER FP8 MoE, QuickReduce INT6, tuned fMoE table for E=256,
`FULL_DECODE_ONLY`, `--max-num-seqs 64`, `--max-num-batched-tokens 32768`. Model revision `236dfdf2…`.

| arm | image | overlay |
|---|---|---|
| `s32` | `vllm/vllm-openai-rocm:v0.32.0` (ROCm 7.2.3, aiter 0.1.24.post1) | none |
| `o32` | same | `overlay/qsa.py` + `overlay/indexer_qsa.py` |
| `u33`, `u33b` | `v0.33.0` tag (ROCm **10.0** build, HIP 7.15, aiter 0.1.26), digest `sha256:4fcd39b7…` | same overlay |
| `r72`, `r72b` | `v0.33.0-rocm72` (ROCm 7.2, HIP 7.2.53211, torch 2.13, aiter 0.1.26, digest `sha256:4fe7bc12…`) | same overlay |

One boot per arm. `u33`/`u33b` and `r72`/`r72b` are the same arm on pair A and pair B, as a check that the two GPU pairs perform alike.
Per boot: GSM8K-200 (zero-shot), then 3 passes of the perfsuite (`prefill_1`, `prefill_8`, `decode_c1/c16/c64`, `agent_32k_c16`, `agent_96k_c8`,
[tools/experiments/perfsuite.py](../../tools/experiments/perfsuite.py)). Tables report the median [min-max] over the 3 passes.

Files: `<arm>-r<round>-<pair>.p<1-3>.perf.json`, `….gsm8k.json`, `….gpu.txt` (PCI addresses seen inside the container, matched against the expected pair).
`python3 agg.py s32-r1-A o32-r1-B u33-r2-A u33b-r2-B r72-r3-A r72b-r3-B` prints the comparison table.
`kernel-test-repeats.txt`: repeated runs of the overlay's kernel tests on the three images, including a flaky test.

The README tables use pair A for `u33` and `r72` and compare against `s32` (pair A) and `o32` (pair B). The second copies (`u33b`, `r72b`) agree within
about 3% on prefill, decode c64 and agent wall time and differ more on decode c1 and c16; the `r72b` GSM8K-200 is 0.960.
