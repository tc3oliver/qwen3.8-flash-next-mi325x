# Phase 2 TP2 end-to-end A/B runbook (stock vs overlay), MI300X

Image: `vllm/vllm-openai-rocm:v0.33.0-rocm72`, digest `sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182` (set in `deploy/.env`, git-ignored).
Stock `ops/qsa.py` / `indexer_qsa.py` sha256 inside it match `overlay/README.md` (checked 2026-11-03). Also checked on v0.32.0 (`sha256:14f6a575...`, smoke in `smoke-v0.32.0/`).
Config = `deploy/compose.yaml` TP2: EP2, MTP 3, AITER FP8 MoE, QuickReduce INT6 (+ CAST_BF16_TO_FP16=0), `tuned_fmoe_qwen38_ep2.csv` (all 5 rows `gfx942,304` CU), FULL_DECODE_ONLY, max-num-seqs 64, max-num-batched-tokens 32768, max-model-len 262144.
Arm A = `stock`, arm B = `overlay` (`PATCH_QSA=1`). Same image, same flags. (Arm "A/B" below = stock/overlay; pairs are named Pair A / Pair B.)

GPUs: Pair A = HIP 6,7 (renderD176+184, container `mi300-e2e-a`, port 8011); Pair B = HIP 2,3 (renderD144+152, container `mi300-e2e-b`, port 8012). HIP 4 is never mounted
(`exp.sh` takes the render-node table from `RENDER_MAP` in `deploy/.env`, see `deploy/.env.mi300x.example`: HIP 4 is not in it and unknown ids are refused; the runs themselves used the first, hard-coded version of this table, commit 0b1c69a). Inside the container ids are 0..1.

`arm.sh <stock|overlay> <A|B> <round> [smoke]` (this dir) does: boot -> hash of the files python actually imports (aborts with exit 2 on MISMATCH) -> PCI BDFs seen in the container (`*.gpu.txt`, expect e6+96 for Pair A, 66+16 for Pair B) -> 3 x perfsuite `--only prefill_8,decode_c64,agent_32k_c16` -> remove container.
Outputs: `<arm>-r<round>-<pair>.{hash.txt,gpu.txt,server.log,p1,p2,p3.perf.json}` here. Default perfsuite behaviour is unchanged (`--only` already existed).

## Rounds (run from the repo root; the two arms of a round run concurrently, one per pair, as in overlay-refresh-2026-11)

```bash
cd /home/user/projects/qwen3.8-flash-next-mi325x/results/mi300-2026-11/phase2-e2e
# Round 1: stock on Pair A, overlay on Pair B
./arm.sh stock A 1 > r1-stock.console 2>&1 &  ./arm.sh overlay B 1 > r1-overlay.console 2>&1 &  wait
# Round 2: swapped: stock on Pair B, overlay on Pair A
./arm.sh stock B 2 > r2-stock.console 2>&1 &  ./arm.sh overlay A 2 > r2-overlay.console 2>&1 &  wait
# Round 3: as Round 1 (stock on A, overlay on B)
./arm.sh stock A 3 > r3-stock.console 2>&1 &  ./arm.sh overlay B 3 > r3-overlay.console 2>&1 &  wait
python3 -I agg.py
```
Round 3 = Round 1 so the third data point is not a new variable; pair bias stays visible because file names carry the pair (earlier r72/r72b check: pairs agree within ~3% on prefill / c64 / agent).
Check before each round: `docker ps | grep mi300-e2e` empty, `ss -ltn | grep -E ':801[12]'` empty. If an arm exits non-zero, read its `*.console` (exit 2 = overlay/stock hash mismatch, results invalid).
Cleanup if interrupted: `docker rm -f mi300-e2e-a mi300-e2e-b`.

## Limits of the comparison

- The overlay is #62589 + #62590 + the indexer_qsa backport: the e2e delta cannot be attributed to a single PR.
- Pair design: R1 stock@A, R2 stock@B, R3 stock@A leaves stock with 2/3 of its data on Pair A and overlay 2/3 on Pair B (pairs sit on different NUMA nodes: HIP 6,7 = NUMA 1, HIP 2,3 = NUMA 0). Pooled numbers carry that bias; the headline is the per-round ratio (both arms run concurrently in a round), plus a pair-bias estimate (R2 ratio vs mean of R1,R3).
- Pass 1 of each boot is a cold warm-up (e.g. r1-stock p1 prefill 9394 vs 11893 / 11338 for p2 / p3), so agg.py also reports passes 2-3 only. p2 vs p3 differ by up to ~5%: differences below ~5% are not resolvable.
- Crashes: overlay r2-A and r3-B each crashed once on the first attempt (HSA_STATUS_ERROR_OUT_OF_RESOURCES / hipErrorUnknown); kept in `invalid-crash-attempt1/`, not aggregated. The reruns passed.

## Aggregation and stopping rule

`python3 -I agg.py` prints (all passes, then passes 2-3 only), per metric: pooled median [min-max], per-round overlay-vs-stock %, the pair-bias estimate, and a verdict from the per-round ratios (separated = every round better by >5%). It warns loudly about missing files/rounds and then does not compute the verdict.
Stopping rule: after 3 rounds, if the improvement is not separated beyond the round-to-round spread (verdict "overlap (within noise)"), report it as within measurement noise and do NOT add rounds.

## Timing

- Boot (observed, 3 boots, one engine at a time): 325 s (v0.32.0, first, cold cache), 275 s (v0.32.0), 265 s (rocm72). Two simultaneous boots: not measured.
- Perf passes: the full 9-test suite is documented as ~5 min per pass (perfsuite.py docstring); the 3-test subset is shorter but its time is NOT measured. Upper bound per round ~5 + 3x5 = 20 min, i.e. 3 rounds <= ~60-65 min (unverified estimate; real round time is expected to be lower).
- Not included, on purpose: GSM8K, needle, functional gate (lead decides).
