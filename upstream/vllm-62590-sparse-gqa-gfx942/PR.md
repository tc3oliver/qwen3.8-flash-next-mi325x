[ROCm][Perf][Qwen4Exp] Use BLOCK_N=32 for sparse GQA prefill on gfx942

<!-- markdownlint-disable -->

## Overview

On gfx942, the >512-program launch profile of `qsa_sparse_paged_attention` uses BLOCK_N=32 instead of 64 when BLOCK_M is 16. The existing profile was tuned on GB300. The BLOCK_M=8 shape (TP4) and other architectures are unchanged.

## Claims

- Sparse GQA on MI325X, 2,051 selected tokens per row, head_dim 256: 1.5x faster at about 520 programs and 2.1-2.3x at 1K-8K programs, for the TP1 and TP2 shapes (BLOCK_M=16).
- Qwen3.8-Flash-Next-FP8, TP2, 2x MI325X: 8 x 32K concurrent prefill 11,486 -> 12,272 tok/s (+6.8%). Other metrics are within noise.
- Output differs from main by at most one bf16 ulp (4.9e-4), from tile accumulation order.

## Validation

New parametrize case `test_qsa_sparse_paged_attention_matches_reference[tp2_split1]` (513 rows, 12 q / 1 kv heads) takes this profile on gfx942. The existing `tp4_split1` case has BLOCK_M=8 and does not.

    pytest tests/models/qwen4_exp/test_qsa_amd.py   # MI325X (gfx942): 17 passed

**Microbenchmark**, `qsa_sparse_paged_attention` on one MI325X: 32K context, page size 1600, 2,051 selected tokens per row (512 local + 1,539 random far), head_dim 256. Synchronized wall-clock, 20 iterations after warmup.

| shape | rows | programs | main (ms) | this PR (ms) | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|---:|
| TP2 (12 q / 1 kv) | 520 | 520 | 0.406 | 0.270 | 1.50x | 4.9e-4 |
| TP2 (12 q / 1 kv) | 1,024 | 1,024 | 0.787 | 0.337 | 2.34x | 4.9e-4 |
| TP2 (12 q / 1 kv) | 4,096 | 4,096 | 2.662 | 1.261 | 2.11x | 4.9e-4 |
| TP1 (24 q / 2 kv) | 260 | 520 | 0.411 | 0.273 | 1.51x | 4.9e-4 |
| TP1 (24 q / 2 kv) | 1,024 | 2,048 | 1.536 | 0.677 | 2.27x | 4.9e-4 |
| TP1 (24 q / 2 kv) | 4,096 | 8,192 | 5.324 | 2.559 | 2.08x | 4.9e-4 |
| TP4 (6 q / 1 kv), unchanged | 4,096 | 4,096 | 2.188 | 2.169 | 1.01x | 0 |

With BLOCK_N=32 the TP4 shape (BLOCK_M=8) measured 2.17 -> 10.66 ms (5x slower) at 4,096 rows, hence the `block_m >= 16` condition.

<details>
<summary>Launch-profile sweep at 4,096 rows (ms; separate harness, each config in its own process)</summary>

| (BLOCK_N, splits, warps, stages) | TP2 | TP1 |
|---|---:|---:|
| (64, 1, 2, 1), main on ROCm | 2.66 | 5.33 |
| **(32, 1, 2, 1), this PR** | **1.37** | **2.65** |
| (32, 1, 4, 1) | 1.72 | 3.30 |
| (32, 1, 2, 2) | 2.15 | 4.21 |
| (32, 1, 4, 2) | 1.76 | 4.44 |
| (128, 1, 8, 1) | 3.01 | 5.78 |

(64/128, 1, 4, 1) fault on gfx942; no profile uses them.
</details>

<details>
<summary>Microbenchmark script</summary>

```python
# stock.py = main's qsa.py, pr3.py = this branch's.
import importlib.util, time, torch
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
stock, pr3 = load("stock.py", "stock"), load("pr3.py", "pr3")
def case(rows, heads, kvh, seq=32000, page=1600, topk=2051, D=256):
    torch.manual_seed(0)
    pages = (seq + page - 1) // page + 2
    k = (torch.randn(pages, page, kvh, D, device="cuda") * .3).bfloat16(); v = torch.randn_like(k)
    q = (torch.randn(rows, heads, D, device="cuda") * .3).bfloat16()
    bt = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :].contiguous()
    g = torch.Generator().manual_seed(1)
    pos = torch.arange(seq - rows, seq)
    far = torch.stack([torch.randperm(int(p) - 511, generator=g)[: topk - 512].sort().values for p in pos])
    local = pos[:, None] - 511 + torch.arange(512)[None, :]
    idx = torch.cat([far, local], 1).int().cuda()
    return (q, k, v, idx, bt, torch.zeros(rows, device="cuda", dtype=torch.int32))
def bench(f, it=20):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3
for h, kvh, rows_list in ((12, 1, (520, 1024, 4096)), (24, 2, (260, 1024, 4096)), (6, 1, (520, 4096))):
    for rows in rows_list:
        a = case(rows, h, kvh)
        diff = (stock.qsa_sparse_paged_attention(*a).float() - pr3.qsa_sparse_paged_attention(*a).float()).abs().max()
        print(h, kvh, rows, bench(lambda: stock.qsa_sparse_paged_attention(*a)),
              bench(lambda: pr3.qsa_sparse_paged_attention(*a)), diff.item())
```
</details>

**End-to-end.** Qwen3.8-Flash-Next-FP8, TP2 + EP on 2x MI325X: MTP 3, max-num-batched-tokens 32768, max-num-seqs 64. vLLM v0.32.0 image with this PR's `qsa.py` mounted (v0.32.0's `qsa.py` is identical to main's). Run-to-run noise is ±3-5%.

| metric | main | this PR |
|---|---:|---:|
| prefill 8 x 32K (tok/s) | 11,486 | 12,272 |
| prefill 1 x 32K (tok/s) | 11,726 | 11,999 (within noise) |
| 16 agents, 32K context: wall (s) | 62.0 | 60.3 (within noise) |
| GSM8K-200 | 0.970 | 0.965 (within noise) |

## Details

On ROCm the profile already uses one pipelining stage (64 KiB LDS), so this changes only BLOCK_N. The arch check follows #57969 (lazy `vllm.platforms.rocm` import under `is_rocm()`). gfx950 is unchanged because I have no MI355X to measure on. CI runs on MI355, so it does not exercise the gfx942 branch.

Not a duplicate: I searched open PRs on 2026-10-20 ("qsa rocm", "qwen4_exp qsa", "qsa_sparse_paged_attention", "QSA gfx942", "qwen4_exp amd"). #57969 touches the same function (gfx11 BLOCK_M) in a different hunk. #60650 and #61529 change dtype handling only. #59437 is a separate opt-in path.
