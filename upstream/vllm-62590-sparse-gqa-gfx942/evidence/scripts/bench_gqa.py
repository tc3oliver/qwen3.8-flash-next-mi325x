import importlib.util, time, torch
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
stock, pr3 = load("/work/stock.py", "stock"), load("/work/pr3.py", "pr3")
dev = "cuda"
def case(rows, heads, kvh, seq=32000, page=1600, topk=2051, D=256):
    torch.manual_seed(0)
    pages = (seq + page - 1) // page + 2
    k = (torch.randn(pages, page, kvh, D, device=dev) * .3).bfloat16(); v = torch.randn_like(k)
    q = (torch.randn(rows, heads, D, device=dev) * .3).bfloat16()
    bt = torch.arange(pages, device=dev, dtype=torch.int32)[None, :].contiguous()
    g = torch.Generator(device="cpu").manual_seed(1)
    pos = torch.arange(seq - rows, seq)
    far = torch.stack([torch.randperm(int(p) - 511, generator=g)[: topk - 512].sort().values for p in pos])
    local = pos[:, None] - 511 + torch.arange(512)[None, :]
    idx = torch.cat([far, local], 1).int().to(dev)
    return (q, k, v, idx, bt, torch.zeros(rows, device=dev, dtype=torch.int32))
def bench(f, it=20):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3
print("shape | rows | programs | stock ms | PR-3 ms | speedup | max abs diff")
for name, (h, kvh) in {"TP2 (12 q / 1 kv)": (12, 1), "TP1 (24 q / 2 kv)": (24, 2), "TP4 (6 q / 1 kv)": (6, 1)}.items():
    for rows in (((520 if kvh == 1 else 260), 1024, 4096) if h != 6 else (520, 4096)):
        a = case(rows, h, kvh)
        o0 = stock.qsa_sparse_paged_attention(*a).float(); o1 = pr3.qsa_sparse_paged_attention(*a).float()
        t0 = bench(lambda: stock.qsa_sparse_paged_attention(*a)); t1 = bench(lambda: pr3.qsa_sparse_paged_attention(*a))
        print(f"{name} | {rows} | {rows*kvh} | {t0:.3f} | {t1:.3f} | {t0/t1:.2f}x | {(o0-o1).abs().max().item():.1e}", flush=True)
