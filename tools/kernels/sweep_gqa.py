import importlib.util, time, torch
spec = importlib.util.spec_from_file_location("p", "/patch/qsa.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = "cuda"; torch.manual_seed(0)
def case(rows, heads, kvh, seq=32000, page=1600, topk=2051, D=256):
    pages = (seq + page - 1) // page + 2
    k = (torch.randn(pages, page, kvh, D, device=dev) * .3).bfloat16(); v = torch.randn_like(k)
    q = (torch.randn(rows, heads, D, device=dev) * .3).bfloat16()
    bt = torch.arange(pages, device=dev, dtype=torch.int32)[None, :].contiguous()
    pos = torch.arange(seq - rows, seq, device=dev)
    # realistic-ish: local window + random far blocks, -1 padding for early rows
    idx = torch.full((rows, topk), -1, device=dev, dtype=torch.int32)
    for i in range(0, rows, 1):
        p = pos[i].item(); n = min(topk, p + 1)
        if n == p + 1:
            idx[i, :n] = torch.arange(n, device=dev, dtype=torch.int32)
        else:
            local = torch.arange(p - 511, p + 1, device=dev, dtype=torch.int32)
            far = torch.randperm(p - 511, device=dev)[: topk - 512].int().sort().values
            idx[i] = torch.cat([far, local])
    t2r = torch.zeros(rows, device=dev, dtype=torch.int32)
    return q, k, v, idx, bt, t2r
def bench(f, it=10):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3
for name, (heads, kvh) in {"TP2 (12q/1kv)": (12, 1), "TP1 (24q/2kv)": (24, 2)}.items():
    c = case(4096, heads, kvh)
    m._QSA_GQA_PREFILL = None
    ref = m.qsa_sparse_paged_attention(*c)
    base = bench(lambda: m.qsa_sparse_paged_attention(*c))
    print(f"{name} 4096 rows stock: {base:.2f} ms", flush=True)
    res = []
    for bn in (16, 32, 64, 128):
        for w in (2, 4, 8):
            for st in (1, 2):
                for sp in (1, 2):
                    m._QSA_GQA_PREFILL = (bn, sp, w, st)
                    try:
                        o = m.qsa_sparse_paged_attention(*c)
                        err = (o.float() - ref.float()).abs().max().item()
                        t = bench(lambda: m.qsa_sparse_paged_attention(*c))
                        res.append((t, bn, w, st, sp, err))
                    except Exception as e:
                        pass
    for r in sorted(res)[:6]:
        print(f"   {r[0]:.2f} ms ({base / r[0]:.2f}x) bn{r[1]} w{r[2]} st{r[3]} sp{r[4]} maxerr {r[5]:.2e}", flush=True)
