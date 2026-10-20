import sys, torch
src = open("/work/harness.py").read().split("cases = {")[0]
src = src.replace('orig = load("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py", "qsa_orig")', 'orig = load("/work/stock.py", "qsa_orig")')
src = src.replace('patched = load("/patch/qsa.py", "qsa_patched")', 'patched = None')
exec(src)
variants = {n: load(f"/work/{n}.py", n) for n in sys.argv[1:]}
torch.manual_seed(1)
cases = {
 "prefill 32K (32000 rows)": [(32000, list(range(32000)))],
 "chunk 16K at 96K": [(98304, list(range(98304 - 16384, 98304)))],
 "decode 64 req x4": [(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 200000, (64,)).tolist()],
 "decode 16 req x4": [(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 120000, (16,)).tolist()],
 "decode 1 req x4": [(60000, list(range(59996, 60000)))],
}
print("case | stock ms | " + " | ".join(variants))
for name, reqs in cases.items():
    c = make_case(reqs)
    it = 5 if c["q"].shape[0] > 1000 else 50
    so = run_select(orig, c); t0 = bench(lambda: run_select(orig, c), it)
    row = [f"{t0:.3f}"]
    for n, m in variants.items():
        # logits equality within visible on first 256 rows
        rs = slice(0, min(256, c["q"].shape[0]))
        lo, vo = run_mqa(orig, c, rs); lp, vp = run_mqa(m, c, rs)
        assert torch.equal(vo, vp)
        mask = torch.arange(lo.shape[1], device="cuda")[None] < vo[:, None].long()
        err = (lo[mask] - lp[mask]).abs().max().item() if mask.any() else 0.0
        t = bench(lambda: run_select(m, c), it)
        row.append(f"{t:.3f} ({t0/t:.1f}x, err {err:.1e})")
        del lp
    print(f"{name} | " + " | ".join(row), flush=True)
    del c; torch.cuda.empty_cache()
