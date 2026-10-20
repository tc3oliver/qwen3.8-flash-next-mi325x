# Point 2c: per-request skip candidates (skipA = load masks, skipB = scalar branch) vs main/branch.
src = open("/work/p2_decode.py").read().split('print("context')[0]
exec(src)
names = list(mods)
print("case | " + " | ".join(f"{m} select ms" for m in names) + " | max |logit diff| vs branch (visible)")
def row(label, c, it=200):
    n = c["q"].shape[0]
    se = [ev(lambda m=m: run_select(m, c), it) for m in mods.values()]
    rs = slice(0, min(n, 512))
    lb, vb = run_mqa(mods["branch"], c, rs)
    mask = torch.arange(lb.shape[1], device="cuda")[None] < vb[:, None]
    d = []
    for m in names[2:]:
        lm, vm = run_mqa(mods[m], c, rs); assert torch.equal(vm, vb)
        d.append(f"{m} {(lm[mask] - lb[mask]).abs().max().item():.1e}")
    print(f"{label} | " + " | ".join(f"{x:.4f}" for x in se) + " | " + ", ".join(d), flush=True)
for n in (16, 64, 128):
    for seed in (0, 1):
        torch.manual_seed(100 + seed)
        c = make_case([(sl, [sl - 1]) for sl in torch.randint(8000, 200000, (n,)).tolist()])
        row(f"MTP-off mixed 8K-200K {n}x1 seed{seed}", c); del c; torch.cuda.empty_cache()
for ctx in (32768, 262000):
    c = make_case([(ctx, [ctx - 1])] * 16); row(f"MTP-off fixed {ctx} 16x1", c); del c; torch.cuda.empty_cache()
torch.manual_seed(1)
c = make_case([(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 200000, (64,)).tolist()]); row("MTP3 decode 64x4 mixed", c); del c
c = make_case([(32000, list(range(32000)))]); row("prefill 32K", c, 10); del c
c = make_case([(20000, list(range(12000, 20000))), (5000, list(range(5000))), (70000, [69999]), (130000, [129996, 129997, 129998, 129999])]); row("mixed prefill+decode", c, 10)
