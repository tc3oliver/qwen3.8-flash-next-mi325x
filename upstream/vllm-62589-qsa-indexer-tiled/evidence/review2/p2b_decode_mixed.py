# Point 2b: MTP-off decode with mixed context lengths (uniform 8K-200K tokens per request).
src = open("/work/p2_decode.py").read().split('print("context')[0]
exec(src)
print("mixed lengths | reqs x tok | main score ms | branch score ms | main select ms | branch select ms")
for n in (16, 64, 128):
    for seed in (0, 1, 2):
        torch.manual_seed(100 + seed)
        c = make_case([(sl, [sl - 1]) for sl in torch.randint(8000, 200000, (n,)).tolist()])
        sc = [ev(lambda m=m: run_mqa(m, c, slice(0, n))) for m in mods.values()]
        se = [ev(lambda m=m: run_select(m, c)) for m in mods.values()]
        print(f"8K-200K seed{seed} | {n} x 1 | " + " | ".join(f"{x:.4f}" for x in sc + se), flush=True)
        del c; torch.cuda.empty_cache()
