# Final microbench: main vs final (qsa_select_paged_tokens, CUDA-event timing).
src = open("/work/p2_decode.py").read().split('print("context')[0]
exec(src)
names = list(mods)
print("case | rows | " + " | ".join(f"{m} ms" for m in names) + " | speedup")
def row(label, c, it=200):
    se = [ev(lambda m=m: run_select(m, c), it) for m in mods.values()]
    print(f"{label} | {c['q'].shape[0]} | " + " | ".join(f"{x:.4f}" for x in se) + f" | {se[0] / se[-1]:.2f}x", flush=True)
c = make_case([(32000, list(range(32000)))]); row("prefill 32K", c, 10); del c
c = make_case([(20000, list(range(12000, 20000))), (5000, list(range(5000))), (70000, [69999]), (130000, [129996, 129997, 129998, 129999])]); row("mixed prefill + decode", c, 10); del c
torch.manual_seed(1)
c = make_case([(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 200000, (64,)).tolist()]); row("MTP3 decode 64x4, contexts 8K-200K", c); del c
for n in (1, 16, 64, 128):
    torch.manual_seed(100)
    c = make_case([(sl, [sl - 1]) for sl in torch.randint(8000, 200000, (n,)).tolist()]); row(f"MTP-off {n}x1, contexts 8K-200K", c); del c; torch.cuda.empty_cache()
for n in (16, 64, 128):
    c = make_case([(250000, [249999])] + [(4096, [4095])] * (n - 1)); row(f"MTP-off {n}x1, one 250K among 4K", c); del c; torch.cuda.empty_cache()
