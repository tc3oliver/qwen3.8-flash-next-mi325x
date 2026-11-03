# usage: python3 -I bench_qsa.py <stock.py> <pr.py> ; harness.py (make_case helpers) is loaded from /work/harness.py
import os, importlib.util, sys, time, statistics as st, torch
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
src = open("/work/harness.py").read().split("cases = {")[0]
src = src.replace('orig = load("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py", "qsa_orig")', f'orig = load("{sys.argv[1]}", "stock")').replace('patched = load("/patch/qsa.py", "qsa_patched")', f'patched = load("{sys.argv[2]}", "pr")')
exec(src)
mods = {"stock": orig, "pr": patched}
IT = 50
def timeit(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize(); g, w = [], []
    for _ in range(IT):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        t = time.perf_counter(); s.record(); fn(); e.record(); torch.cuda.synchronize()
        w.append((time.perf_counter() - t) * 1000); g.append(s.elapsed_time(e))
    f = lambda x: f"{st.median(x):.4f} [{min(x):.4f}-{max(x):.4f}]"
    return f(g), f(w)
def agree(c, n_rows):
    a = run_select(orig, c); b = run_select(patched, c)
    lg, vis = orig.qsa_mqa_paged(c["q"], c["k_cache"], c["page_table"], c["token_to_req"], c["query_positions"], c["sequence_lengths"], CR)
    ident = 0; ties = 0; nontie = 0
    for r in range(n_rows):
        A = set((a[r][a[r] >= 0] // CR).tolist()); B = set((b[r][b[r] >= 0] // CR).tolist())
        if A == B: ident += 1; continue
        d = sorted(A ^ B); v = lg[r, d]
        # tie = every differing block scores the same as the k-th boundary value (all diff blocks equal-valued)
        if (v.max() - v.min()).item() == 0: ties += 1
        else: nontie += 1; print(f"  nontie row {r}: diff blocks {len(d)}, stock logit range {v.min().item():.6f}..{v.max().item():.6f}", flush=True)
    return f"rows={n_rows} identical={ident} diff_tie={ties} diff_nontie={nontie}"
print("arch", torch.cuda.get_device_properties(0).gcnArchName, flush=True)
print("ctx | tok/req | conc | rows | stock score GPUms | pr score GPUms | stock select GPUms | pr select GPUms | stock select wall | pr select wall | topk agreement")
for ctx in ((262000,) if os.environ.get('GRID')=='262k' else (32768, 98304, 262000)):
    for tok in ((1,) if os.environ.get('GRID')=='262k' else (1, 4)):
        for n in ((16, 64) if os.environ.get('GRID')=='262k' else (1, 4, 16, 64)):
            torch.manual_seed(n)
            c = make_case([(ctx, list(range(ctx - tok, ctx)))] * n)
            sc = [timeit(lambda m=m: run_mqa(m, c, slice(0, n * tok))) for m in mods.values()]
            se = [timeit(lambda m=m: run_select(m, c)) for m in mods.values()]
            print(f"{ctx} | {tok} | {n} | {n*tok} | {sc[0][0]} | {sc[1][0]} | {se[0][0]} | {se[1][0]} | {se[0][1]} | {se[1][1]} | {agree(c, n*tok)}", flush=True)
            del c; torch.cuda.empty_cache()
