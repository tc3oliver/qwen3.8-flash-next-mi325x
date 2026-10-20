# Point 2: MTP-off decode (1 query row per request). main vs branch, qsa_mqa_paged alone and
# qsa_select_paged_tokens (score + top-k + expand). Real indexer shapes: 4 heads, D 128,
# ratio 4, page 400, width 164 (65,600 columns, max_model_len 262,144), top-k 2048.
import time, torch
src = open("/work/harness.py").read().split("cases = {")[0]
src = src.replace('orig = load("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py", "qsa_orig")', 'orig = load("/work/stock.py", "main")').replace('patched = load("/patch/qsa.py", "qsa_patched")', 'patched = load("/work/final_skip.py", "final")')
exec(src)
import sys
extra = {n: load(f"/work/{n}.py", n) for n in sys.argv[1:]}
mods = {"main": orig, "branch": patched, **extra}
print("arch", torch.cuda.get_device_properties(0).gcnArchName)
def ev(fn, it=200):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it
print("context | reqs x tok | " + " | ".join(f"{m} score ms" for m in mods) + " | " + " | ".join(f"{m} select ms" for m in mods))
for ctx in (32768, 98304, 262000):
    for n in (1, 16, 64, 128):
        torch.manual_seed(n)
        c = make_case([(ctx, [ctx - 1])] * n)
        sc = [ev(lambda m=m: run_mqa(m, c, slice(0, n))) for m in mods.values()]
        se = [ev(lambda m=m: run_select(m, c)) for m in mods.values()]
        print(f"{ctx} | {n} x 1 | " + " | ".join(f"{x:.4f}" for x in sc) + " | " + " | ".join(f"{x:.4f}" for x in se), flush=True)
        del c; torch.cuda.empty_cache()
