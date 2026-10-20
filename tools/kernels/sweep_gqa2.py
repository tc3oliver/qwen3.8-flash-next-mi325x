"""Isolated sweep of the sparse-GQA prefill launch profile: each config runs in its own process so a
GPU fault in one candidate cannot take the sweep down.  python3 /patch/sweep_gqa2.py"""
import itertools, json, subprocess, sys
CHILD = r'''
import importlib.util, sys, time, torch, json
spec = importlib.util.spec_from_file_location("p", "/patch/qsa.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
heads, kvh, cfg = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dev = "cuda"; torch.manual_seed(0)
rows, seq, page, topk, D = 4096, 32000, 1600, 2051, 256
pages = (seq + page - 1) // page + 2
k = (torch.randn(pages, page, kvh, D, device=dev) * .3).bfloat16(); v = torch.randn_like(k)
q = (torch.randn(rows, heads, D, device=dev) * .3).bfloat16()
bt = torch.arange(pages, device=dev, dtype=torch.int32)[None, :].contiguous()
g = torch.Generator(device="cpu").manual_seed(1)
pos = torch.arange(seq - rows, seq)
far = torch.stack([torch.randperm(int(p) - 511, generator=g)[: topk - 512].sort().values for p in pos])
local = pos[:, None] - 511 + torch.arange(512)[None, :]
idx = torch.cat([far, local], 1).int().to(dev)
t2r = torch.zeros(rows, device=dev, dtype=torch.int32)
args = (q, k, v, idx, bt, t2r)
m._QSA_GQA_PREFILL = None
ref = m.qsa_sparse_paged_attention(*args).float()
if cfg != "stock":
    m._QSA_GQA_PREFILL = tuple(int(x) for x in cfg.split(","))
o = m.qsa_sparse_paged_attention(*args).float()
f = lambda: m.qsa_sparse_paged_attention(*args)
f(); torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): f()
torch.cuda.synchronize()
print("RES", json.dumps({"ms": (time.perf_counter() - t) / 10 * 1e3, "err": (o - ref).abs().max().item()}))
'''
open("/tmp/child.py", "w").write(CHILD)
cfgs = ["stock"] + [f"{bn},{sp},{w},{st}" for bn, sp, w, st in itertools.product((32, 64, 128), (1,), (2, 4, 8), (1, 2))]
for name, (h, kv) in {"TP2": (12, 1), "TP1": (24, 2)}.items():
    res = []
    for c in cfgs:
        p = subprocess.run([sys.executable, "/tmp/child.py", str(h), str(kv), c], capture_output=True, text=True, timeout=600)
        line = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
        if line:
            r = json.loads(line[0][4:]); res.append((r["ms"], c, r["err"]))
        else:
            res.append((float("inf"), c, "FAIL rc=%d %s" % (p.returncode, (p.stderr.strip().splitlines() or [""])[-1][:80])))
        print(name, c, res[-1][0], res[-1][2], flush=True)
    base = [r for r in res if r[1] == "stock"][0][0]
    print(f"== {name} stock {base:.2f} ms; best:")
    for r in sorted(res)[:5]:
        print(f"   {r[0]:.2f} ms ({base / r[0]:.2f}x) cfg(bn,splits,warps,stages)={r[1]} err={r[2]}")
