# Is the mismatch only a permutation of the selected blocks? Does order vary across eager calls too?
import sys, torch
sys.path.insert(0, "/t")
import test_qsa_kernels as T
from vllm.models.qwen4_exp.amd.ops import qsa as Q
Q._on_gfx942 = lambda: True
q, kc, pt, t2r, pos, lens, topk, cr = T._decode_select_case(8, "bf16")
sel = lambda srd: Q.qsa_select_paged_tokens(q, kc, pt, t2r, pos, lens, topk, cr, single_row_decode=srd)
sel(True); torch.accelerator.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): cap = sel(True)
torch.manual_seed(1); q.copy_(torch.randn_like(q))
nl = torch.randint(8, 1500, lens.shape, device="cuda", dtype=lens.dtype)
lens.copy_(nl); pos.copy_((nl - 1).to(torch.int64))
tiled = sel(False).clone()
srt = lambda x: x.sort(dim=1).values
N = 500
def stats(name, fn):
    exact = sameset = 0; orders = set()
    for _ in range(N):
        r = fn().clone(); torch.accelerator.synchronize()
        exact += int((r == tiled).all()); sameset += int((srt(r) == srt(tiled)).all()); orders.add(tuple(r[2].tolist()))
    print(f"{name}: exact_equal={exact}/{N} sorted_equal={sameset}/{N} distinct_row2_orders={len(orders)}")
stats("graph_replay(rows kernel)", lambda: (g.replay(), cap)[1])
stats("eager rows kernel", lambda: sel(True))
stats("eager tiled kernel", lambda: sel(False))
