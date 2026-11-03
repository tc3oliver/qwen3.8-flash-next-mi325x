# fingerprint of top-k output order for the failing case, one process. usage: fp.py [k_pre_kernels]
import sys, torch
sys.path.insert(0, "/t")
import test_qsa_kernels as T
from vllm.models.qwen4_exp.amd.ops import qsa as Q
Q._on_gfx942 = lambda: True
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
q, kc, pt, t2r, pos, lens, topk, cr = T._decode_select_case(8, "bf16")
sel = lambda srd: Q.qsa_select_paged_tokens(q, kc, pt, t2r, pos, lens, topk, cr, single_row_decode=srd)
sel(True); torch.accelerator.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): cap = sel(True)
torch.manual_seed(1); q.copy_(torch.randn_like(q))
nl = torch.randint(8, 1500, lens.shape, device="cuda", dtype=lens.dtype)
lens.copy_(nl); pos.copy_((nl - 1).to(torch.int64))
x = torch.zeros(1, device="cuda")
for _ in range(k): x += 1          # k trivial dispatches before the top-k call
tiled = sel(False).clone()
g.replay(); torch.accelerator.synchronize()
r = sel(True).clone()
f = lambda t: "".join("A" if t[2, c] == v else "?" for c, v in zip(range(4, 12, 4), t[2, 4:12:4].tolist()))
import os
print("FP pid=%d" % os.getpid(), end=" ")
print("tiled_row2=%s rows_row2=%s graph_row2=%s equal(graph,tiled)=%d equal(eager_rows,tiled)=%d" % (
    tiled[2, 4:12:4].tolist(), r[2, 4:12:4].tolist(), cap[2, 4:12:4].tolist(), int((cap == tiled).all()), int((r == tiled).all())))
