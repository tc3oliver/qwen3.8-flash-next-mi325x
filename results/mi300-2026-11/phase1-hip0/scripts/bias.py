# per-process bias: fraction of calls where block 155 precedes 347 in row 2, for three launch kinds, 200 calls each
import sys, os, torch
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
def frac(fn, n=200):
    c = 0
    for _ in range(n):
        r = fn(); torch.accelerator.synchronize()
        c += int(r[2, 4].item() == 620)   # 620 block (=155*4) first
    return c
t = frac(lambda: sel(False)); r = frac(lambda: sel(True)); gg = frac(lambda: (g.replay(), cap)[1])
print(f"BIAS pid={os.getpid()} tiled={t}/200 rows={r}/200 graph={gg}/200")
