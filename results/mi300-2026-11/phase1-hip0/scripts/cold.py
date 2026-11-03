# does cache/TLB coldness before the launch change the order? count of '620 first' (a) vs '1388 first' (b)
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
big = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")     # 2 GiB > L2 + MALL(256 MiB)
a = torch.randn(4096, 4096, device="cuda"); 
def evict(): big.fill_(1)
def other(): (a @ a).sum().item()
def run(name, pre, fn, n=40):
    c = 0
    for _ in range(n):
        pre(); r = fn(); torch.accelerator.synchronize(); c += int(r[2, 4].item() == 620)
    print(f"COLD pid={os.getpid()} {name}: a(620 first)={c}/{n}")
nop = lambda: None
for nm, fn in (("tiled", lambda: sel(False)), ("graph", lambda: (g.replay(), cap)[1])):
    run(nm + " back-to-back", nop, fn); run(nm + " after 2GiB evict", evict, fn); run(nm + " after matmul", other, fn)
