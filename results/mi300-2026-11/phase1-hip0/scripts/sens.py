# Downstream sensitivity: qsa_sparse_paged_attention with the same selected set in different orders.
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
# find a graph replay whose order differs from tiled (the flaky case)
for _ in range(2000):
    g.replay(); torch.accelerator.synchronize()
    if (cap != tiled).any(): break
graph = cap.clone()
print("found differing graph order:", bool((graph != tiled).any()), "rows:", (graph != tiled).any(1).nonzero().flatten().tolist())
print("min index", int(tiled.min()), "max", int(tiled.max()))
# attention inputs: head_dim 256, 6 q heads, 1 kv head (repo test shape), page_size 64
rows, hd, ps = tiled.shape[0], 256, 64
npg = (int(tiled.max()) // ps) + 2
torch.manual_seed(5)
qa = torch.randn(rows, 6, hd, device="cuda", dtype=torch.bfloat16)
kv = torch.randn(rows * npg, ps, 1, 2 * hd, device="cuda", dtype=torch.bfloat16)
kca, vca = kv.split(hd, dim=-1)
bt = torch.randperm(rows * npg, device="cuda").reshape(rows, npg).to(torch.int32)
t2 = torch.arange(rows, device="cuda", dtype=torch.int32)
att = lambda idx: Q.qsa_sparse_paged_attention(qa, kca, vca, idx.contiguous(), bt, t2)
base = att(tiled)
def rep(name, idx):
    o = att(idx); d = (o.float() - base.float()).abs()
    print(f"{name}: bitwise_identical={bool(torch.equal(o, base))} max_abs_diff={d.max().item():.3e} max_abs_diff_rows={[round(x,6) for x in d.amax((1,2)).tolist()]}")
rep("graph order (flaky) vs tiled", graph)
rep("tiled again vs tiled", tiled)
gen = torch.Generator(device="cuda").manual_seed(0)
for i in range(5):
    perm = torch.stack([torch.randperm(tiled.shape[1], device="cuda", generator=gen) for _ in range(rows)])
    rep(f"random column permutation #{i}", tiled.gather(1, perm))
print("ref magnitude: max|out| =", base.abs().max().item(), " repo tolerance in test_qsa_sparse_paged_attention_matches_reference: rtol=2e-2 atol=2e-2")
