# in one process: order of row 2 (first block of the pair) after different idle gaps before the top-k call
import sys, time, torch
sys.path.insert(0, "/t")
import test_qsa_kernels as T
from vllm.models.qwen4_exp.amd.ops import qsa as Q
Q._on_gfx942 = lambda: True
q, kc, pt, t2r, pos, lens, topk, cr = T._decode_select_case(8, "bf16")
sel = lambda srd: Q.qsa_select_paged_tokens(q, kc, pt, t2r, pos, lens, topk, cr, single_row_decode=srd)
sel(True); sel(False); torch.accelerator.synchronize()
torch.manual_seed(1); q.copy_(torch.randn_like(q))
nl = torch.randint(8, 1500, lens.shape, device="cuda", dtype=lens.dtype)
lens.copy_(nl); pos.copy_((nl - 1).to(torch.int64))
for gap in [0, 0.01, 0.1, 0.5, 2.0]:
    n = 30 if gap < 1 else 12
    s = ""
    for _ in range(n):
        time.sleep(gap)
        r = sel(False)
        torch.accelerator.synchronize()
        s += "a" if r[2, 4].item() == 620 else "b"
    print(f"gap={gap}s n={n} first-block-of-pair: {s}  a={s.count('a')} b={s.count('b')}")
