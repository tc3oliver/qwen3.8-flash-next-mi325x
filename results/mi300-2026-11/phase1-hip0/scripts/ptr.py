# per top-k call: logits/out/seq_lens data_ptr and the resulting order of row 2's first block (a=620 first, b=1388 first)
import sys, os, torch
sys.path.insert(0, "/t")
import test_qsa_kernels as T
from vllm.models.qwen4_exp.amd.ops import qsa as Q
Q._on_gfx942 = lambda: True
rec = []
orig = Q.ops.top_k_per_row_decode
def wrap(logits, next_n, seq, out, nrows, s0, s1, k):
    orig(logits, next_n, seq, out, nrows, s0, s1, k); torch.accelerator.synchronize()
    rec.append((logits.data_ptr(), out.data_ptr(), logits.shape[1], s0, "a" if out[2, 0].item() == 155 else "?", out[2].tolist()))
Q.ops.top_k_per_row_decode = wrap
q, kc, pt, t2r, pos, lens, topk, cr = T._decode_select_case(8, "bf16")
sel = lambda srd: Q.qsa_select_paged_tokens(q, kc, pt, t2r, pos, lens, topk, cr, single_row_decode=srd)
torch.manual_seed(1); q.copy_(torch.randn_like(q))
nl = torch.randint(8, 1500, lens.shape, device="cuda", dtype=lens.dtype)
lens.copy_(nl); pos.copy_((nl - 1).to(torch.int64))
for srd in (False, True, False, True, False, False, False, False):
    sel(srd)
for i, (lp, op, w, s0, _, o) in enumerate(rec):
    print(f"pid={os.getpid()} call{i} logits_ptr=0x{lp:x} out_ptr=0x{op:x} cols={w} stride0={s0} out_row2={o}")
