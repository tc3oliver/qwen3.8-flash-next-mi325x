# Same inputs as the test; compare graph replay / eager per-row / tiled / torch reference. Dumps on mismatch.
import sys, math, json, torch
sys.path.insert(0, "/t")
import test_qsa_kernels as T
from vllm.models.qwen4_exp.amd.ops import qsa as Q

iters = int(sys.argv[1]); out = sys.argv[2]
orig_on = Q._on_gfx942
Q._on_gfx942 = lambda: True
q, kc, pt, t2r, pos, lens, topk, cr = T._decode_select_case(8, "bf16")
sel = lambda srd: Q.qsa_select_paged_tokens(q, kc, pt, t2r, pos, lens, topk, cr, single_row_decode=srd)
sel(True); torch.accelerator.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    cap = sel(True)
torch.manual_seed(1)
q.copy_(torch.randn_like(q))
nl = torch.randint(8, 1500, lens.shape, device="cuda", dtype=lens.dtype)
lens.copy_(nl); pos.copy_((nl - 1).to(torch.int64))
print("lengths", lens.tolist(), "visible_blocks", (lens // cr).tolist())
tiled = sel(False).clone()
eager = sel(True).clone()
# float reference logits per row
ref = T._qsa_mqa_paged_reference(q, kc, pt, t2r, (lens // cr).to(torch.int32))
lg_rows, vb = Q.qsa_mqa_paged(q, kc, pt, t2r, pos, lens, cr, single_row_decode=True)
lg_tile, _ = Q.qsa_mqa_paged(q, kc, pt, t2r, pos, lens, cr, single_row_decode=False)
vbl = vb.tolist()
bad = 0; res = []
for it in range(iters):
    g.replay(); torch.accelerator.synchronize()
    r = cap.clone()
    ne = (r != tiled).nonzero().tolist()
    ne_eager = int((eager != tiled).sum()); ne_rt = int((r != eager).sum())
    res.append(len(ne))
    if ne and bad < 3:
        bad += 1
        rows = sorted({a for a, _ in ne})
        info = {"iter": it, "mismatch_pos": ne, "eager_vs_tiled": ne_eager, "graph_vs_eager": ne_rt}
        for row in rows:
            n = vbl[row]
            info[f"row{row}"] = {
                "visible_blocks": n, "length": int(lens[row]),
                "graph": r[row].tolist(), "tiled": tiled[row].tolist(), "eager": eager[row].tolist(),
                "top8_ref": [(i, float(v)) for v, i in zip(*[t.tolist() for t in ref[row, :n].topk(min(8, n))])],
                "rows_kernel_logits_top8": [(i, float(v)) for v, i in zip(*[t.tolist() for t in lg_rows[row, :n].topk(min(8, n))])],
                "tiled_kernel_logits_top8": [(i, float(v)) for v, i in zip(*[t.tolist() for t in lg_tile[row, :n].topk(min(8, n))])],
                "rows_vs_tiled_logits_maxabs": float((lg_rows[row, :n] - lg_tile[row, :n]).abs().max()),
                "n_zero_logits": int((lg_rows[row, :n] == 0).sum()),
            }
        json.dump(info, open(f"{out}/fail{bad}.json", "w"), indent=1)
print("eager_vs_tiled mismatches:", int((eager != tiled).sum()))
print("graph replays:", iters, "with mismatch:", sum(1 for x in res if x), "counts:", sorted(set(res)))
