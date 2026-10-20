# Point 3: branch qsa_mqa_paged -> real ops.top_k_per_row_decode on ROCm, with logits past
# visible_blocks poisoned (NaN, +inf, 1e30, and the raw torch.empty garbage left by the kernel).
# The selection must equal top-k over [0, visible) of the same logits (exact, tie-safe).
import importlib.util, torch
from vllm import _custom_ops as ops
s = importlib.util.spec_from_file_location("b", "/work/final.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("arch", torch.cuda.get_device_properties(0).gcnArchName)
torch.manual_seed(0)
H, D, P, W, CR, K = 4, 128, 400, 164, 4, 512          # real indexer shapes, block top-k 512
# rows: prefill-chunk tail, decode rows of various lengths incl. visible 0, <K, =K, =K+1, long
reqs = [(98304, list(range(98304 - 64, 98304))), (3, [2]), (1000, [999]), (2048, [2047]),
        (2052, [2051]), (2056, [2055]), (150001, [150000]), (262000, [261999])]
n = len(reqs)
k = (torch.randn(n * W + 8, P, 1, D, device="cuda") * 0.5).bfloat16()
pt = (torch.randperm(n * W, device="cuda").int() + 8).view(n, W)
t2r = torch.tensor([i for i, (_, p) in enumerate(reqs) for _ in p], device="cuda", dtype=torch.int32)
pos = torch.tensor([x for _, p in reqs for x in p], device="cuda")
L = torch.tensor([r[0] for r in reqs], device="cuda", dtype=torch.int32)
q = (torch.randn(t2r.numel(), H, D, device="cuda") * 0.5).bfloat16()
# Pre-poison the caching allocator so torch.empty inside qsa_mqa_paged returns NaN-filled memory.
junk = torch.full((q.shape[0], W * P), float("nan"), device="cuda"); del junk
logits, vis = m.qsa_mqa_paged(q, k, pt, t2r, pos, L, CR)
cols = torch.arange(logits.shape[1], device="cuda")[None]
past = cols >= vis[:, None]
print("visible per row (unique):", sorted(set(vis.tolist())), "| capacity", logits.shape[1])
print("raw torch.empty region past visible: NaN fraction %.3f" % torch.isnan(logits[past]).float().mean().item())
clean = logits.masked_fill(past, float("-inf"))
def check(name, lg):
    idx = torch.full((lg.shape[0], K), -777, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_decode(lg, 1, vis, idx, lg.shape[0], lg.stride(0), lg.stride(1), K)
    torch.cuda.synchronize()
    ends = vis[:, None].long()
    valid = (idx >= 0) & (idx < ends)
    ok_count = torch.equal(valid.sum(1), ends.clamp_max(K).flatten())
    ok_rest = bool(torch.all(valid | (idx == -1)))
    srt = idx.sort(1).values
    ok_unique = not bool(((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any())
    exp = clean.topk(K, dim=1).values                         # -inf fill where visible < K
    sel = clean.gather(1, idx.clamp_min(0).long()).masked_fill(~valid, float("-inf")).sort(1, descending=True).values
    ok_vals = torch.equal(sel, exp)
    out_of_range = int(((idx >= ends) ).sum())
    print(f"{name:28s} count_ok={ok_count} only_valid_or_-1={ok_rest} unique={ok_unique} "
          f"selected_scores==topk_over_visible={ok_vals} indices>=visible={out_of_range}", flush=True)
check("raw kernel output (NaN)", logits.clone())
for nm, val in (("NaN past visible", float("nan")), ("+inf past visible", float("inf")), ("1e30 past visible", 1e30)):
    check(nm, logits.masked_fill(past, val))
check("clean (-inf) control", clean.clone())
# Negative control: if top-k were told rows extend to capacity, the 1e30 poison must be picked.
full = torch.full_like(vis, logits.shape[1])
idx = torch.full((logits.shape[0], K), -777, dtype=torch.int32, device="cuda")
lg = logits.masked_fill(past, 1e30)
ops.top_k_per_row_decode(lg, 1, full, idx, lg.shape[0], lg.stride(0), lg.stride(1), K)
torch.cuda.synchronize()
print(f"negative control (seq_lens=capacity, 1e30 poison): indices>=visible={int((idx >= vis[:, None]).sum())} -> check detects reads past visible")
