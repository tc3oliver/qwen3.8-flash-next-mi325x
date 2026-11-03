import importlib.util, sys, torch, struct
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
src = open("/work/harness.py").read().split("cases = {")[0]
src = src.replace('orig = load("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py", "qsa_orig")', 'orig = load("/work/nightly/stock_qsa.py", "stock")').replace('patched = load("/patch/qsa.py", "qsa_patched")', 'patched = load("/work/nightly/pr_qsa.py", "pr")')
exec(src)
ctx, n, r = 98304, 64, 55
torch.manual_seed(n)
c = make_case([(ctx, [ctx - 1])] * n)
args = (c["q"], c["k_cache"], c["page_table"], c["token_to_req"], c["query_positions"], c["sequence_lengths"], CR)
ls, vs = orig.qsa_mqa_paged(*args); lp, vp = patched.qsa_mqa_paged(*args)
a = run_select(orig, c)[r]; b = run_select(patched, c)[r]
A = sorted(set((a[a >= 0] // CR).tolist())); B = sorted(set((b[b >= 0] // CR).tolist()))
vis = int(vs[r]); print("visible", vis, int(vp[r]), "selected", len(A), len(B), "k=", TOPK // CR)
print("logits row bitwise equal over visible:", torch.equal(ls[r, :vis], lp[r, :vis]), "max abs diff", (ls[r,:vis]-lp[r,:vis]).abs().max().item(),
      "n differing cols", int((ls[r,:vis] != lp[r,:vis]).sum()))
D = sorted(set(A) ^ set(B)); print("only stock:", [x for x in A if x not in B], "only PR:", [x for x in B if x not in A])
bits = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
def show(tag, blocks):
    for blk in blocks:
        s, p = ls[r, blk].item(), lp[r, blk].item()
        print(f"  {tag} blk {blk}: stock {s!r} ({s.hex()} bits {bits(s):#010x}) PR {p!r} (bits {bits(p):#010x}) ulp_apart {abs(bits(s)-bits(p))}")
show("diff", D)
# rank k boundary according to each kernel's own logits
for name, L in (("stock", ls), ("PR", lp)):
    v = L[r, :vis]; srt, idx = torch.sort(v, descending=True)
    k = TOPK // CR
    print(name, "rank k-3..k+3 (0-based) value/block:", [(i, srt[i].item(), idx[i].item()) for i in range(k - 3, k + 3)])
    print(name, "count of columns with value == kth value:", int((v == srt[k-1]).sum()), "kth value", srt[k-1].item(), "k+1th", srt[k].item())
    print(name, "own selection == own top-k set?", set(idx[:k].tolist()) == set(A if name=="stock" else B), "(ties may break either way)")
print("selected-by-stock ranks in stock sort of diff blocks:", {blk: int((ls[r,:vis] > ls[r,blk]).sum()) for blk in D})
print("selected-by-PR   ranks in PR sort of diff blocks:", {blk: int((lp[r,:vis] > lp[r,blk]).sum()) for blk in D})
