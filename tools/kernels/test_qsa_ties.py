import importlib.util, torch, sys
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("t", "/patch/test_qsa.py")
src = open("/patch/test_qsa.py").read().split("cases = {")[0]
exec(src)
c = make_case([(32000, list(range(32000)))])
so1 = run_select(orig, c); so2 = run_select(orig, c)
print("orig vs orig identical rows:", (so1 == so2).all(1).float().mean().item())
patched._QSA_MQA_IMPL = "v1"; sp = run_select(patched, c)
a, b = torch.sort(so1, 1).values, torch.sort(sp, 1).values
diff = (~(a == b).all(1)).nonzero().flatten()
print("differing rows:", diff.numel(), "first:", diff[:8].tolist())
# for differing rows: are the swapped tokens ties in the score?
lo, vo = orig.qsa_mqa_paged(c["q"], c["k_cache"], c["page_table"], c["token_to_req"], c["query_positions"], c["sequence_lengths"], CR) if False else (None, None)
for r in diff[:5].tolist():
    sa, sb = set(so1[r].tolist()) - {-1}, set(sp[r].tolist()) - {-1}
    only_a, only_b = sorted(sa - sb)[:4], sorted(sb - sa)[:4]
    rs = slice(r, r + 1)
    l, v = orig.qsa_mqa_paged(c["q"][rs], c["k_cache"], c["page_table"], c["token_to_req"][rs], c["query_positions"][rs], c["sequence_lengths"], CR)
    # token idx -> compressed block idx = tok // CR
    sc = lambda toks: [round(l[0, t // CR].item(), 6) for t in toks]
    print(f"row {r} visible {v.item()} |A-B|={len(sa-sb)} A-only {only_a} scores {sc(only_a)} | B-only {only_b} scores {sc(only_b)}")
