"""python3 -I agg.py -> stock vs overlay, per-round verdict.
Attribution limit: the overlay = #62589 + #62590 + the indexer_qsa backport, so the delta cannot be attributed to a single PR.
Design: each round runs both arms concurrently, one per pair; R1 stock@A, R2 stock@B (swapped), R3 stock@A.
Headline = per-round overlay-vs-stock ratios (pooled medians carry pair bias). p2 vs p3 differ by up to ~5%: smaller differences are not resolvable."""
import glob, json, os, re, statistics as st
U = os.path.dirname(os.path.abspath(__file__))
M = [("prefill_8", "prefill_tok_s", "prefill 8x32K tok/s"), ("decode_c64", "out_tok_s", "decode c64 tok/s"),
     ("decode_c64", "tpot_p50_ms", "decode c64 TPOT ms"), ("agent_32k_c16", "wall_s", "agent 32K c16 wall s")]
HI = {"tok/s"}  # higher is better
NOISE = 0.05
ARMS, ROUNDS, PASSES = ("stock", "overlay"), (1, 2, 3), (1, 2, 3)
print(__doc__.split("\n")[1]); print()
data, warn = {}, []  # (arm, round) -> {pass: perf json}
for arm in ARMS:
    for r in ROUNDS:
        for p in PASSES:
            fs = glob.glob(f"{U}/{arm}-r{r}-*.p{p}.perf.json")
            if not fs: warn.append(f"missing {arm} r{r} p{p}"); continue
            data.setdefault((arm, r), {})[p] = json.load(open(fs[0]))
pair = {(a, r): re.search(r"-r\d-(\w)\.p", glob.glob(f"{U}/{a}-r{r}-*.perf.json")[0]).group(1) for (a, r) in data}
def vals(arm, r, t, k, drop1):
    out = []
    for p, d in sorted(data.get((arm, r), {}).items()):
        if drop1 and p == 1: continue
        try: out.append(d[t][k])
        except KeyError: warn.append(f"no {t}.{k} in {arm} r{r} p{p}")
    return out
print("runs:", {a: sorted(r for (x, r) in data if x == a) for a in ARMS}, "passes:", {a: sum(len(v) for (x, _), v in data.items() if x == a) for a in ARMS})
print("pairs:", {f"{a}-r{r}": p for (a, r), p in sorted(pair.items())})
for w in dict.fromkeys(warn): print("WARNING", w)
full = not warn
for drop1 in (False, True):
    print(f"\n=== {'passes 2-3 only (p1 = cold warm-up dropped)' if drop1 else 'all passes'} ===")
    for t, k, n in M:
        hi = "tok/s" in n
        pooled = {a: [x for r in ROUNDS for x in vals(a, r, t, k, drop1)] for a in ARMS}
        f = lambda v: f"{st.median(v):.4g} [{min(v):.4g}-{max(v):.4g}]" if v else "-"
        print(f"{n:22} pooled (pair-biased) stock {f(pooled['stock'])} | overlay {f(pooled['overlay'])}")
        ratios = {}
        for r in ROUNDS:
            s, o = vals("stock", r, t, k, drop1), vals("overlay", r, t, k, drop1)
            if s and o: ratios[r] = st.median(o) / st.median(s) - 1
        print(f"{'':22} per-round overlay vs stock: " + ", ".join(f"R{r} {x:+.1%}" for r, x in ratios.items()))
        if len(ratios) == 3:
            bias = (1 + ratios[2]) / ((1 + ratios[1]) + (1 + ratios[3])) * 2 - 1
            print(f"{'':22} pair-bias estimate (R2 swapped vs mean R1,R3): {bias:+.1%}")
            good = [(x > NOISE) if hi else (x < -NOISE) for x in ratios.values()]
            print(f"{'':22} verdict: " + ("separated: every round better by >5%" if all(good) else "within noise (not all rounds beyond 5%)"))
        else:
            print(f"{'':22} verdict/pair-bias NOT computed: rounds with both arms = {sorted(ratios)}; 'separated' needs all 3")
