# TP4 sparse-GQA shape (6 q / 1 kv heads, BLOCK_M=8): BLOCK_N 64 (main) vs 32 (ungated) vs branch.
import importlib.util, sys, torch
src = open("/work/bench_gqa.py").read().split('print("shape')[0]
src = src.replace('stock, pr3 = load("/work/stock.py", "stock"), load("/work/pr3.py", "pr3")', "")
exec(src)
mods = {"main (BLOCK_N 64)": load("/work/stock.py", "stock"),
        "ungated (BLOCK_N 32)": load("/work/pr3_ungated.py", "ungated"),
        "branch (gated)": load("/work/pr3.py", "pr3")}
print("arch", torch.cuda.get_device_properties(0).gcnArchName)
for rows in (520, 4096):
    a = case(rows, 6, 1)
    ref = mods["main (BLOCK_N 64)"].qsa_sparse_paged_attention(*a).float()
    for rep in range(3):
        cells = []
        for n, m in mods.items():
            diff = (m.qsa_sparse_paged_attention(*a).float() - ref).abs().max().item()
            cells.append(f"{n} {bench(lambda: m.qsa_sparse_paged_attention(*a)):.3f} ms (diff {diff:.1e})")
        print(f"TP4 6q/1kv rows {rows} rep {rep}: " + " | ".join(cells), flush=True)
