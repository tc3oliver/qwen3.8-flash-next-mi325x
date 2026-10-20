# 1x1 decode spread: main vs final, 5 repeats, 3 contexts.
src = open("/work/p2_decode.py").read().split('print("context')[0]
exec(src)
for ctx in (8192, 98304, 200000):
    c = make_case([(ctx, [ctx - 1])])
    for rep in range(5):
        a = ev(lambda: run_select(mods["main"], c), 500); b = ev(lambda: run_select(mods["branch"], c), 500)
        print(f"1x1 ctx {ctx} rep{rep}: main {a*1000:.1f} us | final {b*1000:.1f} us | delta {1000*(b-a):+.1f} us", flush=True)
