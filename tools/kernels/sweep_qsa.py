import os, sys, torch
src = open("/patch/test_qsa.py").read().split("cases = {")[0]; exec(src)
pre = make_case([(32000, list(range(32000)))])
tail = make_case([(98304, list(range(98304 - 16384, 98304)))])
dec = make_case([(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 200000, (64,)).tolist()])
dec16 = make_case([(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 120000, (16,)).tolist()])
print("orig: pre %.2f tail %.2f dec64 %.3f dec16 %.3f" % tuple(bench(lambda c=c: run_select(orig, c)) for c in (pre, tail, dec, dec16)))
patched._QSA_MQA_IMPL = "v2"
for bm in (16, 32, 64):
    for bn in (32, 64, 128):
        for w in (4, 8):
            patched._QSA_V2_BLOCK_M, patched._QSA_V2_BLOCK_N, patched._QSA_V2_WARPS = bm, bn, w
            try:
                print(f"v2 bm{bm} bn{bn} w{w}: pre %.2f tail %.2f dec64 %.3f" % tuple(bench(lambda c=c: run_select(patched, c)) for c in (pre, tail, dec)), flush=True)
            except Exception as e:
                print(f"v2 bm{bm} bn{bn} w{w}: ERR {str(e)[:80]}")
patched._QSA_MQA_IMPL = "v1"
for bn in (32, 64, 128, 256):
    patched._QSA_V1_BLOCK_N = bn
    print(f"v1 bn{bn}: pre %.2f dec64 %.3f dec16 %.3f" % tuple(bench(lambda c=c: run_select(patched, c)) for c in (pre, dec, dec16)), flush=True)
