import torch
src = open("/patch/test_qsa.py").read().split("cases = {")[0]; exec(src)
for n, hi in ((1, 60000), (4, 120000), (16, 120000), (64, 200000)):
    c = make_case([(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, hi, (n,)).tolist()])
    r = [bench(lambda: run_select(orig, c), 20)]
    patched._QSA_MQA_IMPL = "v1"; patched._QSA_V1_BLOCK_N = 128; r.append(bench(lambda: run_select(patched, c), 20))
    patched._QSA_MQA_IMPL = "v2"; patched._QSA_V2_BLOCK_M, patched._QSA_V2_BLOCK_N, patched._QSA_V2_WARPS = 16, 64, 4; r.append(bench(lambda: run_select(patched, c), 20))
    print(f"decode {n:3d} req x4: orig {r[0]:.3f}  v1/128 {r[1]:.3f}  v2/16x64 {r[2]:.3f} ms")
