# fp8 e4m3 q + cache (as #61529 would pass) vs bf16 run of the same values.
import importlib.util, torch
s = importlib.util.spec_from_file_location("f", "/work/final.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
torch.manual_seed(0)
dt = torch.float8_e4m3fn
rows, H, D, P, W = 49, 4, 128, 16, 24
q8 = torch.randn(rows, H, D, device="cuda").to(dt); k8 = torch.randn(80, P, 1, D, device="cuda").to(dt)
pt = torch.randperm(80, device="cuda", dtype=torch.int32)[: 3 * W].view(3, W)
t2r = torch.tensor([0] * 43 + [1] * 3 + [2] * 3, device="cuda", dtype=torch.int32)
pos = torch.tensor(list(range(1160, 1203)) + [159, 160, 161, 0, 1, 2], device="cuda")
L = torch.tensor([1203, 162, 3], device="cuda", dtype=torch.int32)
a, va = m.qsa_mqa_paged(q8, k8, pt, t2r, pos, L, 4)
b, vb = m.qsa_mqa_paged(q8.bfloat16(), k8.bfloat16(), pt, t2r, pos, L, 4)
mask = torch.arange(a.shape[1], device="cuda")[None] < va[:, None]
print("fp8 compiled; visible equal:", torch.equal(va, vb), "max abs diff vs bf16:", (a[mask] - b[mask]).abs().max().item())
