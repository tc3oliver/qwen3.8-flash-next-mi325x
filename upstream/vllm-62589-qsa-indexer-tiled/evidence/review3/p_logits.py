# Logits accuracy of the final kernel vs an fp64 torch reference, per input dtype.
import importlib.util, math, sys, torch
s = importlib.util.spec_from_file_location("f", sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("arch", torch.cuda.get_device_properties(0).gcnArchName, "| kernel", sys.argv[1])
torch.manual_seed(0)
H, D, P, W = 4, 128, 16, 24
t2r = torch.tensor([0] * 43 + [1] * 3 + [2] * 3, device="cuda", dtype=torch.int32)
pos = torch.tensor(list(range(1160, 1203)) + [159, 160, 161, 0, 1, 2], device="cuda")
L = torch.tensor([1203, 162, 3], device="cuda", dtype=torch.int32)
pt = torch.randperm(80, device="cuda", dtype=torch.int32)[: 3 * W].view(3, W)
q32 = torch.randn(49, H, D, device="cuda"); k32 = torch.randn(80, P, 1, D, device="cuda")
for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp8e4m3fn", torch.float8_e4m3fn)]:
    q, k = q32.to(dt), k32.to(dt)
    lg, vis = m.qsa_mqa_paged(q, k, pt, t2r, pos, L, 4)
    pages = pt.index_select(0, t2r.long()).long()
    keys = k.float()[pages, :, 0, :].flatten(1, 2)
    r = torch.relu(torch.einsum("rhd,rnd->rnh", q.double(), keys.double())).sum(-1) / math.sqrt(D)
    mask = torch.arange(lg.shape[1], device="cuda")[None] < vis[:, None]
    print(f"{name:10s} max abs err vs fp64 reference {(lg.double()[mask] - r[mask]).abs().max().item():.2e} (|ref| max {r[mask].abs().max().item():.1f})", flush=True)
