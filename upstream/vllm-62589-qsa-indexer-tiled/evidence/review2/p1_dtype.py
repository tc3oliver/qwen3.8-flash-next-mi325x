# Point 1: dot dtype. Accuracy of main / branch (bf16 dot) / variant (dtype-selected dot)
# against an fp32 torch reference of the same input values, per input dtype; bf16 speed.
import importlib.util, math, time, torch
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
mods = {"main": load("/work/stock.py", "main"), "branch": load("/work/final.py", "branch"), "variant": load("/work/dtype.py", "variant")}
print("arch", torch.cuda.get_device_properties(0).gcnArchName)

def ref(q, k, pt, t2r, vis):
    pages = pt.index_select(0, t2r.long()).long()
    keys = k[pages, :, 0, :].flatten(1, 2)
    s = torch.einsum("rhd,rnd->rnh", q.double(), keys.double())
    return (torch.relu(s).sum(-1) / math.sqrt(q.shape[-1]))

torch.manual_seed(0)
H, D, P, W = 4, 128, 16, 24
t2r = torch.tensor([0] * 43 + [1] * 3 + [2] * 3, device="cuda", dtype=torch.int32)
pos = torch.tensor(list(range(1160, 1203)) + [159, 160, 161, 0, 1, 2], device="cuda")
L = torch.tensor([1203, 162, 3], device="cuda", dtype=torch.int32)
pt = torch.randperm(80, device="cuda", dtype=torch.int32)[: 3 * W].view(3, W)
q32 = torch.randn(49, H, D, device="cuda"); k32 = torch.randn(80, P, 1, D, device="cuda")
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp8e4m3fn", torch.float8_e4m3fn)]:
    q, k = q32.to(dt), k32.to(dt)
    r = None
    for mn, m in mods.items():
        try:
            lg, vis = m.qsa_mqa_paged(q, k, pt, t2r, pos, L, 4)
        except Exception as e:
            print(f"{name:10s} {mn:8s} ERROR {type(e).__name__}: {str(e)[:120]}"); continue
        if r is None:
            r = ref(q.float() if dt != torch.float8_e4m3fn else q.float(), k.float(), pt, t2r, vis)
        mask = torch.arange(lg.shape[1], device="cuda")[None] < vis[:, None]
        err = (lg.double()[mask] - r[mask]).abs()
        rel = (err / r[mask].abs().clamp_min(1e-3))
        print(f"{name:10s} {mn:8s} max abs err {err.max().item():.2e}  max rel err {rel.max().item():.2e}  (|ref| max {r[mask].abs().max().item():.1f})", flush=True)

# bf16 speed: branch vs variant, real shapes
src = open("/work/harness.py").read().split("cases = {")[0]
src = src.replace('orig = load("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py", "qsa_orig")', "orig = None").replace('patched = load("/patch/qsa.py", "qsa_patched")', "patched = None")
exec(src)
torch.manual_seed(1)
cases = {"prefill 32K": [(32000, list(range(32000)))],
         "decode 64x4": [(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 200000, (64,)).tolist()]}
for cn, reqs in cases.items():
    c = make_case(reqs); it = 10 if c["q"].shape[0] > 1000 else 100
    for rep in range(3):
        print(f"bf16 speed {cn} rep{rep}: " + " | ".join(f"{mn} {bench(lambda: run_select(mods[mn], c), it):.3f} ms" for mn in ("branch", "variant")), flush=True)
    del c; torch.cuda.empty_cache()
