import math, torch, json, sys
from vllm.models.qwen4_exp.amd.ops import qsa
import vllm.platforms.rocm as rocm
from vllm.platforms import current_platform

p = torch.cuda.get_device_properties(0)
print("ARCH", p.gcnArchName, "CUs", p.multi_processor_count, "on_gfx942", rocm.on_gfx942(), "is_rocm", current_platform.is_rocm())
print("qsa.py threshold: `base_programs <= 512` (overlay/qsa.py:1065); BLOCK_N=32 iff rows*kv_heads>512 and on_gfx942 and BLOCK_M>=16 (:1074)")

REAL_ON = rocm.on_gfx942
real_kernel = qsa._qsa_sparse_paged_gqa_splitk_kernel
rec = {}
class Spy:
    def __getitem__(self, grid):
        def call(*a, **kw):
            rec.clear(); rec.update({k: kw[k] for k in ("BLOCK_M", "BLOCK_N", "NUM_SPLITS", "NUM_TILES", "num_warps", "num_stages")}); rec["grid"] = tuple(grid)
            return real_kernel[grid](*a, **kw)
        return call
qsa._qsa_sparse_paged_gqa_splitk_kernel = Spy()

def case(rows, heads, kvh, seq=32000, page=1600, topk=2051, D=256, fp8=False):
    torch.manual_seed(0)
    pages = (seq + page - 1) // page + 2
    k = (torch.randn(pages, page, kvh, D, device="cuda") * .3)
    v = torch.randn_like(k)
    q = (torch.randn(rows, heads, D, device="cuda") * .3)
    if fp8:  # quantize through fp8 then store as bf16: kernel only accepts bf16 (assert in qsa_sparse_paged_attention)
        f8 = current_platform.fp8_dtype()
        k, v, q = (t.to(f8).to(torch.bfloat16) for t in (k, v, q))
    else:
        k, v, q = (t.bfloat16() for t in (k, v, q))
    bt = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :].contiguous()
    g = torch.Generator().manual_seed(1)
    pos = torch.arange(seq - rows, seq)
    far = torch.stack([torch.randperm(int(x) - 511, generator=g)[: topk - 512].sort().values for x in pos])
    local = pos[:, None] - 511 + torch.arange(512)[None, :]
    idx = torch.cat([far, local], 1).int().cuda()
    return q, k, v, idx, bt, torch.zeros(rows, device="cuda", dtype=torch.int32)

def reference(q, k, v, idx, page):
    rows, H, D = q.shape; kvh = k.shape[2]; G = H // kvh
    kf = k.float().reshape(-1, kvh, D); vf = v.float().reshape(-1, kvh, D)
    out = torch.empty(rows, H, D, device=q.device)
    for s in range(0, rows, 32):
        i = idx[s:s+32].long()                      # [r, topk]  (block table is identity)
        kk, vv = kf[i], vf[i]                       # [r, topk, kvh, D]
        qq = q[s:s+32].float().view(-1, kvh, G, D)
        sc = torch.einsum("rkgd,rtkd->rkgt", qq, kk) * D ** -0.5
        out[s:s+32] = torch.einsum("rkgt,rtkd->rkgd", sc.softmax(-1), vv).reshape(-1, H, D)
    return out

# (name, q heads, kv heads, rows); TP1=24/2, TP2=12/1, TP4=6/1
shapes = [("TP1", 24, 2, 256), ("TP1", 24, 2, 257), ("TP1", 24, 2, 1024),
          ("TP2", 12, 1, 512), ("TP2", 12, 1, 513), ("TP2", 12, 1, 1024),
          ("TP4", 6, 1, 512), ("TP4", 6, 1, 513), ("TP4", 6, 1, 1024)]
bad = 0
print("\n| dtype | shape | rows | kvh | programs | on_gfx942 seen | BLOCK_M | BLOCK_N | splits | warps | stages | max abs diff vs fp32 ref | max abs diff vs stock-dispatch(BLOCK_N=64 path) | note |")
print("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
for fp8 in (False, True):
    for name, H, kvh, rows in shapes:
        a = case(rows, H, kvh, fp8=fp8)
        ref = reference(a[0], a[1], a[2], a[3], a[1].shape[1])
        res = {}
        for mode in ("real", "force64"):
            rocm.on_gfx942 = REAL_ON if mode == "real" else (lambda: False)
            try:
                o = qsa.qsa_sparse_paged_attention(*a); torch.cuda.synchronize()
                res[mode] = (o, dict(rec), None)
            except Exception as e:
                res[mode] = (None, dict(rec), repr(e)); bad += 1
        rocm.on_gfx942 = REAL_ON
        for mode in ("real", "force64"):
            o, r, err = res[mode]
            if err:
                print(f"| {'fp8' if fp8 else 'bf16'} | {name} | {rows} | {kvh} | {rows*kvh} | {mode} | - | - | - | - | - | ERROR | - | {err} |"); continue
            d = (o.float() - ref).abs().max().item()
            d2 = (o.float() - res['force64'][0].float()).abs().max().item() if res['force64'][0] is not None else float('nan')
            nan = torch.isnan(o.float()).any().item()
            ok = d < 2e-2 and not nan
            bad += (not ok)
            print(f"| {'fp8' if fp8 else 'bf16'} | {name} | {rows} | {kvh} | {rows*kvh} | {'real(' + str(REAL_ON()) + ')' if mode=='real' else 'forced False'} | {r['BLOCK_M']} | {r['BLOCK_N']} | {r['NUM_SPLITS']} | {r['num_warps']} | {r['num_stages']} | {d:.3e} | {d2:.3e} | {'PASS' if ok else 'FAIL'}{' NaN' if nan else ''} (tol 2e-2) |")
        sys.stdout.flush()
        del a, ref, res; torch.cuda.empty_cache()
print("\nFAILURES", bad)
sys.exit(1 if bad else 0)
