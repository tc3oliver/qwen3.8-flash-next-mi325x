"""Correctness + speed of the patched QSA indexer scoring vs the stock v0.32.0 kernel.
Run inside the vLLM image with the overlay at /patch/qsa.py:  python3 /patch/test_qsa.py"""
import importlib.util, os, sys, time
import torch

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

orig = load("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py", "qsa_orig")
patched = load("/patch/qsa.py", "qsa_patched")
dev = "cuda"
torch.manual_seed(0)
H, D, CR, PAGE, WIDTH, TOPK = 4, 128, 4, 400, 164, 2048  # 164*400 = 65600 columns (262144/4)


def make_case(reqs):
    """reqs: list of (seq_len, query_positions list)"""
    n_req = len(reqs)
    pages_per_req = WIDTH
    num_pages = n_req * pages_per_req + 8
    k_cache = (torch.randn(num_pages, PAGE, 1, D, device=dev) * 0.5).to(torch.bfloat16)
    perm = torch.randperm(num_pages - 8, device=dev)[: n_req * pages_per_req].int() + 8
    page_table = perm.view(n_req, pages_per_req).contiguous()
    tok2req, qpos = [], []
    for i, (sl, pos) in enumerate(reqs):
        tok2req += [i] * len(pos); qpos += pos
    rows = len(qpos)
    q = (torch.randn(rows, H, D, device=dev) * 0.5).to(torch.bfloat16)
    return dict(q=q, k_cache=k_cache, page_table=page_table,
                token_to_req=torch.tensor(tok2req, device=dev, dtype=torch.int32),
                query_positions=torch.tensor(qpos, device=dev, dtype=torch.int64),
                sequence_lengths=torch.tensor([r[0] for r in reqs], device=dev, dtype=torch.int32))


def run_select(mod, c):
    return mod.qsa_select_paged_tokens(c["q"], c["k_cache"], c["page_table"], c["token_to_req"],
                                       c["query_positions"], c["sequence_lengths"], TOPK, CR)


def run_mqa(mod, c, rows=slice(0, 512)):
    return mod.qsa_mqa_paged(c["q"][rows], c["k_cache"], c["page_table"], c["token_to_req"][rows],
                             c["query_positions"][rows], c["sequence_lengths"], CR)


def bench(fn, it=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1000


cases = {
    "prefill_32k": [(32000, list(range(32000)))],
    "prefill_chunk_96k_tail": [(98304, list(range(98304 - 16384, 98304)))],
    "mixed": [(20000, list(range(12000, 20000))), (5000, list(range(5000))), (70000, [69999]), (130000, [129996, 129997, 129998, 129999])],
    "decode_64x4": [(sl, list(range(sl - 4, sl))) for sl in torch.randint(8000, 200000, (64,)).tolist()],
}
impls = [x for x in os.environ.get("IMPLS", "v1,v2").split(",")]
ok_all = True
for name, reqs in cases.items():
    c = make_case(reqs)
    # logits on the first rows (and the last rows) of the case
    for rs in (slice(0, min(512, c["q"].shape[0])), slice(max(0, c["q"].shape[0] - 512), c["q"].shape[0])):
        lo, vo = run_mqa(orig, c, rs)
        for impl in impls:
            patched._QSA_MQA_IMPL = impl
            lp, vp = run_mqa(patched, c, rs)
            assert torch.equal(vo, vp), f"{name} {impl} visible mismatch"
            cols = torch.arange(lo.shape[1], device=dev)[None, :]
            m = cols < vo[:, None].long()
            err = (lo[m] - lp[m]).abs().max().item() if m.any() else 0.0
            rel = err / (lo[m].abs().max().item() + 1e-9) if m.any() else 0.0
            if rel > 2e-3:
                ok_all = False
            print(f"{name:24s} rows {rs.start:>6}-{rs.stop:<6} {impl}: logits max|err| {err:.3e} rel {rel:.2e}")
    so = run_select(orig, c)
    t_o = bench(lambda: run_select(orig, c))
    line = f"{name:24s} rows={c['q'].shape[0]:6d}  orig {t_o:9.2f} ms"
    for impl in impls:
        patched._QSA_MQA_IMPL = impl
        sp = run_select(patched, c)
        # identical selections except near-ties: compare sets per row
        same_rows = (torch.sort(so, 1).values == torch.sort(sp, 1).values).all(1).float().mean().item()
        a, b = torch.sort(so, 1).values, torch.sort(sp, 1).values
        inter = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a[:256], b[:256]))
        tot = sum(len(set(x.tolist())) for x in a[:256])
        t_p = bench(lambda: run_select(patched, c))
        line += f" | {impl} {t_p:8.2f} ms ({t_o / t_p:5.1f}x) identical_rows {same_rows:.4f} overlap {inter / tot:.5f}"
        if same_rows < 0.99:
            ok_all = False
    print(line, flush=True)
print("RESULT", "PASS" if ok_all else "CHECK")
