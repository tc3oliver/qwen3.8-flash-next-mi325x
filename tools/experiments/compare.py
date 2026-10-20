#!/usr/bin/env python3
"""Side-by-side table of perfsuite results.  experiments/compare.py e00-base e02-qsa ...
Reads ../results/experiments/ by default; RESULTS=experiments/results for new runs."""
import json, os, sys, pathlib
R = pathlib.Path(os.environ.get("RESULTS") or pathlib.Path(__file__).resolve().parents[2] / "results/experiments")
names = sys.argv[1:] or sorted(p.stem for p in R.glob("e*.json") if "." not in p.stem)
rows = [
    ("gsm8k200", lambda d: d.get("gsm8k200")),
    ("prefill_1 tok/s", lambda d: d["prefill_1"]["prefill_tok_s"]),
    ("prefill_1 TTFT s", lambda d: d["prefill_1"]["ttft_s"]),
    ("prefill_8 tok/s", lambda d: d["prefill_8"]["prefill_tok_s"]),
    ("decode c1 TPOT ms", lambda d: d["decode_c1"]["tpot_p50_ms"]),
    ("decode c16 tok/s", lambda d: d["decode_c16"]["out_tok_s"]),
    ("decode c16 TPOT ms", lambda d: d["decode_c16"]["tpot_p50_ms"]),
    ("decode c64 tok/s", lambda d: d["decode_c64"]["out_tok_s"]),
    ("decode c64 TPOT ms", lambda d: d["decode_c64"]["tpot_p50_ms"]),
    ("agent32k wall s", lambda d: d["agent_32k_c16"]["wall_s"]),
    ("agent32k cold p50", lambda d: d["agent_32k_c16"]["ttft_cold_p50"]),
    ("agent32k ttft p95", lambda d: d["agent_32k_c16"]["ttft_p95"]),
    ("agent32k warm p50", lambda d: d["agent_32k_c16"]["ttft_warm_p50"]),
    ("agent96k wall s", lambda d: d["agent_96k_c8"]["wall_s"]),
    ("agent96k ttft p95", lambda d: d["agent_96k_c8"]["ttft_p95"]),
    ("agent96k TPOT ms", lambda d: d["agent_96k_c8"]["tpot_p50_ms"]),
    ("agent32k_c8 wall s", lambda d: d["agent_32k_c8"]["wall_s"]),
    ("agent32k_c8 p95", lambda d: d["agent_32k_c8"]["ttft_p95"]),
    ("agent96k_c4 wall s", lambda d: d["agent_96k_c4"]["wall_s"]),
    ("agent96k_c4 p95", lambda d: d["agent_96k_c4"]["ttft_p95"]),
    ("MTP accept", lambda d: (d.get("mtp") or {}).get("acceptance")),
]
data = {}
for n in names:
    try:
        data[n] = json.load(open(R / f"{n}.json"))
    except Exception as e:
        data[n] = {"error": str(e)}
w = max(14, *(len(n) for n in names))
print(f"{'metric':20s}" + "".join(f"{n:>{w+2}s}" for n in names))
for label, f in rows:
    line = f"{label:20s}"
    for n in names:
        try:
            v = f(data[n]); line += f"{v:>{w+2}}" if v is not None else f"{'-':>{w+2}}"
        except Exception:
            line += f"{'-':>{w+2}}"
    print(line)
