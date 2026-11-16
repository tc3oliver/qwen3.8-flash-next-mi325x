#!/usr/bin/env python3
"""One cold request: prompt ~N tokens, ignore_eos, M output tokens. usage: probe.py --tokens N --decode M --seed S --label L --out DIR"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import stage
ap = argparse.ArgumentParser(); ap.add_argument("--tokens", type=int); ap.add_argument("--decode", type=int); ap.add_argument("--seed", type=int); ap.add_argument("--label"); ap.add_argument("--out")
a = ap.parse_args(); prompt, exp, meta = stage.build(a.tokens, a.seed); res = dict(label=a.label, target=a.tokens, decode=a.decode)
try:
    r = stage.chat(prompt, a.decode, ignore_eos=True); u = r["usage"]; ct = u["completion_tokens"]
    res.update(prompt_tokens=u["prompt_tokens"], completion_tokens=ct, total_positions=u["prompt_tokens"] + ct, ttft_s=r["ttft"], ok=True,
               crossed_262144=u["prompt_tokens"] + ct > 262144, tpot_ms=1000 * r["decode_s"] / (ct - 1))
except Exception as e:
    res.update(ok=False, error=repr(e))
os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/{a.label}.json", "w"), indent=1); print(json.dumps(res))
