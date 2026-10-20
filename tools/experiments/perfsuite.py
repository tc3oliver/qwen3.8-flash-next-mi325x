#!/usr/bin/env python3
"""Fixed-length performance suite used to compare configurations (~5 min per run).

All generations use ignore_eos + fixed max_tokens so output length never depends on the model,
and every prompt starts with a unique nonce so nothing is served from a previous run's cache.

  prefill_1   : 2 x cold 32K prompt, 1 output token          -> TTFT, prefill tok/s at batch 1
  prefill_8   : 8 x cold 32K prompts at once, 1 output token -> prefill throughput
  decode_c{1,16,64}: ~1K-token prompts, 512 output tokens    -> TPOT, output tok/s
  agent_32k_c16: 16 sessions x 3 turns at 32K, 256 out/turn  -> TTFT cold/warm, wall, cache hit
  agent_96k_c8 : 8 sessions x 3 turns at 96K, 256 out/turn

    uv run --with httpx experiments/perfsuite.py --base http://127.0.0.1:8001 --label X --out f.json
"""
import argparse
import asyncio
import json
import pathlib
import random
import statistics
import time
import uuid

import httpx

MODEL = "qwen3.8-flash-next"
CPT = 4.5
CORPUS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks/corpus"
TEXTS = [f.read_text(errors="ignore") for f in sorted(CORPUS.rglob("*.py"))]


def code(rng, tokens):
    out, n = [], 0
    while n < tokens * CPT:
        t = rng.choice(TEXTS); out.append(t); n += len(t)
    return "".join(out)[: int(tokens * CPT)]


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    return round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 4) if v else None


async def req(c, base, msgs, max_tokens, temp=0.6):
    body = {"model": MODEL, "messages": msgs, "stream": True, "max_tokens": max_tokens,
            "min_tokens": max_tokens, "ignore_eos": True, "temperature": temp, "seed": 1,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0, ttft, text, usage = time.perf_counter(), None, "", {}
    async with c.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                p = (dl.get("content") or "") + (dl.get("reasoning_content") or dl.get("reasoning") or "")
                if p and ttft is None:
                    ttft = time.perf_counter() - t0
                text += p
    lat = time.perf_counter() - t0
    ct = usage.get("completion_tokens") or 0
    if ttft is None:
        ttft = lat
    return {"ttft": ttft, "lat": lat, "ct": ct, "pt": usage.get("prompt_tokens") or 0,
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "tpot": (lat - ttft) / (ct - 1) if ct > 1 else None, "text": text}


async def prefill(c, base, n, tokens, nonce):
    t0 = time.perf_counter()
    rs = await asyncio.gather(*(req(c, base, [{"role": "user", "content": f"[{nonce}-{i}]\n" + code(random.Random(f"{nonce}{i}"), tokens) + "\nSummarise."}], 1) for i in range(n)))
    wall = time.perf_counter() - t0
    pt = sum(r["pt"] - r["cached"] for r in rs)
    return {"n": n, "ttft_p50": pct([r["ttft"] for r in rs], 50), "ttft_max": round(max(r["ttft"] for r in rs), 3),
            "prefill_tok_s": round(pt / wall), "uncached_tokens": pt}


async def decode(c, base, conc, nonce, out=512):
    async def one(i):
        rng = random.Random(f"{nonce}{i}")
        return await req(c, base, [{"role": "user", "content": f"[{nonce}-{i}]\n" + code(rng, 1000) + "\nRewrite this code with detailed comments."}], out)
    t0 = time.perf_counter()
    rs = await asyncio.gather(*(one(i) for i in range(conc)))
    wall = time.perf_counter() - t0
    return {"conc": conc, "tpot_p50_ms": round(1000 * pct([r["tpot"] for r in rs], 50), 2),
            "tpot_p95_ms": round(1000 * pct([r["tpot"] for r in rs], 95), 2),
            "ttft_p50": pct([r["ttft"] for r in rs], 50),
            "out_tok_s": round(sum(r["ct"] for r in rs) / wall, 1),
            "per_seq_tok_s": round(1 / pct([r["tpot"] for r in rs], 50), 1)}


async def agent(c, base, sessions, ctx, nonce, turns=3, out=256, tool=400):
    async def sess(i):
        rng = random.Random(f"{nonce}{i}")
        msgs = [{"role": "system", "content": f"[{nonce} s{i}] You are a coding agent. Repository:\n" + code(rng, ctx - turns * (out + tool + 50) - 600)},
                {"role": "user", "content": "Find one bug and propose a fix."}]
        rows = []
        for t in range(turns):
            r = await req(c, base, msgs, out)
            rows.append({"turn": t, **{k: v for k, v in r.items() if k != "text"}})
            msgs += [{"role": "assistant", "content": r["text"]},
                     {"role": "user", "content": "Tool result:\n" + code(rng, tool) + "\nNext step?"}]
        return rows
    t0 = time.perf_counter()
    rows = [r for s in await asyncio.gather(*(sess(i) for i in range(sessions))) for r in s]
    wall = time.perf_counter() - t0
    pt, cached = sum(r["pt"] for r in rows), sum(r["cached"] for r in rows)
    return {"sessions": sessions, "ctx": ctx, "wall_s": round(wall, 1),
            "ttft_cold_p50": pct([r["ttft"] for r in rows if r["turn"] == 0], 50),
            "ttft_cold_p95": pct([r["ttft"] for r in rows if r["turn"] == 0], 95),
            "ttft_warm_p50": pct([r["ttft"] for r in rows if r["turn"] > 0], 50),
            "ttft_warm_p95": pct([r["ttft"] for r in rows if r["turn"] > 0], 95),
            "ttft_p95": pct([r["ttft"] for r in rows], 95),
            "tpot_p50_ms": round(1000 * pct([r["tpot"] for r in rows], 50), 2),
            "out_tok_s": round(sum(r["ct"] for r in rows) / wall, 1),
            "cache_hit": round(cached / pt, 4) if pt else None}


async def spec_metrics(c, base):
    try:
        m = (await c.get(f"{base}/metrics")).text
    except Exception:
        return {}
    v = {}
    for line in m.splitlines():
        for k in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total", "spec_decode_num_drafts_total"):
            if line.startswith("vllm:" + k + "{"):
                v[k] = float(line.split()[-1])
    return v


async def main(a):
    nonce = uuid.uuid4().hex[:8]
    res = {"label": a.label, "time": time.strftime("%F %T"), "base": a.base}
    tests = a.only.split(",") if a.only else ["prefill_1", "prefill_8", "decode_c1", "decode_c16", "decode_c64", "agent_32k_c16", "agent_96k_c8", "agent_32k_c8", "agent_96k_c4"]
    async with httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=200)) as c:
        await req(c, a.base, [{"role": "user", "content": "warmup " + nonce}], 8)
        s0 = await spec_metrics(c, a.base)
        for t in tests:
            t0 = time.perf_counter()
            if t == "prefill_1":
                r = await prefill(c, a.base, 1, 32000, nonce + "p1a"); r2 = await prefill(c, a.base, 1, 32000, nonce + "p1b")
                r = {"ttft_s": min(r["ttft_p50"], r2["ttft_p50"]), "prefill_tok_s": max(r["prefill_tok_s"], r2["prefill_tok_s"])}
            elif t == "prefill_8":
                r = await prefill(c, a.base, 8, 32000, nonce + "p8")
            elif t.startswith("decode_c"):
                r = await decode(c, a.base, int(t[8:]), nonce + t)
            elif t.startswith("agent_"):
                ctx_k, n = t[6:].split("_c")
                r = await agent(c, a.base, int(n), int(ctx_k.rstrip("k")) * 1024, nonce + t)
            r["elapsed_s"] = round(time.perf_counter() - t0, 1)
            res[t] = r
            print(t, json.dumps(r), flush=True)
        s1 = await spec_metrics(c, a.base)
        if s1.get("spec_decode_num_draft_tokens_total"):
            d = s1["spec_decode_num_draft_tokens_total"] - s0.get("spec_decode_num_draft_tokens_total", 0)
            acc = s1["spec_decode_num_accepted_tokens_total"] - s0.get("spec_decode_num_accepted_tokens_total", 0)
            n = s1["spec_decode_num_drafts_total"] - s0.get("spec_decode_num_drafts_total", 0)
            res["mtp"] = {"acceptance": round(acc / d, 4) if d else None, "mean_accepted_len": round(1 + acc / n, 3) if n else None}
            print("mtp", json.dumps(res["mtp"]))
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
