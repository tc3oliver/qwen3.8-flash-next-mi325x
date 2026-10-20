#!/usr/bin/env python3
"""Simulate N coding-agent users who never clear their context, through the LiteLLM gateway.

Each user: starts with a 20K-120K token real-code context, then loops: request (256 output tokens,
fixed) -> think/tool time ~ Exp(mean) -> append the answer + a tool result (~2K tokens). Contexts only
grow (users do not clear), capped near max_model_len (the user "starts a new session" when it would not
fit). Measures TTFT (cold first turn vs later turns), cache hit, errors, over a fixed wall time.

    uv run --with httpx tests/sim_users.py --base http://127.0.0.1:4000 --key $K --users 100 --minutes 15 --out r.json
"""
import argparse, asyncio, json, pathlib, random, statistics, time, uuid
import httpx

MODEL = "qwen3.8-flash-next"
CPT = 4.0          # conservative chars/token so prompts never exceed the window
LIMIT = 250_000    # tokens; start a fresh session beyond this
CORPUS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks/corpus"
TEXTS = [f.read_text(errors="ignore") for f in sorted(CORPUS.rglob("*.py"))]


def code(rng, tokens):
    out, n = [], 0
    while n < tokens * CPT:
        t = rng.choice(TEXTS); out.append(t); n += len(t)
    return "".join(out)[: int(tokens * CPT)]


def pct(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 3) if v else None


async def one(c, base, hdr, msgs):
    body = {"model": MODEL, "messages": msgs, "stream": True, "max_tokens": 256, "min_tokens": 256,
            "ignore_eos": True, "temperature": 0.6, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0, ttft, text, usage = time.perf_counter(), None, "", {}
    async with c.stream("POST", f"{base}/v1/chat/completions", json=body, headers=hdr) as r:
        r.raise_for_status()
        mid = r.headers.get("x-litellm-model-id")
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:]); usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                p = ch.get("delta", {}).get("content") or ""
                if p and ttft is None:
                    ttft = time.perf_counter() - t0
                text += p
    lat = time.perf_counter() - t0
    return {"ttft": ttft or lat, "lat": lat, "pt": usage.get("prompt_tokens") or 0,
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "model_id": mid, "text": text}


async def user(i, a, c, auth, rows, t_end):
    rng = random.Random(f"{a.seed}-{i}")
    await asyncio.sleep(rng.uniform(0, a.ramp))
    while time.time() < t_end:
        sid = f"sim-{a.run}-{i}-{uuid.uuid4().hex[:6]}"
        hdr = {**auth, "x-session-id": sid}
        ctx = rng.randint(20_000, 120_000)
        msgs = [{"role": "system", "content": f"[{sid}] You are a coding agent. Repository:\n" + code(rng, ctx)},
                {"role": "user", "content": "Find one bug and propose a fix."}]
        turn, est = 0, ctx
        while time.time() < t_end and est < LIMIT:
            try:
                r = await one(c, a.base, hdr, msgs); err = None
            except Exception as e:
                r, err = {}, repr(e)[:160]
            rows.append({"user": i, "session": sid, "turn": turn, "t": time.time(), "error": err,
                         **{k: v for k, v in r.items() if k != "text"}})
            tool = rng.randint(800, 3000)
            msgs += [{"role": "assistant", "content": r.get("text") or "(error)"},
                     {"role": "user", "content": "Tool result:\n" + code(rng, tool) + "\nNext step?"}]
            est += tool + 300; turn += 1
            await asyncio.sleep(rng.expovariate(1 / a.think))


async def main(a):
    auth = {"Authorization": f"Bearer {a.key}"}
    rows = []
    t_end = time.time() + a.minutes * 60
    async with httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=a.users + 10)) as c:
        await asyncio.gather(*(user(i, a, c, auth, rows, t_end) for i in range(a.users)))
    ok = [r for r in rows if not r["error"]]
    warm = [r for r in ok if r["turn"] > 0]; cold = [r for r in ok if r["turn"] == 0]
    after_ramp = [r for r in warm if r["t"] > rows[0]["t"] + a.ramp] if rows else []
    pt, ca = sum(r["pt"] for r in warm), sum(r["cached"] for r in warm)
    s = {"users": a.users, "minutes": a.minutes, "think_mean_s": a.think, "requests": len(rows),
         "errors": len(rows) - len(ok), "error_samples": list({r["error"][:100] for r in rows if r["error"]})[:3],
         "ttft_cold_p50": pct([r["ttft"] for r in cold], 50), "ttft_cold_p95": pct([r["ttft"] for r in cold], 95),
         "ttft_warm_p50": pct([r["ttft"] for r in warm], 50), "ttft_warm_p95": pct([r["ttft"] for r in warm], 95),
         "ttft_warm_p99": pct([r["ttft"] for r in warm], 99),
         "warm_cache_hit": round(ca / pt, 4) if pt else None,
         "warm_turns_with_cache_miss_gt50pct": sum(1 for r in warm if r["pt"] and r["cached"] / r["pt"] < 0.5),
         "mean_prompt_tokens_warm": round(statistics.mean([r["pt"] for r in warm])) if warm else None,
         "replica_requests": {k: sum(1 for r in ok if r["model_id"] == k) for k in sorted({r["model_id"] for r in ok if r["model_id"]})}}
    print(json.dumps(s, indent=1))
    pathlib.Path(a.out).write_text(json.dumps({"summary": s, "rows": rows}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:4000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--think", type=float, default=30, help="mean seconds between a user's turns")
    ap.add_argument("--ramp", type=float, default=120, help="users start uniformly over this many seconds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run", default=uuid.uuid4().hex[:6])
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
