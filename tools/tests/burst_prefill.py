#!/usr/bin/env python3
"""Concurrent long-prefill burst: N distinct ~T-token prompts at once against one endpoint.

Regression check for vllm#54764 (PLE short-conv prefill padding OOM that killed the engine at
--max-num-batched-tokens 32768 on 0.28/MI300X). PASS = every request returns 200 and the engine
is still healthy afterwards.

    uv run --with httpx tests/burst_prefill.py --base http://127.0.0.1:8001 --n 32 --tokens 8000 --out r.json
"""
import argparse
import asyncio
import json
import pathlib
import random
import time

import httpx

MODEL = "qwen3.8-flash-next"
CORPUS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks/corpus"


async def main(a):
    files = sorted(CORPUS.rglob("*.py"))
    hdr = {"Authorization": f"Bearer {a.key}"} if a.key else {}

    def prompt(i):
        rng, out, n = random.Random(i), [f"[burst {i} {time.time_ns()}]\n"], 0
        for f in rng.sample(files, len(files)):
            t = f.read_text(errors="ignore"); out.append(t); n += len(t)
            if n > a.tokens * 4.5:
                break
        return "".join(out)[: int(a.tokens * 4.5)]

    async def one(c, i):
        t0 = time.perf_counter()
        try:
            r = await c.post(f"{a.base}/v1/chat/completions", headers=hdr, json={
                "model": MODEL, "max_tokens": 32, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt(i) + "\n\nName one function above."}]})
            u = r.json().get("usage", {}) if r.status_code == 200 else {}
            return {"i": i, "status": r.status_code, "s": round(time.perf_counter() - t0, 2),
                    "prompt_tokens": u.get("prompt_tokens")}
        except Exception as e:
            return {"i": i, "status": None, "error": repr(e)[:200]}

    async with httpx.AsyncClient(timeout=1800) as c:
        t0 = time.perf_counter()
        rows = await asyncio.gather(*(one(c, i) for i in range(a.n)))
        wall = time.perf_counter() - t0
        health = (await c.get(f"{a.base}/health")).status_code
    ok = sum(r["status"] == 200 for r in rows)
    pt = sum(r.get("prompt_tokens") or 0 for r in rows)
    res = {"n": a.n, "ok": ok, "health_after": health, "wall_s": round(wall, 1),
           "prompt_tokens": pt, "prefill_tok_s": round(pt / wall), "pass": ok == a.n and health == 200}
    print(json.dumps(res))
    pathlib.Path(a.out).write_text(json.dumps({"summary": res, "rows": rows}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
