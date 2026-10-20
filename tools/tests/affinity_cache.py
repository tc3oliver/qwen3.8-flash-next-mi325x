#!/usr/bin/env python3
"""Session affinity + prefix-cache test through the LiteLLM gateway.

Each simulated agent session owns a distinct code prefix (real source files) and grows its
conversation turn by turn, like a coding agent appending tool results. For every request we
record which replica served it (x-litellm-model-id), TTFT, and usage cached_tokens.

Modes:
  affinity   sessions send a session header (default x-session-id, as Pi Agent does)
  noheader   control: same workload with no session header -> plain simple-shuffle

    uv run --with httpx tests/affinity_cache.py --base http://127.0.0.1:4000 --key $K \
        --sessions 8 --turns 4 --prefix-tokens 6000 --mode affinity --out results/affinity.json
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
CHARS_PER_TOKEN = 4.5   # measured on this corpus with the Qwen3.8 tokenizer (238K tokens per 1.07M chars)


def load_corpus(d):
    files = sorted(pathlib.Path(d).rglob("*.py"))
    if not files:
        raise SystemExit(f"no corpus in {d}; run benchmarks/make_corpus.sh first")
    return files


def make_prefix(files, rng, tokens):
    out, n = [], 0
    for f in rng.sample(files, len(files)):
        t = f.read_text(errors="ignore")
        out.append(f"### FILE: {f.name}\n{t}\n")
        n += len(t)
        if n >= tokens * CHARS_PER_TOKEN:
            break
    return "".join(out)[: int(tokens * CHARS_PER_TOKEN)]


async def one_request(c, base, headers, messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "stream": True, "max_tokens": max_tokens,
            "temperature": 0.2, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0, ttft, text, usage = time.perf_counter(), None, "", {}
    async with c.stream("POST", f"{base}/v1/chat/completions", json=body, headers=headers) as r:
        r.raise_for_status()
        model_id = r.headers.get("x-litellm-model-id")
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                p = ch.get("delta", {}).get("content") or ""
                if p and ttft is None:
                    ttft = time.perf_counter() - t0
                text += p
    total = time.perf_counter() - t0
    det = usage.get("prompt_tokens_details") or {}
    return {"model_id": model_id, "ttft": ttft, "latency": total, "text": text,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": det.get("cached_tokens")}


async def session(i, a, files, c, auth):
    await asyncio.sleep(i * a.stagger)
    rng = random.Random(a.seed * 100003 + 1000 + i)
    sid = f"{a.mode}-{a.run_id}-s{i}"
    hdr = dict(auth)
    if a.mode == "affinity":
        hdr[a.header] = sid
    msgs = [{"role": "system", "content": "You are a coding agent working in this repository:\n\n"
                                          + make_prefix(files, rng, a.prefix_tokens)},
            {"role": "user", "content": "Summarise what the first file does in 3 bullet points."}]
    rows = []
    for turn in range(a.turns):
        try:
            res = await one_request(c, a.base, hdr, msgs, a.max_tokens)
            err = None
        except Exception as e:
            res, err = {}, repr(e)[:300]
        rows.append({"session": sid, "turn": turn, "error": err,
                     **{k: v for k, v in res.items() if k != "text"}})
        msgs += [{"role": "assistant", "content": res.get("text") or "(error)"},
                 {"role": "user", "content": f"Tool output (turn {turn}):\n"
                                             + make_prefix(files, rng, 300)
                                             + "\nWhat should we change next? Be brief."}]
    return rows


async def main(a):
    files = load_corpus(a.corpus)
    auth = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    async with httpx.AsyncClient(timeout=900) as c:
        all_rows = await asyncio.gather(*(session(i, a, files, c, auth) for i in range(a.sessions)))
    rows = [r for s in all_rows for r in s]
    ok = [r for r in rows if not r["error"]]

    per_session = {}
    for r in ok:
        per_session.setdefault(r["session"], []).append(r["model_id"])
    sticky = sum(len(set(v)) == 1 for v in per_session.values())
    first_replica = [str(v[0]) for v in per_session.values()]

    def stats(sel):
        t = [r["ttft"] for r in sel if r["ttft"] is not None]
        pt = sum(r["prompt_tokens"] or 0 for r in sel)
        ct = sum(r["cached_tokens"] or 0 for r in sel)
        return {"n": len(sel), "ttft_p50": round(statistics.median(t), 3) if t else None,
                "ttft_mean": round(statistics.mean(t), 3) if t else None,
                "prompt_tokens": pt, "cached_tokens": ct,
                "cache_hit_rate": round(ct / pt, 4) if pt else None}

    summary = {
        "mode": a.mode, "header": a.header if a.mode == "affinity" else None,
        "sessions": a.sessions, "turns": a.turns, "prefix_tokens_target": a.prefix_tokens,
        "requests": len(rows), "errors": len(rows) - len(ok),
        "sessions_sticky": f"{sticky}/{len(per_session)}",
        "replica_spread_first_turn": {k: first_replica.count(k) for k in sorted(set(first_replica))},
        "turn0": stats([r for r in ok if r["turn"] == 0]),
        "turn_ge1": stats([r for r in ok if r["turn"] >= 1]),
    }
    print(json.dumps(summary, indent=2))
    pathlib.Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--mode", choices=["affinity", "noheader"], default="affinity")
    ap.add_argument("--header", default="x-session-id")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--prefix-tokens", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--corpus", default=str(pathlib.Path(__file__).resolve().parent.parent / "benchmarks/corpus"))
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between session starts")
    ap.add_argument("--seed", type=int, default=0, help="different seed = different prefixes (no cross-run cache hits)")
    ap.add_argument("--run-id", default=uuid.uuid4().hex[:6])
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
