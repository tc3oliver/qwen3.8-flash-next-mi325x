#!/usr/bin/env python3
"""Coding-agent workload benchmark (context x concurrent sessions).

Each session = one simulated agent: a real-source-code system prompt sized to the target context,
then several turns that append the model's answer plus a "tool output" (more real code), exactly
like an agent loop. Turn 0 is cold (unique nonce at the very start defeats cross-run caching);
later turns reuse the session prefix, so prefix caching is exercised realistically.

Measured per request: TTFT, TPOT, generation tok/s, prompt/cached tokens, errors.
Per cell: P50/P95 TTFT and TPOT, per-session gen tok/s, aggregate output tok/s, req/s,
cache hit rate, GPU HBM/utilisation (amd-smi), host CPU/RAM. Writes CSV + JSON.

    uv run --with httpx benchmarks/agent_bench.py --base http://127.0.0.1:4000 --key $K \
        --label dual --contexts 8192,32768,65536 --concurrency 1,4,8,16 --out-dir benchmarks/results
"""
import argparse
import asyncio
import csv
import json
import pathlib
import random
import statistics
import subprocess
import threading
import time
import uuid

import httpx

MODEL = "qwen3.8-flash-next"
CHARS_PER_TOKEN = 4.5   # measured on this corpus with the Qwen3.8 tokenizer (238K tokens per 1.07M chars)
GPUS = ["0", "1", "2", "3"]   # amd-smi indices (HIP 1,3,2,0)


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)


class Sampler(threading.Thread):
    """Samples GPU (amd-smi) and host CPU/RAM every `interval` seconds."""

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval, self.stop_evt, self.samples = interval, threading.Event(), []

    @staticmethod
    def cpu_times():
        f = open("/proc/stat").readline().split()[1:]
        v = list(map(int, f))
        return sum(v), v[3] + v[4]

    def run(self):
        last = self.cpu_times()
        while not self.stop_evt.is_set():
            s = {"t": time.time()}
            try:
                out = subprocess.run(["amd-smi", "metric", "-g", *GPUS, "-u", "-m", "--json"],
                                     capture_output=True, text=True, timeout=10).stdout
                for g in json.loads(out)["gpu_data"]:
                    s[f"gpu{g['gpu']}_util"] = g["usage"]["gfx_activity"]["value"]
                    s[f"gpu{g['gpu']}_vram_mib"] = g["mem_usage"]["used_vram"]["value"]
            except Exception:
                pass
            cur = self.cpu_times()
            dt, di = cur[0] - last[0], cur[1] - last[1]
            s["cpu_pct"] = round(100 * (1 - di / dt), 1) if dt else None
            last = cur
            mem = dict(l.split(":", 1) for l in open("/proc/meminfo"))
            tot, av = int(mem["MemTotal"].split()[0]), int(mem["MemAvailable"].split()[0])
            s["ram_used_gib"] = round((tot - av) / 2**20, 1)
            self.samples.append(s)
            self.stop_evt.wait(self.interval)

    def summary(self):
        out = {}
        if not self.samples:
            return out
        keys = {k for s in self.samples for k in s if k != "t"}
        for k in sorted(keys):
            vals = [s[k] for s in self.samples if isinstance(s.get(k), (int, float))]
            if vals:
                out[f"{k}_max"] = max(vals)
                out[f"{k}_mean"] = round(statistics.mean(vals), 1)
        return out


def load_corpus(d):
    files = sorted(pathlib.Path(d).rglob("*.py"))
    if not files:
        raise SystemExit(f"no corpus in {d}; run benchmarks/make_corpus.sh first")
    return [f.read_text(errors="ignore") for f in files]


def code_block(texts, rng, tokens):
    out, n = [], 0
    while n < tokens * CHARS_PER_TOKEN:
        t = rng.choice(texts)
        out.append(t)
        n += len(t)
    return "".join(out)[: int(tokens * CHARS_PER_TOKEN)]


FIXED_OUTPUT = False   # --fixed-output: ignore_eos + min_tokens so every turn generates max_tokens


async def request(c, base, headers, messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "stream": True, "max_tokens": max_tokens,
            "temperature": 0.6, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if FIXED_OUTPUT:
        body.update(ignore_eos=True, min_tokens=max_tokens)
    t0, ttft, text, usage, model_id = time.perf_counter(), None, "", {}, None
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
    lat = time.perf_counter() - t0
    ct = usage.get("completion_tokens") or 0
    tpot = (lat - ttft) / (ct - 1) if ttft is not None and ct > 1 else None
    return {"ttft": ttft, "latency": lat, "tpot": tpot, "completion_tokens": ct,
            "gen_tps": (ct - 1) / (lat - ttft) if tpot else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "model_id": model_id, "text": text}


async def agent_session(i, ctx, a, texts, c, auth, nonce):
    await asyncio.sleep(i * a.stagger)   # LiteLLM least-busy needs non-simultaneous session starts
    rng = random.Random(f"{nonce}-{i}")
    hdr = {**auth, "x-session-id": f"bench-{nonce}-{ctx}-s{i}"}
    turn_tokens = a.tool_tokens + a.max_tokens + 50
    prefix_tokens = max(1000, ctx - a.turns * turn_tokens - 600)
    msgs = [{"role": "system", "content": f"[run {nonce} session {i}] You are a coding agent. Repository:\n"
                                          + code_block(texts, rng, prefix_tokens)},
            {"role": "user", "content": "Find one bug or code smell in the repository and propose a fix."}]
    rows = []
    for turn in range(a.turns):
        try:
            res, err = await request(c, a.base, hdr, msgs, a.max_tokens), None
        except Exception as e:
            res, err = {}, repr(e)[:300]
        rows.append({"session": i, "turn": turn, "error": err, **{k: v for k, v in res.items() if k != "text"}})
        msgs += [{"role": "assistant", "content": res.get("text") or "(error)"},
                 {"role": "user", "content": "Tool result:\n" + code_block(texts, rng, a.tool_tokens)
                                             + "\nContinue: what is the next step?"}]
    return rows


async def run_cell(ctx, conc, a, texts, auth):
    nonce = uuid.uuid4().hex[:8]
    sampler = Sampler()
    sampler.start()
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=conc + 4)) as c:
        res = await asyncio.gather(*(agent_session(i, ctx, a, texts, c, auth, nonce) for i in range(conc)))
    wall = time.perf_counter() - t0
    sampler.stop_evt.set()
    sampler.join()
    rows = [r for s in res for r in s]
    ok = [r for r in rows if not r["error"]]
    out_tok = sum(r["completion_tokens"] for r in ok)
    pt = sum(r["prompt_tokens"] or 0 for r in ok)
    cached = sum(r["cached_tokens"] or 0 for r in ok)
    cell = {
        "label": a.label, "context": ctx, "concurrency": conc, "turns": a.turns,
        "requests": len(rows), "errors": len(rows) - len(ok),
        "error_rate": round((len(rows) - len(ok)) / len(rows), 4) if rows else None,
        "wall_s": round(wall, 1),
        "ttft_p50": pct([r["ttft"] for r in ok], 50), "ttft_p95": pct([r["ttft"] for r in ok], 95),
        "ttft_warm_p50": pct([r["ttft"] for r in ok if r["turn"] > 0], 50),
        "ttft_cold_p50": pct([r["ttft"] for r in ok if r["turn"] == 0], 50),
        "tpot_p50": pct([r["tpot"] for r in ok], 50), "tpot_p95": pct([r["tpot"] for r in ok], 95),
        "session_gen_tps_p50": pct([r["gen_tps"] for r in ok], 50),
        "aggregate_output_tps": round(out_tok / wall, 1) if wall else None,
        "requests_per_s": round(len(ok) / wall, 3) if wall else None,
        "prompt_tokens": pt, "cached_tokens": cached,
        "cache_hit_rate": round(cached / pt, 4) if pt else None,
        "mean_prompt_tokens": round(pt / len(ok)) if ok else None,
        "replicas": {k: sum(1 for r in ok if r["model_id"] == k) for k in {r["model_id"] for r in ok}},
        **sampler.summary(),
    }
    return cell, rows


async def main(a):
    texts = load_corpus(a.corpus)
    auth = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    outdir = pathlib.Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cells, detail = [], {}
    for ctx in map(int, a.contexts.split(",")):
        for conc in map(int, a.concurrency.split(",")):
            print(f"== {a.label} ctx={ctx} conc={conc}", flush=True)
            cell, rows = await run_cell(ctx, conc, a, texts, auth)
            print(json.dumps({k: cell[k] for k in ("errors", "ttft_p50", "ttft_p95", "tpot_p50",
                                                   "session_gen_tps_p50", "aggregate_output_tps",
                                                   "cache_hit_rate", "replicas")}), flush=True)
            cells.append(cell)
            detail[f"{ctx}x{conc}"] = rows
            (outdir / f"{a.label}-{stamp}.json").write_text(json.dumps({"cells": cells, "rows": detail}, indent=2))
    keys = sorted({k for c in cells for k in c if k != "replicas"}, key=lambda k: (k not in cells[0], k))
    with open(outdir / f"{a.label}-{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["replicas"])
        w.writeheader()
        for c in cells:
            w.writerow({**c, "replicas": json.dumps(c["replicas"])})
    print(f"wrote {outdir}/{a.label}-{stamp}.csv/.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--contexts", default="8192,32768,65536")
    ap.add_argument("--concurrency", default="1,4,8,16")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tool-tokens", type=int, default=400)
    ap.add_argument("--corpus", default=str(pathlib.Path(__file__).resolve().parent / "corpus"))
    ap.add_argument("--stagger", type=float, default=0.25, help="seconds between session starts")
    ap.add_argument("--fixed-output", action="store_true", help="ignore_eos: fixed generation length per turn")
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).resolve().parent / "results"))
    _a = ap.parse_args()
    FIXED_OUTPUT = _a.fixed_output
    asyncio.run(main(_a))
