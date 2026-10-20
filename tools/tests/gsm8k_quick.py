#!/usr/bin/env python3
"""Quick GSM8K accuracy check (greedy, chat template, thinking off) against one endpoint.

Reference on gfx942: vllm#60839 reports GSM8K 5-shot 0.9704 (flexible) for this checkpoint at TP2
on MI300X. This is a zero-shot subset, so expect the same ballpark (> ~0.90), not the same number;
a broken kernel path (wrong FP8 format, stale n-gram IDs) shows up as a collapse, not a 1-2% drift.

    uv run --with httpx,datasets tests/gsm8k_quick.py --base http://127.0.0.1:8001 --n 200 --out r.json
"""
import argparse
import asyncio
import json
import re

import httpx
from datasets import load_dataset

MODEL = "qwen3.8-flash-next"


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


async def main(a):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
    hdr = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    sem = asyncio.Semaphore(a.concurrency)

    async def one(c, ex):
        async with sem:
            r = await c.post(f"{a.base}/v1/chat/completions", headers=hdr, json={
                "model": MODEL, "temperature": 0, "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": ex["question"]
                              + "\nSolve step by step, then give the final answer as 'Answer: <number>'."}]})
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", out)
            pred = (m.group(1).replace(",", "") if m else last_number(out))
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            try:
                ok = pred is not None and float(pred) == float(gold)
            except ValueError:
                ok = False
            return {"gold": gold, "pred": pred, "ok": ok}

    async with httpx.AsyncClient(timeout=600) as c:
        res = await asyncio.gather(*(one(c, ex) for ex in ds))
    acc = sum(r["ok"] for r in res) / len(res)
    print(f"GSM8K zero-shot subset n={len(res)} accuracy={acc:.4f}")
    with open(a.out, "w") as f:
        json.dump({"base": a.base, "n": len(res), "accuracy": acc, "rows": res}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
