#!/usr/bin/env python3
"""Long-context sanity: hide a code-word near the start / middle of a ~T-token real-code prompt
and ask for it at the end; also check the answer is not degenerate (HF Qwen3.8-Flash-Next-FP8 #4
reported garbage output in long agent sessions near 200K).

    uv run --with httpx tests/long_needle.py --base http://127.0.0.1:8001 --tokens 200000 --out r.json
"""
import argparse
import json
import pathlib
import time

import httpx

MODEL = "qwen3.8-flash-next"
CORPUS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks/corpus"


def main(a):
    text, n = [], 0
    for f in sorted(CORPUS.rglob("*.py")):
        t = f.read_text(errors="ignore"); text.append(f"### FILE: {f}\n{t}\n"); n += len(t)
        if n > a.tokens * 4.5:
            break
    body = "".join(text)[: int(a.tokens * 4.5)]
    rows = []
    with httpx.Client(timeout=1800) as c:
        for pos in (0.05, 0.5, 0.95):
            word = f"ZEPHYR-{int(pos * 100)}-{time.time_ns() % 10000}"
            k = int(len(body) * pos)
            p = body[:k] + f"\n# NOTE: the release tag for this build is {word}\n" + body[k:]
            t0 = time.perf_counter()
            r = c.post(f"{a.base}/v1/chat/completions", json={
                "model": MODEL, "max_tokens": 64, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": p + "\n\nWhat is the release tag for this build? Reply with the tag only."}]})
            r.raise_for_status(); d = r.json()
            out = d["choices"][0]["message"]["content"] or ""
            rows.append({"pos": pos, "prompt_tokens": d["usage"]["prompt_tokens"], "s": round(time.perf_counter() - t0, 1),
                         "expected": word, "got": out.strip()[:120], "ok": word in out})
            print(json.dumps(rows[-1]), flush=True)
    res = {"tokens_target": a.tokens, "pass": all(r["ok"] for r in rows)}
    print(json.dumps(res))
    pathlib.Path(a.out).write_text(json.dumps({"summary": res, "rows": rows}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
