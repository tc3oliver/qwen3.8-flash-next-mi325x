#!/usr/bin/env python3
"""Replica failover + restart test through LiteLLM. Stops/starts ONLY this stack's containers.

1. Warm a session (long code prefix) and record its pinned replica + cached_tokens.
2. `docker stop` that replica.
3. Next turn of the same session must still succeed on the other replica; cached_tokens is
   reported as measured (expected cold), never assumed.
4. Start the replica again via scripts/up.sh, wait healthy, send more turns, record routing.

    uv run --with httpx tests/failover.py --key $LITELLM_MASTER_KEY --out results/failover.json
"""
import argparse
import json
import pathlib
import subprocess
import time
import uuid

import httpx

MODEL = "qwen3.8-flash-next"
SVC = pathlib.Path(__file__).resolve().parent.parent


def ask(c, base, key, sid, msgs):
    t0 = time.perf_counter()
    r = c.post(f"{base}/v1/chat/completions",
               headers={"Authorization": f"Bearer {key}", "x-session-id": sid},
               json={"model": MODEL, "messages": msgs, "max_tokens": 64, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False}})
    dt = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    u = d.get("usage") or {}
    return {"model_id": r.headers.get("x-litellm-model-id"), "latency_s": round(dt, 2),
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": d["choices"][0]["message"]["content"]}


def main(a):
    corpus = sorted((SVC / "benchmarks/corpus").rglob("*.py"))
    prefix = "".join(f.read_text(errors="ignore") for f in corpus[:40])[:40000]
    sid = f"failover-{uuid.uuid4().hex[:8]}"
    msgs = [{"role": "system", "content": f"[{sid}] Repository:\n{prefix}"},
            {"role": "user", "content": "Name one class defined above."}]
    log = []
    c = httpx.Client(timeout=900)

    def turn(stage):
        res = ask(c, a.base, a.key, sid, msgs)
        log.append({"stage": stage, **{k: v for k, v in res.items() if k != "text"}})
        print(json.dumps(log[-1]), flush=True)
        msgs.extend([{"role": "assistant", "content": res["text"]},
                     {"role": "user", "content": "Name another one."}])
        return res

    turn("warm-1")
    pinned = turn("warm-2")["model_id"]
    short = pinned.split("-", 1)[1]          # replica-3 -> 3, replica-a -> a
    name = f"qwen38-vllm-{short}"
    print(f"session pinned to {pinned}; stopping {name}", flush=True)
    t_stop = time.time()
    subprocess.run(["docker", "stop", "-t", "30", name], check=True, capture_output=True)

    failed_over = []
    for i in range(3):
        try:
            failed_over.append(turn(f"after-stop-{i}"))
        except Exception as e:
            log.append({"stage": f"after-stop-{i}", "error": repr(e)[:300]})
            print(log[-1], flush=True)

    print(f"restarting {name}", flush=True)
    subprocess.run([str(SVC.parent / "deploy/scripts/up.sh"), short], check=True)
    t_back = time.time()
    time.sleep(a.settle)
    for i in range(3):
        turn(f"after-restart-{i}")

    ok_fo = [r for r in failed_over if r.get("model_id") and r["model_id"] != pinned]
    summary = {
        "session": sid, "pinned_replica": pinned,
        "failover_success": f"{len(ok_fo)}/3",
        "first_failover_cached_tokens": failed_over[0].get("cached_tokens") if failed_over else None,
        "first_failover_prompt_tokens": failed_over[0].get("prompt_tokens") if failed_over else None,
        "replica_downtime_s": round(t_back - t_stop, 1),
        "routing_after_restart": [r.get("model_id") for r in log if r["stage"].startswith("after-restart")],
    }
    print(json.dumps(summary, indent=2))
    pathlib.Path(a.out).write_text(json.dumps({"summary": summary, "log": log}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:4000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--settle", type=int, default=35, help="wait > LiteLLM cooldown_time before re-testing")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
