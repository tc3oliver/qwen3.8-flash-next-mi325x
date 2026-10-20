#!/usr/bin/env python3
"""Functional tests for the Qwen3.8-Flash-Next deployment.

Runs against one replica (direct) or the LiteLLM gateway. Every check prints PASS/FAIL with
evidence and the full result set is written as JSON. Nothing is marked PASS without a check.

    uv run --with httpx tests/functional.py --base http://127.0.0.1:8001 --out results/a.json
    uv run --with httpx tests/functional.py --base http://127.0.0.1:4000 --key $LITELLM_MASTER_KEY --gateway
"""
import argparse
import json
import time
import uuid

import httpx

MODEL = "qwen3.8-flash-next"
TOOLS = [
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run the project's test suite and return the output.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "test file or directory"},
            "verbose": {"type": "boolean"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
]


class T:
    def __init__(self, base, key, gateway):
        self.base = base.rstrip("/")
        self.gateway = gateway
        self.h = {"Authorization": f"Bearer {key}"} if key else {}
        self.c = httpx.Client(timeout=600)
        self.results = []

    def record(self, name, ok, **evidence):
        self.results.append({"test": name, "pass": bool(ok), **evidence})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(evidence, ensure_ascii=False)[:300]}")

    def chat(self, messages, headers=None, **kw):
        body = {"model": MODEL, "messages": messages, "max_tokens": kw.pop("max_tokens", 2048), **kw}
        r = self.c.post(f"{self.base}/v1/chat/completions", json=body, headers={**self.h, **(headers or {})})
        r.raise_for_status()
        return r.json(), r.headers

    @staticmethod
    def reasoning_of(msg):
        return msg.get("reasoning_content") or msg.get("reasoning") or ""

    def run(self):
        steps = [self.t_health, self.t_models, self.t_chat, self.t_stream, self.t_reasoning,
                 self.t_tool_loop, self.t_multi_turn, self.t_short_prompt_determinism]
        for step in steps:
            try:
                step()
            except Exception as e:  # a crash in one check must not hide the others
                self.record(step.__name__, False, error=repr(e)[:500])

    def t_health(self):
        path = "/health/liveliness" if self.gateway else "/health"
        r = self.c.get(self.base + path, headers=self.h)
        self.record("health", r.status_code == 200, status=r.status_code)

    def t_models(self):
        r = self.c.get(self.base + "/v1/models", headers=self.h)
        ids = [m["id"] for m in r.json().get("data", [])]
        self.record("v1_models", MODEL in ids, ids=ids)

    def t_chat(self):
        d, h = self.chat([{"role": "user", "content": "What is 17*23? Reply with just the number."}],
                         max_tokens=1024, chat_template_kwargs={"enable_thinking": False})
        txt = d["choices"][0]["message"]["content"] or ""
        self.record("chat_completion", "391" in txt, content=txt.strip()[:80],
                    usage=d.get("usage"), model_id=h.get("x-litellm-model-id"))

    def t_stream(self):
        body = {"model": MODEL, "stream": True, "max_tokens": 256,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Count from 1 to 10, comma separated."}]}
        chunks, text, t0, ttft, usage = 0, "", time.time(), None, None
        with self.c.stream("POST", f"{self.base}/v1/chat/completions", json=body, headers=self.h) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    piece = ch.get("delta", {}).get("content") or ""
                    if piece and ttft is None:
                        ttft = time.time() - t0
                    text += piece
                    chunks += 1
        ok = chunks > 3 and "10" in text and ttft is not None
        self.record("streaming", ok, chunks=chunks, ttft_s=round(ttft or -1, 3), text=text[:60], usage=usage)

    def t_reasoning(self):
        d, _ = self.chat([{"role": "user", "content": "Is 221 prime? Think, then answer yes or no."}],
                         max_tokens=4096, chat_template_kwargs={"enable_thinking": True})
        m = d["choices"][0]["message"]
        r, c = self.reasoning_of(m), m.get("content") or ""
        ok = len(r) > 20 and "<think>" not in c and "</think>" not in c and c.strip() != ""
        self.record("reasoning_content", ok, reasoning_chars=len(r), content=c.strip()[:120])

    def t_tool_loop(self):
        msgs = [{"role": "system", "content": "You are a coding agent. Use tools when needed."},
                {"role": "user", "content": "Run the tests in tests/test_math.py with verbose output."}]
        d, _ = self.chat(msgs, tools=TOOLS, tool_choice="auto", max_tokens=4096)
        m = d["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        args_ok = False
        if calls:
            try:
                args = json.loads(calls[0]["function"]["arguments"])
                args_ok = calls[0]["function"]["name"] == "run_tests" and "test_math" in args.get("path", "")
            except json.JSONDecodeError:
                args_ok = False
        leak = "<tool_call>" in (m.get("content") or "") or "<function=" in (m.get("content") or "")
        self.record("tool_call_parse", bool(calls) and args_ok and not leak,
                    finish=d["choices"][0]["finish_reason"], calls=calls[:1], content_leak=leak)
        if not calls:
            return
        msgs += [{"role": "assistant", "content": m.get("content"), "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0]["id"],
                  "content": "FAILED tests/test_math.py::test_add - assert add(2, 2) == 5\n1 failed, 3 passed"}]
        d2, _ = self.chat(msgs, tools=TOOLS, tool_choice="auto", max_tokens=4096)
        m2 = d2["choices"][0]["message"]
        txt = (m2.get("content") or "") + json.dumps(m2.get("tool_calls") or [])
        ok = ("test_add" in txt or "add" in txt) and d2["choices"][0]["finish_reason"] in ("stop", "tool_calls")
        self.record("tool_result_loop", ok, finish=d2["choices"][0]["finish_reason"],
                    next_calls=[c["function"]["name"] for c in (m2.get("tool_calls") or [])],
                    content=(m2.get("content") or "")[:160])

    def t_multi_turn(self):
        sid = f"func-{uuid.uuid4().hex[:12]}"
        hdr = {"x-litellm-session-id": sid} if self.gateway else {}
        msgs, ids = [{"role": "user", "content": "Remember this code word: ORCHID-7. Reply 'ok'."}], []
        for follow in ("What is 2+2? Answer briefly.", "What was the code word? Reply with it only."):
            d, h = self.chat(msgs, headers=hdr, max_tokens=1024, chat_template_kwargs={"enable_thinking": False})
            ids.append(h.get("x-litellm-model-id"))
            msgs += [{"role": "assistant", "content": d["choices"][0]["message"]["content"]},
                     {"role": "user", "content": follow}]
        d, h = self.chat(msgs, headers=hdr, max_tokens=1024, chat_template_kwargs={"enable_thinking": False})
        ids.append(h.get("x-litellm-model-id"))
        ans = d["choices"][0]["message"]["content"] or ""
        sticky = (len(set(ids)) == 1) if self.gateway else True
        self.record("multi_turn", "ORCHID-7" in ans and sticky, answer=ans.strip()[:60], model_ids=ids)

    def t_short_prompt_determinism(self):
        """Regression probe for vllm#61486 (stale n-gram IDs in PIECEWISE graphs on AMD):
        greedy short prompts must give identical output regardless of what ran before."""
        prompts = ["Capital of France?", "2+3=?", "Opposite of hot?", "Color of the sky?"]
        outs = {p: set() for p in prompts}
        for _ in range(3):
            for p in prompts:
                d, _ = self.chat([{"role": "user", "content": p}], temperature=0, max_tokens=32,
                                 chat_template_kwargs={"enable_thinking": False})
                outs[p].add(d["choices"][0]["message"]["content"])
            # perturb scheduler state with a longer request between rounds
            self.chat([{"role": "user", "content": "Write a 120-word paragraph about rivers."}],
                      max_tokens=200, chat_template_kwargs={"enable_thinking": False})
        stable = all(len(v) == 1 for v in outs.values())
        self.record("short_prompt_determinism", stable,
                    variants={p: len(v) for p, v in outs.items()},
                    sample={p: sorted(v)[0][:40] for p, v in outs.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--gateway", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = T(a.base, a.key, a.gateway)
    t.run()
    n = sum(r["pass"] for r in t.results)
    summary = {"base": a.base, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "passed": n,
               "total": len(t.results), "results": t.results}
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{n}/{len(t.results)} passed -> {a.out}")
