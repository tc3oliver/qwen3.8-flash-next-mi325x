#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock that records request headers.

Used to verify which session headers OpenCode / Pi Agent actually send,
without needing the real model. Writes one JSON line per request.

    python3 tests/header_capture.py --port 9099 --out headers.jsonl
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "qwen3.8-flash-next"
REDACT = {"authorization", "x-api-key", "api-key"}


class Handler(BaseHTTPRequestHandler):
    out = None

    def log_message(self, *a):
        pass

    def _record(self, body):
        headers = {k.lower(): ("<redacted>" if k.lower() in REDACT else v) for k, v in self.headers.items()}
        rec = {"ts": time.time(), "method": self.command, "path": self.path, "headers": headers,
               "stream": bool(body.get("stream")), "n_messages": len(body.get("messages", []))}
        with open(self.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._record({})
        if self.path.rstrip("/").endswith("/models"):
            return self._json({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "mock"}]})
        self._json({"ok": True})

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        self._record(body)
        text = "OK from header-capture mock."
        if not body.get("stream"):
            return self._json({
                "id": "mock-1", "object": "chat.completion", "created": int(time.time()), "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6}})
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        base = {"id": "mock-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL}
        for delta, fin in (({"role": "assistant", "content": text}, None), ({}, "stop")):
            chunk = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": fin}])
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = dict(base, choices=[], usage={"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6})
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Handler.out = a.out
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
