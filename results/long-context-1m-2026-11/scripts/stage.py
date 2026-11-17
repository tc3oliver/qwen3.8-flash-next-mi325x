#!/usr/bin/env python3
"""One long-context stage against an isolated vLLM endpoint (never the gateway).
usage: stage.py --tokens 262144 --label A_256K --out data/ [--seed N]
Builds a haystack of unique log lines, hides 3 passcodes (10%/50%/90% depth) and 1 unique log line (70%),
asks for all four. Prompt tokens are the server's own count (usage.prompt_tokens). Request 1 = cold prefill + retrieval
(TTFT, prefill tok/s); request 2 = same prompt, ignore_eos, 256 tokens (decode tok/s / TPOT at long context)."""
import argparse, json, random, subprocess, threading, time, urllib.request, os, re

BASE = os.environ.get("LC_BASE", "http://127.0.0.1:8190")
def post(path, body, stream=False, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def ntok(text):
    return len(json.load(post("/tokenize", {"model": "qwen3.8-flash-next", "prompt": text, "add_special_tokens": False}))["tokens"])

WORDS = "amber basalt cobalt dune ember fjord garnet harbor iris juniper kelp lantern marble nickel onyx pewter quartz russet sable tundra umber velvet willow xenon yarrow zephyr".split()
def make_line(rng, i):
    return (f"Log entry {i:07d}: station {rng.choice(WORDS)}-{rng.randrange(100,999)} reported pressure {rng.uniform(1,9):.2f} bar "
            f"at {rng.randrange(24):02d}:{rng.randrange(60):02d} and valve V-{rng.randrange(1,99)} stayed {rng.choice(['open','closed','half open'])}.")

def build(target, seed):
    rng = random.Random(seed)
    sample = "\n".join(make_line(rng, i) for i in range(2000))
    per_line = ntok(sample) / 2000
    overhead = 400  # question + template, corrected below from usage
    n = int((target - overhead) / per_line)
    codes = {k: f"{rng.choice(WORDS)}-{rng.randrange(1000,9999)}-{rng.choice(WORDS)}" for k in ("ALPHA", "BRAVO", "CHARLIE")}
    uniq_i = int(n * 0.70); uniq_station = f"zz{rng.randrange(1000,9999)}"; uniq_val = f"{rng.uniform(10,99):.3f}"
    lines = []
    for i in range(n):
        l = make_line(rng, i)
        if i == int(n*0.10): l += f" NOTE: the vault passcode for project ALPHA is '{codes['ALPHA']}'."
        if i == int(n*0.50): l += f" NOTE: the vault passcode for project BRAVO is '{codes['BRAVO']}'."
        if i == int(n*0.90): l += f" NOTE: the vault passcode for project CHARLIE is '{codes['CHARLIE']}'."
        if i == uniq_i: l = f"Log entry {i:07d}: station {uniq_station} reported pressure {uniq_val} bar at 03:17 and valve V-7 stayed closed."
        lines.append(l)
    q = (f"\n\nAnswer from the log above only. What are the vault passcodes for projects ALPHA, BRAVO and CHARLIE, and what pressure did "
         f"station {uniq_station} report? Reply exactly as: ALPHA=..., BRAVO=..., CHARLIE=..., PRESSURE=...")
    exp = dict(codes); exp["PRESSURE"] = uniq_val
    return "\n".join(lines) + q, exp, dict(lines=n, per_line=per_line)

class HBM(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop=False; self.peak={}; self.samples=[]
    def run(self):
        while not self.stop:
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True).stdout
            row = {}
            for m in re.finditer(r"GPU\[(\d)\]\s*:\s*VRAM Total Used Memory \(B\): (\d+)", out):
                g = int(m.group(1))
                if 4 <= g <= 7: row[g] = int(m.group(2)) / 2**30
            if row:
                self.samples.append((time.time(), row))
                for g, v in row.items(): self.peak[g] = max(self.peak.get(g, 0), v)
            time.sleep(3)

def chat(prompt, max_tokens, ignore_eos=False):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos: body["ignore_eos"] = True
    t0 = time.time(); first = None; last = None; n = 0; text = ""; usage = None
    for raw in post("/v1/chat/completions", body):
        raw = raw.strip()
        if not raw.startswith(b"data:") or raw == b"data: [DONE]": continue
        d = json.loads(raw[5:])
        if d.get("usage"): usage = d["usage"]
        for c in d.get("choices", []):
            piece = (c["delta"].get("content") or "") + (c["delta"].get("reasoning") or c["delta"].get("reasoning_content") or "")
            if piece:
                now = time.time()
                if first is None: first = now
                last = now; n += 1; text += piece
    return dict(ttft=(first - t0) if first else None, total=time.time() - t0, chunks=n, text=text, usage=usage,
                decode_s=(last - first) if first and last else None)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tokens", type=int, required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=None); ap.add_argument("--decode-tokens", type=int, default=256)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else a.tokens
    prompt, exp, meta = build(a.tokens, seed)
    for _ in range(2):  # calibrate with the server tokenizer (template adds ~30 tokens)
        actual = ntok(prompt) + 30
        if abs(actual - a.tokens) / a.tokens < 0.003: break
        prompt, exp, meta = build(int(a.tokens * a.tokens / actual), seed)
    print(f"[{a.label}] built {meta['lines']} lines, {len(prompt)} chars", flush=True)
    h = HBM(); h.start(); res = dict(label=a.label, target=a.tokens, seed=seed, expected=exp, start=time.strftime("%F %T"))
    try:
        r1 = chat(prompt, 128)
        pt = r1["usage"]["prompt_tokens"]
        ok = {k: (v in r1["text"]) for k, v in exp.items()}
        res.update(prompt_tokens=pt, ttft_s=r1["ttft"], total_s=r1["total"], prefill_tok_s=pt / r1["ttft"], answer=r1["text"], correct=ok,
                   completion_tokens=r1["usage"]["completion_tokens"])
        print(f"[{a.label}] prompt_tokens={pt} ttft={r1['ttft']:.1f}s prefill={pt/r1['ttft']:.0f} tok/s correct={ok}\n  answer={r1['text']!r}", flush=True)
        r2 = chat(prompt, a.decode_tokens, ignore_eos=True)
        ct = r2["usage"]["completion_tokens"]; tpot = r2["decode_s"] / (ct - 1) if ct > 1 else None
        res.update(decode_req=dict(ttft_s=r2["ttft"], completion_tokens=ct, tpot_ms=tpot * 1000 if tpot else None,
                                   decode_tok_s=1 / tpot if tpot else None, cached_tokens=(r2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")))
        print(f"[{a.label}] decode: ttft(2nd)={r2['ttft']:.1f}s tokens={ct} TPOT={tpot*1000:.1f} ms ({1/tpot:.1f} tok/s)", flush=True)
    except Exception as e:
        res["error"] = repr(e); print(f"[{a.label}] ERROR {e!r}", flush=True)
    finally:
        h.stop = True; time.sleep(0.2)
        res["hbm_peak_gib"] = h.peak; res["end"] = time.strftime("%F %T")
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, f"{a.label}.json"), "w"), indent=1)
