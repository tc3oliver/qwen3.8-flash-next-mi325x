# python3 long_probe.py http://127.0.0.1:<port>
import json, random, sys, time, urllib.request
base = sys.argv[1]; rng = random.Random(1)
lines = [f"Log {i:07d}: station s{rng.randrange(1000)} pressure {rng.uniform(1,9):.2f} bar." for i in range(12600)]
lines[1260] += " NOTE: passcode ALPHA is 'amber-4821-kelp'."; lines[10000] += " NOTE: passcode BRAVO is 'onyx-7730-dune'."
prompt = "\n".join(lines) + "\n\nWhat are passcodes ALPHA and BRAVO? Reply as ALPHA=..., BRAVO=..."
def chat(mt, ign):
    b = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0,
         "ignore_eos": ign, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
    return r, time.time() - t
r, dt = chat(64, False); u = r["usage"]; a = r["choices"][0]["message"]["content"]
print("prompt_tokens", u["prompt_tokens"], "time", round(dt, 1), "ok", "amber-4821-kelp" in a and "onyx-7730-dune" in a, repr(a))
r, dt = chat(512, True); u = r["usage"]; print("crossing: positions", u["prompt_tokens"] + u["completion_tokens"], "time", round(dt, 1))
