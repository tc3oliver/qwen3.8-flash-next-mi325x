import sys, json, httpx, collections
# Exact functional.py t_short_prompt_determinism loop, repeated; records every distinct answer per prompt.
base, n = sys.argv[1], int(sys.argv[2]); c = httpx.Client(timeout=120)
P = ["Capital of France?", "2+3=?", "Opposite of hot?", "Color of the sky?"]
def chat(msgs, **kw):
    return c.post(base + "/v1/chat/completions", json={"model": "qwen3.8-flash-next", "messages": msgs, **kw}).json()["choices"][0]["message"]["content"]
fails = 0; allv = {p: collections.Counter() for p in P}
for i in range(n):
    outs = {p: set() for p in P}
    for _ in range(3):
        for p in P:
            o = chat([{"role": "user", "content": p}], temperature=0, max_tokens=32, chat_template_kwargs={"enable_thinking": False})
            outs[p].add(o); allv[p][o] += 1
        chat([{"role": "user", "content": "Write a 120-word paragraph about rivers."}], max_tokens=200, chat_template_kwargs={"enable_thinking": False})
    fails += any(len(v) > 1 for v in outs.values())
print(base, "runs", n, "failed", fails)
for p, cn in allv.items():
    if len(cn) > 1:
        for t, k in cn.most_common(): print("  ", p, k, repr(t))
