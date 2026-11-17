# Short-prompt decode throughput: N concurrent streams x 512 tokens (ignore_eos, T=0). usage: dec.py BASE N
import asyncio, json, sys, time, httpx
base, n = sys.argv[1], int(sys.argv[2])
async def one(c, i):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": f"[{i}] Write a long essay about the history of bridges."}],
            "max_tokens": 512, "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = None; ct = 0
    async with c.stream("POST", base + "/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): ct = d["usage"]["completion_tokens"]
            if first is None and d.get("choices") and d["choices"][0]["delta"].get("content"): first = time.perf_counter()
    return ct, first - t0, time.perf_counter() - first
async def main():
    async with httpx.AsyncClient(timeout=600) as c:
        await asyncio.gather(*(one(c, -k) for k in range(2)))  # warmup
        t = time.perf_counter(); res = await asyncio.gather(*(one(c, i) for i in range(n))); wall = time.perf_counter() - t
    tok = sum(r[0] for r in res); per = [r[0] / r[2] for r in res]
    print(json.dumps({"n": n, "agg_tok_s": round(tok / wall, 1), "per_stream_tok_s": round(sum(per) / len(per), 1), "ttft_mean_s": round(sum(r[1] for r in res) / n, 3)}))
asyncio.run(main())
