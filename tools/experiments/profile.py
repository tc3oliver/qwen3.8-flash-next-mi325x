#!/usr/bin/env python3
"""Profile one phase on an engine started with --profiler-config (torch, dir /profiles/<name>).

  prefill: one cold 32K prompt, max_tokens 1
  decode : 16 concurrent ~1K prompts generating 256 tokens (profiling window starts after their prefill)

    uv run --with httpx experiments/profile.py --base http://127.0.0.1:8011 --phase prefill
"""
import argparse, asyncio, random, sys, time, uuid
import httpx
sys.path.insert(0, __import__("os").path.dirname(__file__))
from perfsuite import code, req  # noqa: E402


async def main(a):
    n = uuid.uuid4().hex[:8]
    async with httpx.AsyncClient(timeout=3600) as c:
        await req(c, a.base, [{"role": "user", "content": "warm " + code(random.Random(n), 8000)}], 4)
        if a.phase == "prefill":
            await c.post(f"{a.base}/start_profile")
            await req(c, a.base, [{"role": "user", "content": f"[{n}]" + code(random.Random(n + "x"), 32000)}], 1)
            await c.post(f"{a.base}/stop_profile")
        else:
            tasks = [asyncio.create_task(req(c, a.base, [{"role": "user", "content": f"[{n}{i}]" + code(random.Random(n + str(i)), 1000)}], 384)) for i in range(16)]
            await asyncio.sleep(4)
            await c.post(f"{a.base}/start_profile")
            await asyncio.sleep(2)
            await c.post(f"{a.base}/stop_profile")
            await asyncio.gather(*tasks)
    print("profiled", a.phase)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--phase", choices=["prefill", "decode"], required=True)
    asyncio.run(main(ap.parse_args()))
