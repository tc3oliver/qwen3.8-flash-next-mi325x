"""Which session-id formats does the gateway pin to one replica?

Each format: 8 sessions x 4 sequential 1-token requests with `x-session-id`; a session
"switched" if it was served by more than one replica (x-litellm-model-id).
Usage: LITELLM_MASTER_KEY=... python3 session_id_probe.py [base_url] > out.json
"""
import concurrent.futures as cf, json, os, random, sys, urllib.request, uuid

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4000"
KEY = os.environ["LITELLM_MASTER_KEY"]


def call(sid):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps({"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json", "x-session-id": sid},
    )
    return urllib.request.urlopen(req, timeout=60).headers.get("x-litellm-model-id")


h = lambda: uuid.uuid4().hex
FORMATS = {
    "t-<hex8>": lambda: "t-" + h()[:8],
    "hex32": lambda: h(),
    "uuid4": lambda: str(uuid.uuid4()),
    "sess-<n>": lambda: "sess-%d" % random.randint(1000, 9999),
    "digits": lambda: str(random.randint(10**8, 10**9)),
    "a_b": lambda: f"{h()[:6]}_{h()[:6]}",
    "a.b": lambda: f"{h()[:6]}.{h()[:6]}",
    "a:b": lambda: f"{h()[:6]}:{h()[:6]}",
    "0.<digits>": lambda: "0.%d" % random.randint(10**8, 10**9),
}

out = {}
for name, make in FORMATS.items():
    def session(_):
        sid = make()
        return [call(sid) for _ in range(4)]
    with cf.ThreadPoolExecutor(8) as ex:
        runs = list(ex.map(session, range(8)))
    out[name] = {"sessions": len(runs), "switched": sum(len(set(r)) > 1 for r in runs), "replicas": ["".join(x[-1] for x in r) for r in runs]}
print(json.dumps(out, indent=1))
