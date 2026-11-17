import os
import sys, json, importlib.util
spec = importlib.util.spec_from_file_location("f", os.path.join(os.environ["R"], "tools/tests/functional.py")); f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
base, n = sys.argv[1], int(sys.argv[2]); res = []
for i in range(n):
    t = f.T(base, "", False); t.results = [] if not hasattr(t, "results") else t.results
    rec = {}
    t.record = lambda name, ok, **kw: rec.setdefault(name, (ok, kw))
    t.t_tool_loop()
    ok, kw = rec.get("tool_result_loop", (None, {}))
    res.append(ok)
    if not ok: print("FAIL", json.dumps(kw)[:300])
print(base, "pass", sum(bool(x) for x in res), "/", n)
