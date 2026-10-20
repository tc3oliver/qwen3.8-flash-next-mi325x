#!/usr/bin/env bash
# One full experiment: start engine (exp.sh args), GSM8K-200 gate, perfsuite, save, keep server log.
#   experiments/run.sh <slot> <name> [exp.sh args...]
set -uo pipefail
cd "$(dirname "$0")/.."
slot=$1 name=$2; case $slot in a) PORT=8011;; b) PORT=8012;; g0) PORT=8020;; g1) PORT=8021;; g2) PORT=8022;; g3) PORT=8023;; esac; R=experiments/results
experiments/exp.sh "$@" || { echo "{\"label\":\"$name\",\"error\":\"start failed\"}" > $R/$name.json; exit 1; }
B=http://127.0.0.1:$PORT
uv run -q --with httpx,datasets tests/gsm8k_quick.py --base $B --n 200 --concurrency 32 --out $R/$name.gsm8k.json 2>&1 | grep -v Warning
uv run -q --with httpx experiments/perfsuite.py --base $B --label "$name" ${PERF_ONLY:+--only $PERF_ONLY} --out $R/$name.json
docker logs qwen38-exp-$slot > $R/$name.server.log 2>&1
python3 - "$R/$name.json" "$R/$name.gsm8k.json" <<'PY'
import json,sys; d=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2])); d["gsm8k200"]=g["accuracy"]; json.dump(d,open(sys.argv[1],"w"),indent=1)
PY
echo "[$name] done"
