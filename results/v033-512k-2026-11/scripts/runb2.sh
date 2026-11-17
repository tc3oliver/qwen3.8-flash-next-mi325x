#!/usr/bin/env bash
# Control: stock 262K on HIP4+5 (same pair as Y) to separate GPU-pair effect from max-model-len effect in sim50.
R=${R:?repo root}; W=${W:-$PWD}   # run from the work dir that holds these scripts
set -u; cd $W; P=8194
docker rm -f lc1m-y >/dev/null; NAME=lc1m-b2 PORT=$P RENDERS="160 168" TAG=a YARN=0 ./up.sh >/dev/null; echo "B2 launch $(date +%T)"
until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$P/health)" = 200 ]; do [ "$(docker inspect -f '{{.State.Running}}' lc1m-b2)" = true ] || { echo "B2-ABORT died"; exit 1; }; sleep 10; done
echo "B2 healthy $(date +%T)"
uv run -q --with httpx python $R/tools/tests/sim_users.py --base http://127.0.0.1:$P --key x --users 50 --minutes 15 --seed 7 --run simb2 --out data/sim50-b2.json > logs/sim50-b2.log 2>&1
echo "B2 sim: $(python3 -c "import json;s=json.load(open('data/sim50-b2.json'))['summary'];print({k:s[k] for k in ['requests','errors','ttft_cold_p50','ttft_warm_p50','ttft_warm_p95','warm_cache_hit']})" 2>&1)"
echo "B2 all-done $(date +%T)"
