#!/usr/bin/env bash
# 512K (YaRN factor 2) deployment candidate on HIP6+7 once the baseline B is done (phase 3 uses only Y).
R=${R:?repo root}; W=${W:-$PWD}   # run from the work dir that holds these scripts
set -u; W=$W; T=$R/tools/tests; S=$R/results/long-context-1m-2026-11/scripts; D=$W/data; cd $W; P=8193
until grep -q 'PHASE 3-start\|PHASE all-done' logs/run.log; do sleep 20; done
docker rm -f lc1m-b >/dev/null; echo "P512 launch $(date +%T)"
NAME=lc1m-512 PORT=$P RENDERS="176 184" TAG=b YARN=2 ./up.sh >/dev/null
until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$P/health)" = 200 ]; do [ "$(docker inspect -f '{{.State.Running}}' lc1m-512)" = true ] || { echo "P512-ABORT died"; docker logs --tail 20 lc1m-512 2>&1 | cut -c1-200; exit 1; }; sleep 10; done
echo "P512 healthy $(date +%T) $(docker logs lc1m-512 2>&1 | grep -oE 'GPU KV cache size.*' | tail -1)"
uv run -q --with httpx python $T/functional.py --base http://127.0.0.1:$P --out $D/functional-512.json > logs/functional-512.log 2>&1
uv run -q --with httpx,datasets python $T/gsm8k_quick.py --base http://127.0.0.1:$P --n 1319 --concurrency 16 --out $D/gsm8k-512.json > logs/gsm8k-512.log 2>&1
echo "P512 functional: $(tail -1 logs/functional-512.log | cut -c1-60) | $(tail -1 logs/gsm8k-512.log)"
LC_BASE=http://127.0.0.1:$P python3 $S/stage.py --tokens 500000 --label P512_500K --out $D/ > logs/stage512-500000.log 2>&1
echo "P512 stage500K: $(tail -2 logs/stage512-500000.log | tr '\n' ' ' | cut -c1-300)"
uv run -q --with httpx python $T/sim_users.py --base http://127.0.0.1:$P --key x --users 50 --minutes 15 --seed 7 --run sim512 --out $D/sim50-512.json > logs/sim50-512.log 2>&1
echo "P512 sim: $(python3 -c "import json;s=json.load(open('$D/sim50-512.json'))['summary'];print({k:s[k] for k in ['requests','errors','ttft_cold_p50','ttft_warm_p50','ttft_warm_p95','warm_cache_hit']})" 2>&1)"
echo "P512 faults: $(docker logs lc1m-512 2>&1 | grep -ciE 'memory access fault|HSA_STATUS|hipError|EngineDead') running=$(docker inspect -f '{{.State.Running}}' lc1m-512)"
echo "P512 all-done $(date +%T)"
