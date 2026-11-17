#!/usr/bin/env bash
# Final candidate: 512K (YaRN f=2) + QuickReduce max 256 MB. Pair45: functional+GSM8K+500K; pair67: sim50.
R=${R:?repo root}; W=${W:-$PWD}   # run from the work dir that holds these scripts
set -u; cd $W; D=data
up(){ NAME=$1 PORT=$2 RENDERS="$3" TAG=$4 YARN=2 QRMAX=256 ./up.sh >/dev/null; until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$2/health)" = 200 ]; do [ "$(docker inspect -f '{{.State.Running}}' $1)" = true ] || { echo "QR-ABORT $1 died"; docker logs --tail 20 $1 2>&1 | cut -c1-200; return 1; }; sleep 10; done
  echo "QR $1 healthy $(date +%T) | $(docker logs $1 2>&1 | grep -oE 'GPU KV cache size.*' | tail -1) | env=$(docker exec $1 printenv VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB) | $(docker logs $1 2>&1 | grep -m1 -oE "Using \[[^]]*\] all-reduce backends[^']*'tp:0'")"; }
echo "QR start $(date +%T)"
( up lc1m-qr45 8196 "160 168" a || exit 1
  uv run -q --with httpx python $R/tools/tests/functional.py --base http://127.0.0.1:8196 --out $D/functional-qr.json > logs/functional-qr.log 2>&1
  uv run -q --with httpx,datasets python $R/tools/tests/gsm8k_quick.py --base http://127.0.0.1:8196 --n 1319 --concurrency 16 --out $D/gsm8k-qr.json > logs/gsm8k-qr.log 2>&1
  echo "QR functional: $(tail -1 logs/functional-qr.log | cut -c1-40) | $(tail -1 logs/gsm8k-qr.log)"
  LC_BASE=http://127.0.0.1:8196 python3 $R/results/long-context-1m-2026-11/scripts/stage.py --tokens 500000 --label QR_500K --out $D/ > logs/stageqr-500000.log 2>&1
  echo "QR stage500K: $(tail -2 logs/stageqr-500000.log | tr '\n' ' ' | cut -c1-260)" ) &
( up lc1m-qr67 8197 "176 184" b || exit 1
  uv run -q --with httpx python $R/tools/tests/sim_users.py --base http://127.0.0.1:8197 --key x --users 50 --minutes 15 --seed 7 --run simqr --out $D/sim50-qr.json > logs/sim50-qr.log 2>&1
  echo "QR sim: $(python3 -c "import json;s=json.load(open('$D/sim50-qr.json'))['summary'];print({k:s[k] for k in ['requests','errors','ttft_cold_p50','ttft_warm_p50','ttft_warm_p95','warm_cache_hit']})" 2>&1)" ) &
wait
for c in lc1m-qr45 lc1m-qr67; do echo "QR $c faults=$(docker logs $c 2>&1 | grep -ciE 'memory access fault|HSA_STATUS|hipError|EngineDead|out of memory') running=$(docker inspect -f '{{.State.Running}}' $c 2>/dev/null)"; done
docker rm -f lc1m-qr45 lc1m-qr67 >/dev/null 2>&1; echo "QR all-done $(date +%T)"
