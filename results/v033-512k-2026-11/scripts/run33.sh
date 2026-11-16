#!/usr/bin/env bash
# v0.33 candidates, 512K + QR256 + deploy layout + 2026-11-03 overlay. pair45 = v0.33.0 (ROCm 10), pair67 = v0.33.0-rocm72.
R=${R:?repo root}; W=${W:-$PWD}   # run from the work dir that holds these scripts
set -u; cd $W; D=data
up(){ IMG=$5 OVL33=1 DEPLOY=1 QRMAX=256 YARN=2 NAME=$1 PORT=$2 RENDERS="$3" TAG=$4 ./up.sh >/dev/null
  until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$2/health)" = 200 ]; do [ "$(docker inspect -f '{{.State.Running}}' $1)" = true ] || { echo "V33-ABORT $1 died"; docker logs --tail 25 $1 2>&1 | cut -c1-200; return 1; }; sleep 10; done
  echo "V33 $1 healthy $(date +%T) | $(docker exec $1 python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null) | $(docker logs $1 2>&1 | grep -oE 'Model loading took [0-9.]+ GiB' | tail -1) | $(docker logs $1 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1)"; }
suite(){ n=$1; p=$2
  uv run -q --with httpx python $R/tools/tests/functional.py --base http://127.0.0.1:$p --out $D/functional-$n.json > logs/functional-$n.log 2>&1
  uv run -q --with httpx,datasets python $R/tools/tests/gsm8k_quick.py --base http://127.0.0.1:$p --n 1319 --concurrency 16 --out $D/gsm8k-$n.json > logs/gsm8k-$n.log 2>&1
  echo "V33 $n functional: $(tail -1 logs/functional-$n.log | cut -c1-14) fails: $(grep -h '^FAIL' logs/functional-$n.log | cut -c7-30 | tr '\n' ' ')| $(tail -1 logs/gsm8k-$n.log)"
  echo "V33 $n long_probe: $(python3 long_probe.py http://127.0.0.1:$p 2>&1 | tr '\n' ' ' | cut -c1-200)"
  LC_BASE=http://127.0.0.1:$p python3 $R/results/long-context-1m-2026-11/scripts/stage.py --tokens 500000 --label ${n}_500K --out $D/ > logs/stage-$n.log 2>&1
  echo "V33 $n stage500K: $(grep -oE 'ttft=[0-9.]+s|correct=\{[^}]*\}|TPOT=[0-9.]+ ms' logs/stage-$n.log | tr '\n' ' ')"
  echo "V33 $n decode: c1 $(uv run -q --with httpx python dec.py http://127.0.0.1:$p 1) c16 $(uv run -q --with httpx python dec.py http://127.0.0.1:$p 16)"
  uv run -q --with httpx python $R/tools/tests/sim_users.py --base http://127.0.0.1:$p --key x --users 50 --minutes 15 --seed 7 --run sim$n --out $D/sim50-$n.json > logs/sim50-$n.log 2>&1
  echo "V33 $n sim: $(python3 -c "import json;s=json.load(open('$D/sim50-$n.json'))['summary'];print({k:s[k] for k in ['requests','errors','ttft_cold_p50','ttft_warm_p50','ttft_warm_p95','warm_cache_hit']})" 2>&1)"
  echo "V33 $n faults=$(docker logs lc1m-$n 2>&1 | grep -ciE 'memory access fault|HSA_STATUS|hipError|EngineDead|out of memory') running=$(docker inspect -f '{{.State.Running}}' lc1m-$n)"; }
echo "V33 start $(date +%T)"
( up lc1m-r10 8201 "160 168" 33a vllm/vllm-openai-rocm:v0.33.0 && suite r10 8201
  docker rm -f lc1m-r10 >/dev/null
  # same-harness decode baseline: v0.32.0 + d02 overlay, 512K
  IMG=vllm/vllm-openai-rocm:v0.32.0 DEPLOY=1 QRMAX=256 YARN=2 NAME=lc1m-b32 PORT=8203 RENDERS="160 168" TAG=a ./up.sh >/dev/null
  until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8203/health)" = 200 ]; do sleep 10; done
  echo "V33 b32 decode: c1 $(uv run -q --with httpx python dec.py http://127.0.0.1:8203 1) c16 $(uv run -q --with httpx python dec.py http://127.0.0.1:8203 16)"
  docker rm -f lc1m-b32 >/dev/null ) &
( up lc1m-r72 8202 "176 184" 33b vllm/vllm-openai-rocm:v0.33.0-rocm72 && suite r72 8202
  docker rm -f lc1m-r72 >/dev/null ) &
wait; echo "V33 all-done $(date +%T)"
