#!/usr/bin/env bash
# Phases: 1 functional+GSM8K (Y,B parallel)  2 sim 50 users x 15 min (Y,B parallel)  3 long-context stages on Y only
R=${R:?repo root}; W=${W:-$PWD}   # run from the work dir that holds these scripts
set -u; W=$W; T=$R/tools/tests; S=$R/results/long-context-1m-2026-11/scripts; D=$W/data; cd $W
alive(){ for c in "$@"; do [ "$(docker inspect -f '{{.State.Running}}' $c)" = true ] || { echo "PHASE-ABORT $c not running"; return 1; }; done; }
ph(){ echo "PHASE $* $(date +%T)"; }
if [ "${1:-1}" -le 1 ]; then
  ph 1-start
  for x in y:8191 b:8192; do t=${x%%:*}; p=${x##*:}
    ( uv run -q --with httpx python $T/functional.py --base http://127.0.0.1:$p --out $D/functional-$t.json > logs/functional-$t.log 2>&1
      uv run -q --with httpx,datasets python $T/gsm8k_quick.py --base http://127.0.0.1:$p --n 1319 --concurrency 16 --out $D/gsm8k-$t.json > logs/gsm8k-$t.log 2>&1
      echo "PHASE 1-done-$t: $(tail -1 logs/functional-$t.log | cut -c1-120) | $(tail -1 logs/gsm8k-$t.log)" ) &
  done; wait; alive lc1m-y lc1m-b || true
fi
if [ "${1:-1}" -le 2 ]; then
  ph 2-start
  for x in y:8191 b:8192; do t=${x%%:*}; p=${x##*:}
    ( uv run -q --with httpx python $T/sim_users.py --base http://127.0.0.1:$p --key x --users 50 --minutes 15 --seed 7 --run sim$t --out $D/sim50-$t.json > logs/sim50-$t.log 2>&1
      echo "PHASE 2-done-$t: $(python3 -c "import json;s=json.load(open('$D/sim50-$t.json'))['summary'];print({k:s[k] for k in ['requests','errors','ttft_cold_p50','ttft_warm_p50','ttft_warm_p95','warm_cache_hit','mean_prompt_tokens_warm']})" 2>&1)" ) &
  done; wait; alive lc1m-y lc1m-b || true
fi
if [ "${1:-1}" -le 3 ]; then
  ph 3-start
  for n in 500000 985000; do
    alive lc1m-y || break
    LC_BASE=http://127.0.0.1:8191 python3 $S/stage.py --tokens $n --label Y_$((n/1000))K --out $D/ > logs/stage-$n.log 2>&1
    echo "PHASE 3-done-$n: $(tail -2 logs/stage-$n.log | tr '\n' ' ' | cut -c1-300)"
  done; alive lc1m-y || true
fi
ph all-done
