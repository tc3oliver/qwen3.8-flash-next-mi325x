#!/usr/bin/env bash
# Idle-server A/B of tool_result_loop pass rate (default sampling T=1.0), 60 trials per (config, GPU pair).
W=${W:-$PWD}
set -u; W=$W; cd $W
until grep -q 'B2 all-done' logs/runb2.log && grep -q 'P512 all-done\|P512-ABORT' logs/run512.log; do sleep 20; done
echo "AB start $(date +%T)"
up(){ NAME=$1 PORT=$2 RENDERS="$3" TAG=$4 YARN=$5 ./up.sh >/dev/null; until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$2/health)" = 200 ]; do [ "$(docker inspect -f '{{.State.Running}}' $1)" = true ] || { echo "AB-ABORT $1 died"; return 1; }; sleep 10; done; }
trial(){ (for k in 1 2 3 4 5 6; do uv run -q --with httpx python trl.py http://127.0.0.1:$2 10 2>&1; done) > logs/ab-$1.log; echo "AB $1: $(grep -h pass logs/ab-$1.log | awk '{s+=$3; n+=$5} END{print s"/"n}')"; }
# pair 45: b2 (already up, 8194) -> y 1M -> 512
( trial 262k-p45 8194
  docker rm -f lc1m-b2 >/dev/null; up lc1m-y 8191 "160 168" a 1 && trial 1m-p45 8191
  docker rm -f lc1m-y >/dev/null; up lc1m-512b 8195 "160 168" a 2 && trial 512k-p45 8195
  docker rm -f lc1m-512b >/dev/null ) &
# pair 67: 512 (already up, 8193) -> 262k
( trial 512k-p67 8193
  docker rm -f lc1m-512 >/dev/null; up lc1m-b 8192 "176 184" b 0 && trial 262k-p67 8192
  docker rm -f lc1m-b >/dev/null ) &
wait; echo "AB all-done $(date +%T)"; docker ps --format '{{.Names}}' | grep lc1m || echo "no test containers left"
