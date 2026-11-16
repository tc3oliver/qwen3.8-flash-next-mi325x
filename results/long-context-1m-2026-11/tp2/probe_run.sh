#!/usr/bin/env bash
# usage: probe_run.sh <name> [VAR=val ...]: boot, probe below 262144 (200K+1500), probe crossing it (262K+1500), log fault yes/no
R=$(cd "$(dirname "$0")/.." && pwd)   # results/long-context-1m-2026-11
name=$1; shift; D=$R/tp2/rootcause/$name; mkdir -p $D
env "$@" $R/tp2/up.sh >/dev/null
for i in $(seq 1 60); do sleep 10; docker inspect -f '{{.State.Status}}' lc1m-tp2 | grep -q running || { echo "$name BOOT_DIED"; break; }; curl -fs localhost:8191/health >/dev/null && break; done
export LC_BASE=http://127.0.0.1:8191
python3 $R/scripts/probe.py --tokens 200000 --decode 1500 --seed 11 --label below --out $D > $D/client.txt 2>&1
python3 $R/scripts/probe.py --tokens 262000 --decode 1500 --seed 12 --label crossing --out $D >> $D/client.txt 2>&1
sleep 5; docker logs lc1m-tp2 > $D/server.log 2>&1
echo "$name faults=$(grep -c 'Memory access fault' $D/server.log) | $(grep -h -o '"label": "[a-z]*"\|"ok": [a-z]*\|"total_positions": [0-9]*' $D/client.txt | paste -sd' ')" | tee -a $R/tp2/rootcause/summary.txt
docker rm -f lc1m-tp2 >/dev/null 2>&1
