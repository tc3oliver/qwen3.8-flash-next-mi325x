#!/usr/bin/env bash
# usage: bisect.sh <name> [VAR=val ...]   boots TP2 1M with the given knobs, runs 8K smoke + 256K, records fault yes/no, removes the container
R=$(cd "$(dirname "$0")/.." && pwd)   # results/long-context-1m-2026-11
name=$1; shift; mkdir -p $R/tp2/bisect/$name
env "$@" $R/tp2/up.sh >/dev/null
for i in $(seq 1 60); do sleep 10; docker inspect -f '{{.State.Status}}' lc1m-tp2 | grep -q running || { echo "$name BOOT_DIED"; break; }; curl -fs localhost:8191/health >/dev/null && break; done
export LC_BASE=http://127.0.0.1:8191
python3 $R/scripts/stage.py --tokens 8192 --label T0_8K --out $R/tp2/bisect/$name > $R/tp2/bisect/$name/client.txt 2>&1
python3 $R/scripts/stage.py --tokens 262144 --label A_256K --out $R/tp2/bisect/$name >> $R/tp2/bisect/$name/client.txt 2>&1
sleep 5; docker logs lc1m-tp2 > $R/tp2/bisect/$name/server.log 2>&1
f=$(grep -c 'Memory access fault' $R/tp2/bisect/$name/server.log); echo "$name faults=$f $(tail -3 $R/tp2/bisect/$name/client.txt | grep -E 'decode:|ERROR' | cut -c1-120)" | tee -a $R/tp2/bisect/summary.txt
docker rm -f lc1m-tp2 >/dev/null 2>&1
