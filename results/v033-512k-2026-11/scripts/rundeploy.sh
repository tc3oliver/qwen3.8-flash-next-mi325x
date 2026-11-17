#!/usr/bin/env bash
# Boot the exact deploy layout ($W/deploy/model-yarn512 from make_model_yarn512.sh, /models-orig symlinks, 512K, QR 256) and smoke it.
R=${R:?repo root}; W=${W:-$PWD}   # run from the work dir that holds these scripts
set -u; cd $W
until grep -q 'QR all-done' logs/runqr.log; do sleep 15; done
echo "DEP launch $(date +%T)"
NAME=lc1m-dep PORT=8198 RENDERS="160 168" TAG=a YARN=2 QRMAX=256 DEPLOY=1 ./up.sh >/dev/null
until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8198/health)" = 200 ]; do [ "$(docker inspect -f '{{.State.Running}}' lc1m-dep)" = true ] || { echo "DEP-ABORT died"; docker logs --tail 20 lc1m-dep 2>&1 | cut -c1-200; exit 1; }; sleep 10; done
echo "DEP healthy $(date +%T) | $(docker logs lc1m-dep 2>&1 | grep -oE 'GPU KV cache size.*' | tail -1) | links via: $(docker exec lc1m-dep readlink /models/qwen3.8-flash-next-fp8/LICENSE)"
uv run -q --with httpx python $R/tools/tests/functional.py --base http://127.0.0.1:8198 --out data/functional-dep.json > logs/functional-dep.log 2>&1
echo "DEP functional: $(tail -1 logs/functional-dep.log | cut -c1-40) | fails: $(grep -h '^FAIL' logs/functional-dep.log | cut -c1-80 | tr '\n' ';')"
LC_BASE=http://127.0.0.1:8198 python3 $R/results/long-context-1m-2026-11/scripts/stage.py --tokens 300000 --label DEP_300K --out data/ > logs/stagedep-300000.log 2>&1
echo "DEP stage300K: $(tail -2 logs/stagedep-300000.log | tr '\n' ' ' | cut -c1-260)"
echo "DEP faults=$(docker logs lc1m-dep 2>&1 | grep -ciE 'memory access fault|HSA_STATUS|hipError|EngineDead|out of memory')"
docker rm -f lc1m-dep >/dev/null; echo "DEP all-done $(date +%T)"
