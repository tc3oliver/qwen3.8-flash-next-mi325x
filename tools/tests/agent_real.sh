#!/usr/bin/env bash
# Real agent task through the LiteLLM gateway with OpenCode or Pi Agent (non-interactive).
# Task: fix tests/agenttask/inventory/stock.py so its pytest suite passes (4 of 5 tests fail).
# Isolated HOME/config per run (never touches the user's agent configs); records the per-replica
# request deltas from vLLM /metrics to show which replica(s) served the session.
#   tests/agent_real.sh opencode|pi <label> [base_url]
set -uo pipefail
SVC=$(cd "$(dirname "$0")/.." && pwd); set -a; source "$SVC/../deploy/.env"; set +a
agent=$1 label=$2 base=${3:-http://127.0.0.1:${LITELLM_PORT}/v1}
out=$SVC/tests/results; mkdir -p "$out"
work=$(mktemp -d /tmp/qwen38-agent-XXXX); home=$work/home; mkdir -p "$home"
cp -r "$SVC/tests/agenttask/." "$work/repo"; cd "$work/repo"
NODE_BIN=$(PATH="$HOME/.local/share/mise/shims:$PATH" mise which node 2>/dev/null | xargs dirname)   # resolve before HOME changes
uv venv -q "$work/venv" && uv pip install -q --python "$work/venv/bin/python" pytest
export HOME=$home LITELLM_API_KEY=$LITELLM_MASTER_KEY PATH="$work/venv/bin:$NODE_BIN:$PATH"
ports=(); case ${LAYOUT:-tp1x4} in tp1x4) ports=($R1_PORT $R2_PORT $R3_PORT $R4_PORT);; tp2x2) ports=($REPLICA_A_PORT $REPLICA_B_PORT);; esac
reqs() { for p in "${ports[@]}"; do curl -fsS "http://127.0.0.1:$p/metrics" 2>/dev/null | awk '/^vllm:request_success_total/ {s+=$NF} END {printf "%d ", s}'; done; }
before=$(reqs)
prompt="The pytest suite in tests/ fails. Fix the bugs in inventory/stock.py so that all tests pass. Do not modify the tests. Run the tests with: python3 -m pytest -q   Stop when they all pass."
t0=$(date +%s)
if [[ $agent == opencode ]]; then
  sed "s#http://LAB_HOST:4000/v1#$base#" "$SVC/../deploy/config/agents/opencode.json" > opencode.json
  timeout 1200 opencode run --print-logs --model mi325x-lab/qwen3.8-flash-next --format json "$prompt" < /dev/null > "$out/$label.opencode.jsonl" 2> "$out/$label.opencode.err"; rc=$?
else
  mkdir -p "$home/.pi/agent"; sed "s#http://LAB_HOST:4000/v1#$base#" "$SVC/../deploy/config/agents/pi-models.json" > "$home/.pi/agent/models.json"
  timeout 1200 pi -p --no-session --provider mi325x-lab --model qwen3.8-flash-next --mode json "$prompt" < /dev/null > "$out/$label.pi.jsonl" 2> "$out/$label.pi.err"; rc=$?
fi
dt=$(( $(date +%s) - t0 ))
after=$(reqs)
"$work/venv/bin/python" -m pytest -q > "$out/$label.pytest.txt" 2>&1; prc=$?
tests_ok=$([[ $prc == 0 ]] && echo PASS || echo FAIL)
tests_untouched=$(diff -rq -x __pycache__ "$SVC/tests/agenttask/tests" tests >/dev/null && echo yes || echo NO)
read -ra b <<< "$before"; read -ra a <<< "$after"; d=(); for i in "${!a[@]}"; do d+=($(( a[i] - b[i] ))); done
echo "{\"agent\":\"$agent\",\"label\":\"$label\",\"rc\":$rc,\"seconds\":$dt,\"tests\":\"$tests_ok\",\"tests_untouched\":\"$tests_untouched\",\"requests_per_replica\":[$(IFS=,; echo "${d[*]}")],\"pytest\":\"$(tail -1 "$out/$label.pytest.txt")\"}" | tee "$out/$label.json"
rm -rf "$work"
