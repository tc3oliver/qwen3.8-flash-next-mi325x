#!/usr/bin/env bash
# Full correctness gate for a finalist engine already running on <port>:
# functional (9 checks incl. #61486 determinism), full GSM8K (1319, zero-shot), 238K needle x3.
#   experiments/gate.sh <port> <name>
set -uo pipefail
cd "$(dirname "$0")/.."
port=$1 name=$2; B=http://127.0.0.1:$port; R=experiments/results
uv run -q --with httpx tests/functional.py --base $B --out $R/$name.functional.json 2>&1 | tail -1
uv run -q --with httpx,datasets tests/gsm8k_quick.py --base $B --n 1319 --concurrency 64 --out $R/$name.gsm8k1319.json 2>&1 | grep -v Warning
uv run -q --with httpx tests/long_needle.py --base $B --tokens 238000 --out $R/$name.needle.json 2>&1 | tail -1
