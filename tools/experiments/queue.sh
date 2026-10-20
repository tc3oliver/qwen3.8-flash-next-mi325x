#!/usr/bin/env bash
# Run experiments from a queue file one by one on a slot. Each line: <name> [exp.sh args...]
# Lines are consumed from the top (the file is re-read before each run, so it can be appended to).
#   experiments/queue.sh <slot> <queue-file>
set -uo pipefail
cd "$(dirname "$0")/.."
slot=$1 q=$2
while :; do
  line=$(grep -v '^\s*#' "$q" | grep -v '^\s*$' | head -1)
  [[ -z $line ]] && { echo "[queue $slot] empty, exiting"; break; }
  # remove that line from the queue (first exact match)
  python3 - "$q" "$line" <<'PY'
import sys; p, l = sys.argv[1], sys.argv[2]; s = open(p).read().split("\n")
s.remove(l); open(p, "w").write("\n".join(s))
PY
  eval "set -- $line"; name=$1; shift
  echo "[queue $slot] $(date +%T) start $name"
  experiments/run.sh "$slot" "$name" "$@" > "experiments/results/$name.run.log" 2>&1
  echo "[queue $slot] $(date +%T) end $name: $(tail -1 experiments/results/$name.run.log)"
done
docker rm -f qwen38-exp-$slot >/dev/null 2>&1
