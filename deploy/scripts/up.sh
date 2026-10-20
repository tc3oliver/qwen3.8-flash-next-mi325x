#!/usr/bin/env bash
# Start the stack: all replicas of LAYOUT (in parallel) -> wait until each is healthy -> LiteLLM.
#   scripts/up.sh              all
#   scripts/up.sh 1|2|3|4|a|b  one replica      scripts/up.sh gw   LiteLLM only
source "$(dirname "$0")/_lib.sh"
mkdir -p runtime/aiter; for r in "${REPLICAS[@]}"; do mkdir -p "runtime/$r"; done
if [[ $# -gt 0 ]]; then
  s=$(svc_of "$1"); "${DC[@]}" up -d --no-deps "$s"; [[ $s == litellm ]] || wait_healthy "$s"; exit
fi
"${DC[@]}" up -d --no-deps "${REPLICAS[@]}"
for r in "${REPLICAS[@]}"; do wait_healthy "$r"; done
"${DC[@]}" up -d --no-deps litellm
"$SVC_DIR/scripts/status.sh"
