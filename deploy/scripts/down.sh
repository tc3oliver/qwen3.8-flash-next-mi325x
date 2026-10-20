#!/usr/bin/env bash
# Stop and remove ONLY this stack's containers (never `compose down`, never prune).
#   scripts/down.sh                  LiteLLM + all replicas of LAYOUT
#   scripts/down.sh 1|2|3|4|a|b|gw   one component
source "$(dirname "$0")/_lib.sh"
if [[ $# -gt 0 ]]; then svcs=("$(svc_of "$1")"); else svcs=(litellm "${REPLICAS[@]}"); fi
"$SVC_DIR/scripts/collect_logs.sh" "${svcs[@]}" || true
"${DC[@]}" stop -t 60 "${svcs[@]}"
"${DC[@]}" rm -f "${svcs[@]}"
