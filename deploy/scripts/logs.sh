#!/usr/bin/env bash
# Follow logs: scripts/logs.sh 1|2|3|4|a|b|gw [extra docker logs args]
source "$(dirname "$0")/_lib.sh"
svc=$(svc_of "${1:?usage: logs.sh 1|2|3|4|a|b|gw}"); shift
"${DC[@]}" logs -f --tail 200 "$@" "$svc"
