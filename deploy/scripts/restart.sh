#!/usr/bin/env bash
# Restart one component and wait for health. Prefix cache on that replica is lost (expected).
#   scripts/restart.sh 1|2|3|4|a|b|gw
source "$(dirname "$0")/_lib.sh"
svc=$(svc_of "${1:?usage: restart.sh 1|2|3|4|a|b|gw}")
"${DC[@]}" restart -t 60 "$svc"
[[ $svc == litellm ]] || wait_healthy "$svc"
