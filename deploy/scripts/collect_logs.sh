#!/usr/bin/env bash
# Snapshot container logs to $LOG_DIR/<svc>-<ts>.log (default deploy/logs/)
source "$(dirname "$0")/_lib.sh"
ts=$(date +%Y%m%d-%H%M%S)
(( $# )) || set -- "${REPLICAS[@]}" litellm
for svc in "$@"; do
  c=$("${DC[@]}" ps -aq "$svc" 2>/dev/null) || continue; [[ -n $c ]] || continue
  docker logs "$c" > "$LOG_DIR/$svc-$ts.log" 2>&1 && echo "saved $LOG_DIR/$svc-$ts.log"
done
