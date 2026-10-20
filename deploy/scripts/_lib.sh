# Shared helpers for the operation scripts. Sourced, not executed.
set -euo pipefail
SVC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SVC_DIR"
[[ -f .env ]] || { echo ".env missing (cp .env.example .env && chmod 600 .env)"; exit 1; }
set -a; source .env; set +a
LOG_DIR=${LOG_DIR:-$SVC_DIR/logs}; mkdir -p "$LOG_DIR"
LAYOUT=${LAYOUT:-tp1x4}
case $LAYOUT in
  tp1x4) REPLICAS=(vllm-1 vllm-2 vllm-3 vllm-4) ;;
  tp2x2) REPLICAS=(vllm-a vllm-b) ;;
  *) echo "LAYOUT must be tp1x4 or tp2x2 (got '$LAYOUT')" >&2; exit 1 ;;
esac
DC=(docker compose --profile "$LAYOUT" --profile gateway)

# Targets: 1..4 / a|b (replica, must belong to LAYOUT), gw|litellm.
svc_of() {
  local s
  case $1 in gw|litellm) echo litellm; return;; [1-4]) s=vllm-$1;; a|b) s=vllm-$1;; vllm-*) s=$1;;
    *) echo "unknown target '$1' (${REPLICAS[*]#vllm-} gw)" >&2; exit 1;; esac
  [[ " ${REPLICAS[*]} " == *" $s "* ]] || { echo "$s is not part of LAYOUT=$LAYOUT" >&2; exit 1; }
  echo "$s"
}
port_of() { case $1 in vllm-1) echo "$R1_PORT";; vllm-2) echo "$R2_PORT";; vllm-3) echo "$R3_PORT";; vllm-4) echo "$R4_PORT";;
  vllm-a) echo "$REPLICA_A_PORT";; vllm-b) echo "$REPLICA_B_PORT";; esac; }

wait_healthy() { # wait_healthy <service> [timeout_s]
  local svc=$1 t=${2:-3600} c; c=$("${DC[@]}" ps -q "$svc")
  echo "waiting for $svc to become healthy (timeout ${t}s)..."
  for ((i = 0; i < t; i += 15)); do
    case $(docker inspect -f '{{.State.Health.Status}}' "$c" 2>/dev/null || echo gone) in
      healthy) echo "$svc healthy after ${i}s"; return 0 ;;
      unhealthy|gone) echo "$svc is unhealthy/gone"; docker logs --tail 80 "$c" 2>&1 | tail -40; return 1 ;;
    esac
    if [[ $(docker inspect -f '{{.State.Running}}' "$c") != true ]]; then echo "$svc exited"; docker logs --tail 80 "$c" 2>&1; return 1; fi
    sleep 15
  done
  echo "timeout waiting for $svc"; return 1
}
