#!/usr/bin/env bash
# Health, models, prefix-cache counters and GPU memory for the stack.
source "$(dirname "$0")/_lib.sh"
"${DC[@]}" ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}\t{{.Ports}}'
for svc in "${REPLICAS[@]}"; do
  p=$(port_of "$svc")
  h=$(curl -fsS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p/health" 2>/dev/null || echo down)
  m=$(curl -fsS "http://127.0.0.1:$p/metrics" 2>/dev/null | awk '
    /^vllm:prefix_cache_queries_total/ {q+=$NF} /^vllm:prefix_cache_hits_total/ {h+=$NF}
    /^vllm:num_requests_running/ {r+=$NF} /^vllm:num_requests_waiting/ {w+=$NF}
    /^vllm:kv_cache_usage_perc/ {k=$NF}
    END { if (q>0) printf "prefix_hit=%.1f%% (%d/%d tok) running=%d waiting=%d kv=%.1f%%", 100*h/q, h, q, r, w, 100*k }')
  echo "$svc :$p health=$h ${m:-}"
done
if curl -fsS -o /dev/null "http://127.0.0.1:$LITELLM_PORT/health/liveliness" 2>/dev/null; then
  echo "litellm :$LITELLM_PORT live; models: $(curl -fsS -H "Authorization: Bearer $LITELLM_MASTER_KEY" "http://127.0.0.1:$LITELLM_PORT/v1/models" | jq -r '[.data[].id]|join(",")')"
else echo "litellm :$LITELLM_PORT down"; fi
echo "-- GPU (amd-smi GPU0-3 = HIP 1,3,2,0), layout $LAYOUT"
amd-smi metric -g 0 1 2 3 -m -u --json 2>/dev/null | jq -r '.gpu_data[] |
  "GPU\(.gpu) util=\(.usage.gfx_activity.value)% vram=\((.mem_usage.used_vram.value/1024*10|round)/10)/\((.mem_usage.total_vram.value/1024)|round) GiB"'
