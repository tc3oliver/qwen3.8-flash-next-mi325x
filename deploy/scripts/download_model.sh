#!/usr/bin/env bash
# Download the pinned model revision into the shared read-only model dir. Resumable: re-run to continue.
# Usage: deploy/scripts/download_model.sh            (HF_TOKEN optional; repo is not gated)
set -euo pipefail
cd "$(dirname "$0")/.."; set -a; source .env; set +a
REPO=Qwen/Qwen3.8-Flash-Next-FP8
REVISION=236dfdf285828023ca3bcd3f37366c58a3469b13   # pinned commit, see README
DEST=${MODEL_DIR:?set MODEL_DIR in .env}
LOG=${LOG_DIR:-logs}/download.log

export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
mkdir -p "$DEST" "$(dirname "$LOG")"

need=$((173 * 1024 * 1024))   # KiB
have=$(df -k --output=avail "$DEST" | tail -1)
(( have > need + 20*1024*1024 )) || { echo "not enough space on $(df --output=target "$DEST" | tail -1)"; exit 1; }

echo "$(date -Is) start $REPO@$REVISION -> $DEST" | tee -a "$LOG"
hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" --max-workers 16 2>&1 | tee -a "$LOG"
echo "$REVISION" > "$DEST/.revision"
echo "$(date -Is) done" | tee -a "$LOG"
