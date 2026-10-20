#!/usr/bin/env bash
# Verify every downloaded file against the Hub manifest of the pinned revision.
# LFS files: sha256 == lfs.oid; small git files: git blob sha1 == oid. Also checks sizes and file count.
set -euo pipefail
cd "$(dirname "$0")/.."; set -a; source .env; set +a
REPO=Qwen/Qwen3.8-Flash-Next-FP8
REVISION=236dfdf285828023ca3bcd3f37366c58a3469b13
DEST=${MODEL_DIR:?set MODEL_DIR in .env}
OUT=$PWD/config/model-manifest.json
JOBS=${JOBS:-32}
mkdir -p "${LOG_DIR:-logs}"; VLOG=${LOG_DIR:-logs}/verify-$(date +%Y%m%d-%H%M%S).log   # never write into the model dir

curl -fsSL "https://huggingface.co/api/models/$REPO/tree/$REVISION?recursive=true" \
  | jq '[.[] | select(.type=="file") | {path, size, sha256: .lfs.oid, blob: .oid}]' > "$OUT"
echo "manifest: $(jq length "$OUT") files -> $OUT"

check() { # path size sha256 blob
  local f="$DEST/$1" got
  [[ -f $f ]] || { echo "MISSING $1"; return 1; }
  [[ $(stat -c %s "$f") == "$2" ]] || { echo "SIZE    $1"; return 1; }
  if [[ $3 != null ]]; then got=$(sha256sum "$f" | cut -d' ' -f1); [[ $got == "$3" ]] || { echo "SHA256  $1"; return 1; }
  else got=$(git hash-object "$f"); [[ $got == "$4" ]] || { echo "SHA1    $1"; return 1; }; fi
  echo "OK      $1"
}
export -f check; export DEST

jq -r '.[] | [.path, .size, (.sha256 // "null"), .blob] | @tsv' "$OUT" \
  | xargs -P "$JOBS" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r p s h b <<< "{}"; check "$p" "$s" "$h" "$b"' \
  | sort > "$VLOG" || true

ok=$(grep -c '^OK' "$VLOG" || true); total=$(jq length "$OUT")
grep -v '^OK' "$VLOG" || true
echo "verified $ok / $total files"
[[ $ok == "$total" ]] && echo "RESULT: PASS ($REVISION)" || { echo "RESULT: FAIL"; exit 1; }
