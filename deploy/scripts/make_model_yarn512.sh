#!/usr/bin/env bash
# Build <dest>/: config.json = original + YaRN (factor 2 = 512K by default); every other file a symlink to $LINK_ROOT/<file>
# (deploy/compose.yaml mounts the original model dir read-only at /models-orig). Needs python3 on the host.
# usage: make_model_yarn512.sh <original model dir> <dest>     env: FACTOR=2, LINK_ROOT=/models-orig
set -euo pipefail
SRC=${1:?original model dir}; DST=${2:?destination}; FACTOR=${FACTOR:-2}; LINK_ROOT=${LINK_ROOT:-/models-orig}
mkdir -p "$DST"
for f in "$SRC"/*; do b=$(basename "$f"); [ "$b" = config.json ] && continue; ln -sfn "$LINK_ROOT/$b" "$DST/$b"; done
python3 - "$SRC/config.json" "$DST/config.json" "$FACTOR" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); t = c.get("text_config", c); rp = t["rope_parameters"]
assert rp.get("rope_type") == "default", rp
rp.update(rope_type="yarn", factor=float(sys.argv[3]), original_max_position_embeddings=262144)
json.dump(c, open(sys.argv[2], "w"), indent=2)
print("rope_parameters:", rp)
PY
echo "files: $(ls "$DST" | wc -l) (source: $(ls "$SRC" | wc -l))"
