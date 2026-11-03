#!/usr/bin/env bash
# One arm of the TP2 end-to-end A/B on this MI300X host.
#   arm.sh <stock|overlay> <A|B> <round> [smoke]
# Pair A = HIP 6,7 (renderD176+184, slot a, port 8011); Pair B = HIP 2,3 (renderD144+152, slot b, port 8012). HIP 4 is production: never used.
# smoke: boot, verify the file the engine imports, 1 tiny request, remove container. No benchmark.
# otherwise: boot, verify, 3 x perfsuite (prefill 8x32K, decode C64, agent 32K C16), remove container.
set -uo pipefail
arm=$1 pair=$2 round=$3 mode=${4:-full}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); cd "$REPO/tools"
case $pair in A) slot=a HIP=6,7 PORT=8011;; B) slot=b HIP=2,3 PORT=8012;; *) echo "pair A|B"; exit 1;; esac
P=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd
case $arm in
  stock)   Q=7a7c8501e0b92a4236e382c6ef494670d684ee1dfeb2d673f942a4f38fccf69b I=ca8ede9f27c933b85a3a3e45e8e1b4f7693c8f0271ff3cab3fa6fb64de8c2476; PATCH=();;
  overlay) Q=f59670bf046ad58b051b74d80ba23883477b845824278704eff6ee67116ac524 I=dbb72baa9b578553ecb58e0cb334e6f9f96d172f216d0ed4267e0ce536caa46a; PATCH=(PATCH_QSA=1);;
  *) echo "arm stock|overlay"; exit 1;; esac
name=$arm-r$round-$pair; [[ $mode == smoke ]] && name=smoke-$name
C=mi300-e2e-$slot; B=http://127.0.0.1:$PORT; OUT=$HERE; export EXP_PREFIX=mi300-e2e
trap 'docker logs $C > "$OUT/$name.server.log" 2>&1; docker rm -f $C >/dev/null 2>&1' EXIT
experiments/exp.sh $slot "$name" HIP=$HIP "${PATCH[@]}" \
  VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT6 VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  AITER_CONFIG_FMOE=/usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv:/tuned/tuned_fmoe_qwen38_ep2.csv || exit 1
# the file python really imports, not just the mount path (d02 ran stock because of a wrong path)
docker exec -i $C python3 - <<PY | tee "$OUT/$name.hash.txt"
import hashlib, vllm.models.qwen4_exp.amd.ops.qsa as q, vllm.models.qwen4_exp.amd.indexer_qsa as i
for m, want in ((q, "$Q"), (i, "$I")):
    h = hashlib.sha256(open(m.__file__, "rb").read()).hexdigest()
    print("$arm", m.__file__, h, "OK" if h == want else "MISMATCH")
PY
grep -q MISMATCH "$OUT/$name.hash.txt" && { echo "HASH MISMATCH: abort"; exit 2; }
[[ $(grep -c OK "$OUT/$name.hash.txt") == 2 ]] || { echo "hash check did not run"; exit 2; }
docker exec $C python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i); print(i,'%04x:%02x:%02x.0'%(p.pci_domain_id,p.pci_bus_id,p.pci_device_id))" > "$OUT/$name.gpu.txt" 2>/dev/null
cat "$OUT/$name.gpu.txt"
case $pair in A) WANT="0 0000:e6:00.0 1 0000:96:00.0";; B) WANT="0 0000:66:00.0 1 0000:16:00.0";; esac
[[ $(tr '\n' ' ' < "$OUT/$name.gpu.txt" | xargs) == "$WANT" ]] || { echo "BDF MISMATCH: abort"; exit 3; }
if [[ $mode == smoke ]]; then
  curl -fsS $B/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' | tee "$OUT/$name.req.json" | head -c 600; echo
  exit 0
fi
noise() { { date +%T; uptime; rocm-smi --showuse | grep -E 'GPU\[[0-9]\].*(use|Use)'; } >> "$OUT/$name.noise.txt" 2>&1; }
for p in 1 2 3; do
  echo "== before p$p" >> "$OUT/$name.noise.txt"; noise
  uv run -q --with httpx experiments/perfsuite.py --base $B --label "$name.p$p" --only prefill_8,decode_c64,agent_32k_c16 --out "$OUT/$name.p$p.perf.json"
  echo "== after p$p" >> "$OUT/$name.noise.txt"; noise
done
