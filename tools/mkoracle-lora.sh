#!/usr/bin/env bash
# Synthetic PEFT adapters (tests/mk-lora.py) and their reference conversions by the pinned convert_lora_to_gguf.py,
# with wall time and peak RSS. CPU only. Needs ORACLE_PY (python with torch, transformers, safetensors, numpy).
#   tools/mkoracle-lora.sh HF_BASE_DIR      HF_BASE_DIR = Qwen2.5-0.5B-Instruct; writes .work/oracle/lora/
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 1 ] || { sed -n 5,5p "$0"; exit 2; }
PY=${ORACLE_PY:-python3}; LLAMACPP=${LLAMACPP:-.work/llama.cpp}
O=.work/oracle/lora; mkdir -p "$O"
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
"$PY" tests/mk-lora.py "$1" "$O" > "$O/mk.log" 2>&1 || { echo "FAIL mk-lora: see $O/mk.log"; exit 1; }
for a in a b; do for t in f32 bf16 q8_0; do
  /usr/bin/time -f "%e s %M KB" -o "$O/$a-$t.gguf.log.time" env PYTHONPATH="$LLAMACPP/gguf-py" "$PY" "$LLAMACPP/convert_lora_to_gguf.py" "$O/$a" --base "$1" --outfile "$O/$a-$t.gguf" --outtype "$t" > "$O/$a-$t.gguf.log" 2>&1 || { echo "FAIL convert $a-$t: see $O/$a-$t.gguf.log"; exit 1; }
done; done
echo "PASS lora fixtures -> $O"
