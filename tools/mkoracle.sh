#!/usr/bin/env bash
# Build the reference GGUF set for one HF model: converter bf16 + q8_0, then llama-quantize --pure per type.
#   mkoracle.sh HF_DIR NAME [TYPES...]     writes .work/oracle/NAME/, one oracle_time line per step
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 2 ] || { sed -n 2,3p "$0"; exit 2; }
H=$1; O=.work/oracle/$2; shift 2
types=("$@"); [ ${#types[@]} -gt 0 ] || types=(Q8_0 Q4_0 Q4_K Q6_K Q2_K Q2_0 TQ1_0 TQ2_0)
mkdir -p "$O"
fails=()
step() { local name=$1; shift; local r; if r=$(tools/oracle.sh "$@" 2>&1); then echo "$name: $r"; else echo "$r"; fails+=("$name"); fi; }
step convert-bf16 convert "$H" "$O/bf16.gguf" bf16
step convert-q8_0 convert "$H" "$O/q8_0-py.gguf" q8_0
[ -s "$O/bf16.gguf" ] || { echo "FAIL convert-bf16: no $O/bf16.gguf"; exit 1; }
for t in "${types[@]}"; do step "quantize-$t" quantize "$O/bf16.gguf" "$O/$t.gguf" "$t"; done
FLASH=${FLASH_GGUF:-/nonexistent}  # optional: a GGUF with big tokenizer arrays, for the edit oracle
edit() { local src=$1 out=$2; if [ -s "$src" ]; then tools/oracle.sh new_metadata "$src" "$out" --general-name mojo-gguf-edit --remove-metadata tokenizer.chat_template > "$out.log" 2>&1 && echo "edit $(basename "$out"): ok" || { echo "FAIL edit $out: see $out.log"; fails+=("edit-$(basename "$out")"); }; else echo "FAIL edit: $src missing"; fails+=("edit-$(basename "$out")"); fi; }
edit "$O/Q4_K.gguf" "$O/edit-Q4_K.gguf"
n=$((3 + ${#types[@]}))
if [ -s "$FLASH" ]; then edit "$FLASH" "$O/edit-flash.gguf"; n=$((n + 1)); else echo "note: FLASH_GGUF not found, no edit-flash oracle"; fi
[ ${#fails[@]} -eq 0 ] || { echo "FAIL ${#fails[@]}/$n ${fails[*]}"; exit 1; }
echo "PASS $n/$n -> $O"
