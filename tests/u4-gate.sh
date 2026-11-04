#!/usr/bin/env bash
# U4 gate: `quantize --pure IN OUT TYPE` (llama-quantize's argv) from the oracle bf16.gguf, byte-identical to the pinned
# llama-quantize --pure output for each type, plus wall + peak RSS per type against the oracle's.
#   tests/u4-gate.sh [TYPE...]    exit 0 only on PASS (3 + types); default: all 8 oracle types
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u4; rm -rf "$W"; mkdir -p "$W"
TYPES=("$@"); [ ${#TYPES[@]} -gt 0 ] || TYPES=(Q8_0 Q4_0 Q4_K Q6_K Q2_K Q2_0 TQ1_0 TQ2_0)
check llama-cpp-pinned pinned
build() { "$MOJO" build src/quantize.mojo -I src -I lib -o "$W/quantize" > "$W/build.log" 2>&1; }
check build build
quant() { local t=$1 ref=$ORACLE/$1.gguf
  /usr/bin/time -f "%e s %M KB" -o "$W/$t.time" "$W/quantize" --pure "$ORACLE/bf16.gguf" "$W/$t.gguf" "$t" > "$W/$t.log" 2>&1 || return 1
  echo "     $t: $(cat "$W/$t.time") (oracle $(cut -d' ' -f2- "$ref.log.time"))"
  "$PY" -I tools/gguf-cmp.py "$ref" "$W/$t.gguf" > "$W/$t.cmp" 2>&1; grep -q '^PASS identical-bytes' "$W/$t.cmp" || { sed 's/^/     /' "$W/$t.cmp" | head -8; return 1; }; }
for t in "${TYPES[@]}"; do check "$t-identical" quant "$t"; done
finish
