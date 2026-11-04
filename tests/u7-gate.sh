#!/usr/bin/env bash
# U7 gate: `lora2gguf ADAPTER_DIR --base HF_DIR --outfile OUT --outtype f32|bf16|q8_0` (convert_lora_to_gguf.py argv)
# byte-identical to the pinned script on two synthetic PEFT adapters (tests/mk-lora.py: a = f32 r8 attention,
# b = bf16 r4 mlp) at three out types, plus wall + peak RSS against the script's.
#   tests/u7-gate.sh    exit 0 only on PASS 9/9
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u7; rm -rf "$W"; mkdir -p "$W"; LO=$MAIN/.work/oracle/lora
check llama-cpp-pinned pinned
build() { "$MOJO" build src/lora2gguf.mojo -I src -I lib -o "$W/lora2gguf" > "$W/build.log" 2>&1; }
check build build
conv() { local a=$1 t=$2 ref=$LO/$1-$2.gguf
  /usr/bin/time -f "%e s %M KB" -o "$W/$a-$t.time" "$W/lora2gguf" "$LO/$a" --base "$HF05" --outfile "$W/$a-$t.gguf" --outtype "$t" > "$W/$a-$t.log" 2>&1 || return 1
  echo "     $a $t: $(cat "$W/$a-$t.time") (oracle $(cat "$ref.log.time"))"
  "$PY" -I tools/gguf-cmp.py "$ref" "$W/$a-$t.gguf" > "$W/$a-$t.cmp" 2>&1; grep -q '^PASS identical-bytes' "$W/$a-$t.cmp" || { sed 's/^/     /' "$W/$a-$t.cmp" | head -8; return 1; }; }
for a in a b; do for t in f32 bf16 q8_0; do check "$a-$t-identical" conv $a $t; done; done
finish
