#!/usr/bin/env bash
# U3a gate: safetensors reader + `hf2gguf HF_DIR --outfile OUT --outtype bf16|q8_0` for qwen2 (qwen2.5-0.5b-instruct),
# byte-identical to the pinned convert_hf_to_gguf.py output (tools/mkoracle.sh), plus wall + peak RSS per run.
#   tests/u3-gate.sh    exit 0 only on PASS 5/5
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u3; rm -rf "$W"; mkdir -p "$W"
check llama-cpp-pinned pinned
build() { "$MOJO" build src/hf2gguf.mojo -I src -o "$W/hf2gguf" > "$W/build.log" 2>&1; }
check build build
conv() { local t=$1 ref=$2
  /usr/bin/time -f "%e s %M KB" -o "$W/$t.time" "$W/hf2gguf" "$HF05" --outfile "$W/$t.gguf" --outtype "$t" > "$W/$t.log" 2>&1 || return 1
  echo "     $t: $(cat "$W/$t.time") (oracle $(cut -d' ' -f2- "$ref.log.time"))"
  "$PY" -I tools/gguf-cmp.py "$ref" "$W/$t.gguf" > "$W/$t.cmp" 2>&1; grep -q '^PASS identical-bytes' "$W/$t.cmp" || { sed 's/^/     /' "$W/$t.cmp" | head -8; return 1; }; }
check bf16-identical conv bf16 "$ORACLE/bf16.gguf"
check q8_0-identical conv q8_0 "$ORACLE/q8_0-py.gguf"
finish
