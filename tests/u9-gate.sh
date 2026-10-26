#!/usr/bin/env bash
# U9 gate: `ggml2gguf --input IN --output OUT` (convert_llama_ggml_to_gguf.py argv) on four legacy fixtures
# (tests/mk-ggml.py): GGJTv3 F16, GGMFv1 F32, GGML F32 byte-identical to the pinned script; GGJTv3 Q8_0, where the
# pinned script exits 1 (it reads quantized row bytes as elements), must exit 1 too: parity, not a fix.
#   tests/u9-gate.sh    exit 0 only on PASS 7/7
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u9; rm -rf "$W"; mkdir -p "$W"; GO=$MAIN/.work/oracle/ggml
check llama-cpp-pinned pinned
build() { "$MOJO" build src/ggml2gguf.mojo -I src -o "$W/ggml2gguf" > "$W/build.log" 2>&1; }
check build build
conv() { local f=$1
  "$W/ggml2gguf" --input "$GO/$f.bin" --output "$W/$f.gguf" > "$W/$f.log" 2>&1 || return 1
  "$PY" -I tools/gguf-cmp.py "$GO/$f.gguf" "$W/$f.gguf" > "$W/$f.cmp" 2>&1; grep -q '^PASS identical-bytes' "$W/$f.cmp" || { sed 's/^/     /' "$W/$f.cmp" | head -8; return 1; }; }
refuse() { local rc=0; "$W/ggml2gguf" --input "$GO/ggjt3-q8_0.bin" --output "$W/q8.gguf" > "$W/q8.log" 2>&1 || rc=$?; [ "$rc" = 1 ]; }
check ggjt3-f16-identical conv ggjt3-f16
check ggmf1-f32-identical conv ggmf1-f32
check ggml-f32-identical conv ggml-f32
check ggjt3-q8_0-refused-like-upstream refuse
finish
