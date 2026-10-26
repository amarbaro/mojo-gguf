#!/usr/bin/env bash
# U6 gate (CPU): `gguf-dequant MODEL TENSOR ROW0 N OUT.f32` bit-equal to the pinned llama.cpp to_float on every tensor
# type of the given files, driver tests/u6-check.py.
#   tests/u6-gate.sh FILE.gguf...    exit 0 only on PASS
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u6; rm -rf "$W"; mkdir -p "$W"
FILES=("$@"); [ ${#FILES[@]} -gt 0 ] || { echo "FAIL usage: tests/u6-gate.sh FILE.gguf... (scripts/oracle-gates.sh passes the oracle files)"; exit 2; }
check llama-cpp-pinned pinned
build() { "$MOJO" build src/gguf_dequant.mojo -I src -o "$W/gguf-dequant" > "$W/build.log" 2>&1; }
check build build
rows() { "$PY" -I tests/u6-check.py "$GGUFPY" "$LLAMACPP/build/bin/libggml-base.so" "$W/gguf-dequant" "$W" "${FILES[@]}" 2>&1 | tee "$W/check.log" | sed 's/^/     /'; grep -q '^PASS' "$W/check.log"; }
check rows-bit-equal rows
finish
