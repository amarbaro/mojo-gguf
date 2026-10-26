#!/usr/bin/env bash
# U2 gate: tables generated from the pinned gguf-py into Mojo, checked by a Mojo dump against a gguf-py dump.
#   tests/u2-gate.sh    exit 0 only on PASS 7/7
# Inputs: tools/gen-tables.py GGUFPY_DIR OUT_DIR (writes OUT_DIR/consts.mojo, OUT_DIR/tensor_map.mojo),
# the generated files committed as src/gguf/consts.mojo + src/gguf/tensor_map.mojo, and src/u2_dump.mojo whose stdout
# is exactly tests/u2-oracle.py's JSON (compact, sorted keys) built from the generated Mojo tables only.
source "$(dirname "$0")/gate-lib.sh"
W=.work/u2; rm -rf "$W"; mkdir -p "$W"
check llama-cpp-pinned pinned
gen() { "$PY" -I tools/gen-tables.py "$GGUFPY" "$1" > "$W/gen.log" 2>&1; }
check generator-runs gen "$W/a"
check generator-deterministic bash -c "$(declare -f gen); PY='$PY' GGUFPY='$GGUFPY' W='$W'; gen '$W/b' && diff -r '$W/a' '$W/b' > /dev/null"
check committed-equals-generated bash -c "cmp -s '$W/a/consts.mojo' src/gguf/consts.mojo && cmp -s '$W/a/tensor_map.mojo' src/gguf/tensor_map.mojo"
build() { "$MOJO" build src/u2_dump.mojo -I src -o "$W/u2-dump" > "$W/build.log" 2>&1; }
check build build
dump() { "$PY" -I tests/u2-oracle.py "$GGUFPY" > "$W/ref.json" && "$W/u2-dump" > "$W/got.json" && cmp -s "$W/ref.json" "$W/got.json"; }
check dump-equals-gguf-py dump
[ -f "$W/got.json" ] && ! cmp -s "$W/ref.json" "$W/got.json" && echo "     first diff: $(cmp "$W/ref.json" "$W/got.json" 2>&1 | head -1)"
finish
