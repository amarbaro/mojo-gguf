#!/usr/bin/env bash
# U3a gate: safetensors reader + `hf2gguf HF_DIR --outfile OUT --outtype bf16|q8_0` for qwen2 (qwen2.5-0.5b-instruct),
# byte-identical to the pinned convert_hf_to_gguf.py output (tools/mkoracle.sh), plus wall + peak RSS per run.
#   tests/u3-gate.sh    exit 0 only on PASS n/n
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
# a negative token id must be refused (it used to index tokens[-1]: an abort), in the vocab and in added_tokens
negid() { local name=$1 from=$2 to=$3 want=$4 d=$W/neg-$1; rm -rf "$d"; mkdir -p "$d/hf"
  for f in "$HF05"/*; do ln -s "$f" "$d/hf/$(basename "$f")"; done; rm "$d/hf/tokenizer.json"
  "$PY" -I -c 'import sys; b = open(sys.argv[1], "rb").read(); assert sys.argv[2].encode() in b; open(sys.argv[4], "wb").write(b.replace(sys.argv[2].encode(), sys.argv[3].encode(), 1))' "$HF05/tokenizer.json" "$from" "$to" "$d/hf/tokenizer.json" || return 1
  local rc=0; "$W/hf2gguf" "$d/hf" --outfile "$d/o.gguf" --outtype bf16 > "$d/log" 2>&1 || rc=$?
  [ "$rc" = 1 ] && grep -q "$want" "$d/log" || { echo "     $name: exit $rc, $(tail -1 "$d/log" | cut -c1-100)"; return 1; }; }
check negative-vocab-id    negid vocab  '"!": 0,' '"!": -1,' 'token id -1'
check negative-added-id    negid added  '"id": 151643,' '"id": -1,' 'added token id -1'
finish
