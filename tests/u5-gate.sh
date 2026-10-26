#!/usr/bin/env bash
# U5 gate: gguf-py scripts rewritten as one Mojo CLI, `gguf-tool dump|hash|set-metadata|convert-endian ARGS`, with
# the same argv as gguf_dump.py / gguf_hash.py / gguf_set_metadata.py / gguf_convert_endian.py.
#   tests/u5-gate.sh    exit 0 only on PASS (cases + 3)
# Per case: stdout, stderr-free exit code and every file the command writes must equal the pinned Python script's.
# stdin carries $IN (convert-endian asks for YES). In-place commands run on copies under the same relative name in two directories, so printed paths agree.
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u5; rm -rf "$W"; mkdir -p "$W"
check llama-cpp-pinned pinned
build() { "$MOJO" build src/gguf_tool.mojo -I src -o "$W/gguf-tool" > "$W/build.log" 2>&1; }
check build build
script() { case $1 in dump) echo gguf_dump.py;; hash) echo gguf_hash.py;; set-metadata) echo gguf_set_metadata.py;; convert-endian) echo gguf_convert_endian.py;; esac; }
# case: NAME SRC_FILE CMD ARGS... ; "m.gguf" in ARGS is the working copy of SRC_FILE
run_case() {
  local name=$1 src=$2; shift 2; local cmd=$1; shift
  rm -rf "$W/$name"; mkdir -p "$W/$name/py" "$W/$name/mj"
  cp --reflink=auto "$src" "$W/$name/py/m.gguf"; cp --reflink=auto "$src" "$W/$name/mj/m.gguf"
  local rp=0 rm_=0
  (cd "$W/$name/py" && env PYTHONPATH="$GGUFPY" "$PY" "$GGUFPY/gguf/scripts/$(script "$cmd")" "$@" > out 2> err <<< "${IN:-}") || rp=$?
  (cd "$W/$name/mj" && "$W/gguf-tool" "$cmd" "$@" > out 2> err <<< "${IN:-}") || rm_=$?
  [ "$rp" = "$rm_" ] || { echo "     $name: exit py $rp vs mojo $rm_"; return 1; }
  cmp -s "$W/$name/py/out" "$W/$name/mj/out" || { echo "     $name: stdout differs: $(cmp "$W/$name/py/out" "$W/$name/mj/out" 2>&1 | head -1)"; return 1; }
  cmp -s "$W/$name/py/m.gguf" "$W/$name/mj/m.gguf" || { echo "     $name: file differs"; return 1; }
}
FLASH=${FLASH_GGUF:-/nonexistent}
check dump-q4k            run_case dump-q4k  "$ORACLE/Q4_K.gguf"  dump m.gguf
if [ -s "$FLASH" ]; then check dump-flash-notensor run_case dump-fl   "$FLASH"             dump --no-tensors m.gguf; else echo "note: FLASH_GGUF not found, the tokenizer-array dump is not run"; fi
check dump-json           run_case dump-json "$ORACLE/Q8_0.gguf"  dump --json m.gguf
check dump-markdown       run_case dump-md   "$ORACLE/Q4_K.gguf"  dump --markdown m.gguf
check dump-offsets        run_case dump-off  "$ORACLE/bf16.gguf"  dump --data-offset --data-alignment m.gguf
check hash-q4_0           run_case hash-q40  "$ORACLE/Q4_0.gguf"  hash m.gguf
check hash-nolayer        run_case hash-nl   "$ORACLE/Q2_K.gguf"  hash --no-layer m.gguf
check set-int             run_case set-int   "$ORACLE/Q4_K.gguf"  set-metadata m.gguf qwen2.context_length 4096 --force
check set-missing-key     run_case set-miss  "$ORACLE/Q4_K.gguf"  set-metadata m.gguf no.such.key 1 --force
IN=YES check endian-big-q8       run_case end-q8    "$ORACLE/Q8_0.gguf"  convert-endian m.gguf big
IN=YES check endian-big-q4k      run_case end-q4k   "$ORACLE/Q4_K.gguf"  convert-endian m.gguf big
finish
