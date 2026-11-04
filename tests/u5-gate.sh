#!/usr/bin/env bash
# U5 gate: gguf-py scripts rewritten as one Mojo CLI, `gguf-tool dump|hash|set-metadata|convert-endian ARGS`, with
# the same argv as gguf_dump.py / gguf_hash.py / gguf_set_metadata.py / gguf_convert_endian.py.
#   tests/u5-gate.sh    exit 0 only on PASS (cases + 3)
# Per case (run_case in gate-lib.sh): the exit code both tools must return (EXP, default 0), a stderr message when it is not 0,
# stdout and every file the command writes must equal the pinned Python script's.
# stdin carries $IN (convert-endian asks for YES). In-place commands run on copies under the same relative name in two directories, so printed paths agree.
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u5; rm -rf "$W"; mkdir -p "$W"
check llama-cpp-pinned pinned
build() { "$MOJO" build src/gguf_tool.mojo -I src -I lib -o "$W/gguf-tool" > "$W/build.log" 2>&1; }
check build build
FLASH=${FLASH_GGUF:-/nonexistent}
check dump-q4k            run_case dump-q4k  "$ORACLE/Q4_K.gguf"  dump m.gguf
if [ -s "$FLASH" ]; then check dump-flash-notensor run_case dump-fl   "$FLASH"             dump --no-tensors m.gguf; else echo "note: FLASH_GGUF not found, the tokenizer-array dump is not run"; fi
check dump-json           run_case dump-json "$ORACLE/Q8_0.gguf"  dump --json m.gguf
check dump-markdown       run_case dump-md   "$ORACLE/Q4_K.gguf"  dump --markdown m.gguf
check dump-offsets        run_case dump-off  "$ORACLE/bf16.gguf"  dump --data-offset --data-alignment m.gguf
check hash-q4_0           run_case hash-q40  "$ORACLE/Q4_0.gguf"  hash m.gguf
check hash-nolayer        run_case hash-nl   "$ORACLE/Q2_K.gguf"  hash --no-layer m.gguf
check set-int             run_case set-int   "$ORACLE/Q4_K.gguf"  set-metadata m.gguf qwen2.context_length 4096 --force
EXP=1 check set-missing-key run_case set-miss  "$ORACLE/Q4_K.gguf"  set-metadata m.gguf no.such.key 1 --force
IN=YES check endian-big-q8       run_case end-q8    "$ORACLE/Q8_0.gguf"  convert-endian m.gguf big
IN=YES check endian-big-q4k      run_case end-q4k   "$ORACLE/Q4_K.gguf"  convert-endian m.gguf big
# corrupt input: a truncated model (cut inside the metadata) is an error exit with a message on every command, from both tools
head -c 400 "$ORACLE/Q4_K.gguf" > "$W/trunc.gguf"
EXP=1 check dump-truncated  run_case dump-tr "$W/trunc.gguf" dump m.gguf
EXP=1 check hash-truncated  run_case hash-tr "$W/trunc.gguf" hash m.gguf
EXP=1 check set-truncated   run_case set-tr  "$W/trunc.gguf" set-metadata m.gguf qwen2.context_length 4096 --force
IN=YES EXP=1 check endian-truncated run_case end-tr "$W/trunc.gguf" convert-endian m.gguf big
check mutant-both-fail-rejected rejects run_case mut-tr "$W/trunc.gguf" dump m.gguf
finish
