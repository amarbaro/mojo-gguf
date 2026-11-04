#!/usr/bin/env bash
# U1 gate: src/gguf_rewrite.mojo parses a GGUF and serializes it again.
#   tests/u1-gate.sh            exit 0 only on PASS n/n (n = 15 with FLASH_GGUF, 12 without)
# Fixtures: .work/oracle/qwen25-05b (tools/mkoracle.sh); FLASH_GGUF = optional, a GGUF with large tokenizer arrays
# (the author used one shard of a 3-file model; without it the two array cases are not run).
# Checks: build; 11 round trips identical-bytes; first stdout line `kv K tensors T alignment A data_offset D`
# equal to gguf-py's values on 2 files (a byte copy cannot print it); 2 edits (--name, --remove) identical-bytes
# to gguf_new_metadata.py output; bf16 wall + RSS line.
source "$(dirname "$0")/gate-lib.sh"
FLASH=${FLASH_GGUF:-/nonexistent}
W=.work/u1; mkdir -p "$W"

build() { "$MOJO" build src/gguf_rewrite.mojo -I src -I lib -o "$W/gguf-rewrite" > "$W/build.log" 2>&1; }
if ! build; then echo "FAIL build: see $W/build.log"; tail -5 "$W/build.log"; echo "FAIL 1/1 build"; exit 1; fi

rewrite() {
  local src=$1 tag=$2 ref=${3:-$1}; shift 3 || shift $#
  rm -f "$W/out.gguf"
  /usr/bin/time -f "%e s %M KB" -o "$W/$tag.time" "$W/gguf-rewrite" "$src" "$W/out.gguf" "$@" > "$W/$tag.log" 2>&1 || { echo "  rewrite exited non-zero: $W/$tag.log"; return 1; }
  "$PY" -I "$MAIN/tools/gguf-cmp.py" "$ref" "$W/out.gguf" > "$W/$tag.cmp" 2>&1 || true
  grep -q "^PASS identical-bytes" "$W/$tag.cmp" || { head -3 "$W/$tag.cmp" | sed 's/^/  /'; return 1; }
}
summary() {
  local src=$1 tag=$2 want got
  want=$("$PY" -I -c "import sys; sys.path.insert(0, sys.argv[2]); from gguf import GGUFReader; r = GGUFReader(sys.argv[1]); print(f'kv {len(r.fields) - 3} tensors {len(r.tensors)} alignment {r.alignment} data_offset {r.data_offset}')" "$src" "$GGUFPY")
  got=$(head -1 "$W/$tag.log")
  [ "$got" = "$want" ] || { echo "  want '$want' got '$got'"; return 1; }
}
for t in bf16 q8_0-py Q8_0 Q4_0 Q4_K Q6_K Q2_K Q2_0 TQ1_0 TQ2_0; do
  [ -s "$ORACLE/$t.gguf" ] || { echo "FAIL fixture: $ORACLE/$t.gguf missing (tools/mkoracle.sh)"; exit 1; }
  check "rewrite-$t" rewrite "$ORACLE/$t.gguf" "$t"
done
have_flash() { [ -s "$FLASH" ]; }
have_flash || echo "note: FLASH_GGUF not found, the tokenizer-array cases are not run"
if have_flash; then check rewrite-flash-shard1 rewrite "$FLASH" flash; fi
check summary-Q4_K summary "$ORACLE/Q4_K.gguf" Q4_K
if have_flash; then check summary-flash summary "$FLASH" flash; fi
for e in Q4_K flash; do
  [ $e = flash ] && ! have_flash && continue
  [ -s "$ORACLE/edit-$e.gguf" ] || { echo "FAIL fixture: $ORACLE/edit-$e.gguf missing (tools/mkoracle.sh)"; exit 1; }
  src=$ORACLE/Q4_K.gguf; [ $e = flash ] && src=$FLASH
  check "edit-$e" rewrite "$src" "edit-$e" "$ORACLE/edit-$e.gguf" --name mojo-gguf-edit --remove tokenizer.chat_template
done
rm -f "$W/out.gguf"
echo "bf16 rewrite: $(cat "$W/bf16.time" 2>/dev/null || echo none) (994 MB file)"
[ ${#fails[@]} -eq 0 ] || { echo "FAIL ${#fails[@]}/$n ${fails[*]}"; exit 1; }
echo "PASS $n/$n"
