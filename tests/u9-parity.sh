#!/usr/bin/env bash
# U9 parity beyond tests/u9-gate.sh: other containers and versions, F16 and F32 weights, byte / space / empty
# tokens, a 300-token vocab, and the argv flags. Each case runs convert_llama_ggml_to_gguf.py and ggml2gguf on the
# same argv; both must exit alike and, on success, write the same bytes. Scratch: .work/u9-parity.
#   tests/u9-parity.sh    exit 0 only on PASS n/n
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u9-parity; rm -rf "$W"; mkdir -p "$W"; GO=$MAIN/.work/oracle/ggml
check llama-cpp-pinned pinned
check build "$MOJO" build src/ggml2gguf.mojo -I src -o "$W/ggml2gguf"
check fixtures "$PY" tests/mk-ggml-variants.py "$W/fix"

# parity INPUT EXPECT ARGS...: EXPECT is ok (both exit 0, identical bytes) or refuse (both exit 1 with a FAIL line)
parity() {
  local in=$1 expect=$2; shift 2
  local rc1=0 rc2=0
  [ -f "$in" ] || { echo "     missing input $in"; return 1; }
  rm -f "$W/ref.gguf" "$W/got.gguf"
  "$PY" "$LLAMACPP/convert_llama_ggml_to_gguf.py" --input "$in" --output "$W/ref.gguf" "$@" > "$W/ref.log" 2>&1 || rc1=$?
  "$W/ggml2gguf" --input "$in" --output "$W/got.gguf" "$@" > "$W/got.log" 2>&1 || rc2=$?
  if [ "$expect" = refuse ]; then
    [ "$rc1" != 0 ] && [ "$rc2" = "$rc1" ] && grep -q "^FAIL ggml2gguf" "$W/got.log" && return 0
    echo "     expected refusal: script $rc1, ggml2gguf $rc2"; return 1
  fi
  if [ "$rc1" != 0 ] || [ "$rc2" != 0 ]; then echo "     expected success: script $rc1, ggml2gguf $rc2"; return 1; fi
  cmp -s "$W/ref.gguf" "$W/got.gguf" || { echo "     bytes differ: $(cmp "$W/ref.gguf" "$W/got.gguf" 2>&1 | head -1)"; return 1; }
}
# notsup ARGS...: the metadata and vocab directory options are not ported: exit 2 and the reason
notsup() { local rc=0; "$W/ggml2gguf" --input "$GO/ggjt3-f16.bin" --output "$W/x.gguf" "$@" > "$W/m.log" 2>&1 || rc=$?; [ "$rc" = 2 ] && grep -q "not supported" "$W/m.log"; }

for f in ggjt3-f16 ggmf1-f32 ggml-f32; do check "$f default" parity "$GO/$f.bin" ok; done
for f in ggjt2-f16 ggjt1-f32 ggml-f16 ggmf1-f16 ggjt3-f16-vocab300; do check "$f default" parity "$W/fix/$f.bin" ok; done
check "ggjt2-q8_0 refused" parity "$W/fix/ggjt2-q8_0.bin" refuse
check "ggjt3-f16 name+desc" parity "$GO/ggjt3-f16.bin" ok --name custom --desc "two words"
check "ggjt3-f16 empty desc" parity "$GO/ggjt3-f16.bin" ok --desc ""
check "ggjt3-f16 empty name" parity "$GO/ggjt3-f16.bin" ok --name ""
check "ggjt3-f16 gqa 2" parity "$GO/ggjt3-f16.bin" ok --gqa 2
check "ggjt3-f16 gqa 4" parity "$GO/ggjt3-f16.bin" ok --gqa 4
check "ggjt3-f16 gqa 3 refused" parity "$GO/ggjt3-f16.bin" refuse --gqa 3
check "ggml-f16 eps 1e-6 ctx 4096" parity "$W/fix/ggml-f16.bin" ok --eps 1e-6 --context-length 4096
check "ggmf1-f16 eps=1e-5 -c 512" parity "$W/fix/ggmf1-f16.bin" ok --eps=1e-5 -c 512
check "ggjt3-f16-vocab300 verbose" parity "$W/fix/ggjt3-f16-vocab300.bin" ok --verbose --vocabtype spm,hfft
check "already GGUF refused" parity "$GO/ggjt3-f16.gguf" refuse
check "metadata dir not supported" notsup -m "$W/fix"
check "vocab dir not supported" notsup --vocab-dir "$W/fix"
finish
