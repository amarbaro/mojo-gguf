#!/usr/bin/env bash
# lora2gguf vs convert_lora_to_gguf.py on an adapter that exercises what tests/u7-gate.sh does not: embedding
# transpose, norm pass-through, F16 out type, mixed dtypes, aLoRA tokens, name heuristics, dir/template outfile.
#   tools/lora-extra-check.sh      exit 0 only on PASS n/n (needs .work/u7/lora2gguf from tests/u7-gate.sh)
source "$(dirname "$0")/../tests/gate-lib.sh"
W=$PWD/.work/u7/extra; X=$W/Qwen-Foo-lora-v2; rm -rf "$W"; mkdir -p "$W"
ORPY=${ORACLE_PY:-python3}
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
"$ORPY" tools/mk-lora-extra.py "$X" > "$W/gen.log" 2>&1 || { echo "FAIL generate: $W/gen.log"; exit 1; }
ref() { PYTHONPATH=$GGUFPY "$ORPY" "$LLAMACPP/convert_lora_to_gguf.py" "$X" --base "$HF05" --outfile "$1" --outtype "$2" > "$1.log" 2>&1; }
same() { "$PY" -I tools/gguf-cmp.py "$1" "$2" 2>&1 | grep -q '^PASS identical-bytes'; }
one() { local t=$1
  ref "$W/ref-$t.gguf" "$t" && .work/u7/lora2gguf "$X" --base "$HF05" --outfile "$W/my-$t.gguf" --outtype "$t" > /dev/null && same "$W/ref-$t.gguf" "$W/my-$t.gguf"; }
for t in f32 f16 bf16 q8_0; do check "extra-$t-identical" one $t; done
dirname_out() { mkdir -p "$W/d-ref" "$W/d-my"; ref "$W/d-ref" q8_0 && .work/u7/lora2gguf "$X" --base "$HF05" --outfile "$W/d-my" --outtype q8_0 > /dev/null \
  && [ "$(ls "$W/d-ref")" = "$(ls "$W/d-my")" ] && same "$W/d-ref/$(ls "$W/d-ref")" "$W/d-my/$(ls "$W/d-my")"; }
check dir-outfile-name dirname_out
templ() { ref "$W/t-{ftype}-{OUTTYPE}.gguf" bf16 && .work/u7/lora2gguf "$X" --base "$HF05" --outfile "$W/u-{ftype}-{OUTTYPE}.gguf" --outtype bf16 > /dev/null \
  && same "$W/t-bf16-BF16.gguf" "$W/u-bf16-BF16.gguf"; }
check templated-outfile templ
finish
