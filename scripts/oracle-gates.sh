#!/usr/bin/env bash
# The gates scripts/ci.sh cannot run on a GitHub runner: they need Qwen2.5-0.5B-Instruct, a llama.cpp build at the pin
# (llama-quantize, libggml-base.so) and a Python with torch. Run them locally, CPU only, from the repo root:
#   LLAMACPP=/path/to/llama.cpp ORACLE_PY=/path/to/python QWEN_HF=/path/to/Qwen2.5-0.5B-Instruct scripts/oracle-gates.sh
#     LLAMACPP    llama.cpp checkout at 1744c6bde with `cmake --build build --target llama-quantize ggml-base`
#     ORACLE_PY   python with torch, transformers, tokenizers, safetensors, numpy, pyyaml, tqdm, requests
#     QWEN_HF     the Hugging Face directory of Qwen/Qwen2.5-0.5B-Instruct
#   optional: FLASH_GGUF (a GGUF with big tokenizer arrays: two more cases in u1 and u5)
# Gates: u1 u3 u4 u5 u6 u7 u8, the converter, quantizer and dequant gates against llama.cpp's own outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LLAMACPP:?FAIL setup: set LLAMACPP to a llama.cpp checkout at the pin}"
: "${ORACLE_PY:?FAIL setup: set ORACLE_PY to a python with torch and transformers}"
: "${QWEN_HF:?FAIL setup: set QWEN_HF to the Qwen2.5-0.5B-Instruct directory}"
export LLAMACPP ORACLE_PY QWEN_HF
scripts/deps.sh
O=.work/oracle/qwen25-05b
[ -s "$O/bf16.gguf" ] && [ -s "$O/Q4_K.gguf" ] || tools/mkoracle.sh "$QWEN_HF" qwen25-05b
[ -s .work/oracle/lora/b-q8_0.gguf ] || tools/mkoracle-lora.sh "$QWEN_HF"
tools/mkoracle-ggml.sh > /dev/null
mkdir -p .work/oracle/u8 .work/t
"$ORACLE_PY" tests/u8-oracle.py "$LLAMACPP" .work/oracle/u8 "$QWEN_HF" > .work/oracle/u8/qwen-summary.txt 2> .work/oracle/u8/oracle.err
export U8_SUMMARY=$PWD/.work/oracle/u8/qwen-summary.txt
gate() { local name=$1; shift
  "$@" > ".work/t/$name.log" 2>&1 || { echo "FAIL $name: $*"; tail -20 ".work/t/$name.log"; exit 1; }
  [[ $(tail -1 ".work/t/$name.log") == PASS* ]] || { echo "FAIL $name: no PASS line at the end of .work/t/$name.log"; tail -5 ".work/t/$name.log"; exit 1; }
  echo "ok   $name: $(tail -1 ".work/t/$name.log")"; }
gate u1 tests/u1-gate.sh
gate u3 tests/u3-gate.sh
gate u4 tests/u4-gate.sh
gate u5 tests/u5-gate.sh
gate u6 tests/u6-gate.sh "$O"/{Q8_0,Q4_0,Q4_K,Q6_K,Q2_K,Q2_0,TQ1_0}.gguf  # TQ2_0 has no dequantizer
gate u6-all-types "$ORACLE_PY" tools/u6-extra-types.py "$LLAMACPP/gguf-py" "$LLAMACPP/build/bin/libggml-base.so" .work/u6/gguf-dequant .work/u6/extra
gate u7 tests/u7-gate.sh
gate u7-extra tools/lora-extra-check.sh
gate u8 tests/u8-gate.sh
echo "PASS oracle-gates"
