#!/usr/bin/env bash
# Legacy GGML fixtures (tests/mk-ggml.py) and their reference conversions by the pinned convert_llama_ggml_to_gguf.py.
# Needs only numpy and the pinned gguf-py (scripts/fetch-llamacpp.sh). Writes .work/oracle/ggml/.
#   tools/mkoracle-ggml.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}; LLAMACPP=${LLAMACPP:-.work/llama.cpp}
O=.work/oracle/ggml; mkdir -p "$O"
"$PY" tests/mk-ggml.py "$O" > "$O/mk.log" 2>&1 || { echo "FAIL mk-ggml: see $O/mk.log"; exit 1; }
for f in ggjt3-f16 ggmf1-f32 ggml-f32; do
  PYTHONPATH=$LLAMACPP/gguf-py "$PY" "$LLAMACPP/convert_llama_ggml_to_gguf.py" --input "$O/$f.bin" --output "$O/$f.gguf" > "$O/$f.log" 2>&1 || { echo "FAIL convert $f: see $O/$f.log"; exit 1; }
done
echo "PASS ggml fixtures -> $O"
