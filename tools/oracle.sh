#!/usr/bin/env bash
# Reference outputs from the pinned llama.cpp checkout, with wall time and peak RSS. CPU only.
#   oracle.sh convert HF_DIR OUT.gguf OUTTYPE     convert_hf_to_gguf.py (f32|f16|bf16|q8_0|tq1_0|tq2_0)
#   oracle.sh quantize IN.gguf OUT.gguf TYPE      llama-quantize --pure (one type for every tensor)
#   oracle.sh new_metadata IN.gguf OUT.gguf ARGS...  gguf_new_metadata.py --force (edit oracles)
#   oracle.sh dump IN.gguf                        gguf_dump.py (diagnosis only)
# env: LLAMACPP (default .work/llama.cpp from scripts/fetch-llamacpp.sh, pinned 1744c6bde), ORACLE_PY (python with
# torch, transformers, safetensors, numpy; only `convert` needs torch)
set -euo pipefail
LLAMACPP=${LLAMACPP:-$(cd "$(dirname "$0")/.." && pwd)/.work/llama.cpp}
PY=${ORACLE_PY:-python3}
PIN=1744c6bde
export HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
have=$(git -C "$LLAMACPP" rev-parse --short=9 HEAD)
[ "$have" = "$PIN" ] || { echo "FAIL pin: $LLAMACPP at $have, oracle pinned to $PIN"; exit 1; }
[ $# -ge 2 ] || { sed -n 2,7p "$0"; exit 2; }
timed() { local log=$1; shift; /usr/bin/time -f 'oracle_time %e s %M KB' -o "$log.time" "$@" > "$log" 2>&1 || { echo "FAIL $1: see $log"; exit 1; }; cat "$log.time"; }
case $1 in
  convert)  timed "$3.log" env PYTHONPATH="$LLAMACPP/gguf-py" "$PY" "$LLAMACPP/convert_hf_to_gguf.py" "$2" --outfile "$3" --outtype "$4" ;;
  quantize) timed "$3.log" "$LLAMACPP/build/bin/llama-quantize" --pure "$2" "$3" "$4" ;;
  new_metadata) PYTHONPATH="$LLAMACPP/gguf-py" "$PY" "$LLAMACPP/gguf-py/gguf/scripts/gguf_new_metadata.py" --force "$2" "$3" "${@:4}" ;;
  dump)     PYTHONPATH="$LLAMACPP/gguf-py" "$PY" "$LLAMACPP/gguf-py/gguf/scripts/gguf_dump.py" "$2" ;;
  *) echo "FAIL usage: unknown command $1"; exit 2 ;;
esac
