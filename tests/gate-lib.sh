# Shared by tests/*-gate.sh (source it). Fixtures come from .work/oracle (tools/mkoracle.sh), the oracle scripts and
# library from the pinned llama.cpp checkout (scripts/fetch-llamacpp.sh, or LLAMACPP=/path to your own at the pin).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MAIN=$PWD
ORACLE=$MAIN/.work/oracle/qwen25-05b
LLAMACPP=${LLAMACPP:-$MAIN/.work/llama.cpp}
GGUFPY=$LLAMACPP/gguf-py
PY=$MAIN/.venv/bin/python
MOJO=$MAIN/.venv/bin/mojo
MODELS=${MODELS:-$HOME/Models}
HF05=${QWEN_HF:-$MODELS/qwen2.5-0.5b-instruct-hf}
fails=(); n=0
check() { n=$((n + 1)); local name=$1; shift; if "$@"; then echo "ok   $name"; else echo "FAIL $name"; fails+=("$name"); fi; }
pinned() { [ "$(git -C "$LLAMACPP" rev-parse --short=9 HEAD)" = 1744c6bde ]; }
finish() { if [ ${#fails[@]} -gt 0 ]; then echo "FAIL ${#fails[@]}/$n ${fails[*]}"; exit 1; fi; echo "PASS $n/$n"; }
