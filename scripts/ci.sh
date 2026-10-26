#!/usr/bin/env bash
# CI entry point, same on a GitHub runner and a local machine: install, build, then every CPU gate that needs no model
# file and no llama.cpp build (fixtures are synthetic, the oracle is the pinned llama.cpp's Python). The gates that
# need Qwen2.5-0.5B, a llama.cpp build or the author's model files run locally: scripts/oracle-gates.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "FAIL setup: uv not installed"; exit 1; }
uv sync || { echo "FAIL setup: uv sync"; exit 1; }
.venv/bin/python -c 'import platform; print("oracle python", platform.python_version())'
scripts/fetch-llamacpp.sh || { echo "FAIL setup: scripts/fetch-llamacpp.sh"; exit 1; }
mkdir -p .work/t
M=.venv/bin/mojo
# every program builds; the ones a gate below does not build itself are built here
for t in src/hf2gguf:hf2gguf src/quantize:quantize src/lora2gguf:lora2gguf src/hf_update:hf-update tools/type-census:type-census examples/quickstart:quickstart; do
  $M build -I src "${t%%:*}.mojo" -o ".work/t/${t##*:}" 2> .work/build.err || { echo "FAIL build: ${t%%:*}.mojo"; cat .work/build.err; exit 1; }
done
gate() { local name=$1; shift
  "$@" > ".work/t/$name.log" 2>&1 || { echo "FAIL $name: $*"; tail -20 ".work/t/$name.log"; exit 1; }
  tail -1 ".work/t/$name.log" | grep -q '^PASS' || { echo "FAIL $name: no PASS line at the end of .work/t/$name.log"; tail -5 ".work/t/$name.log"; exit 1; }
  echo "ok   $name: $(tail -1 ".work/t/$name.log")"; }
gate fixtures-ggml tools/mkoracle-ggml.sh
gate core          tests/core-gate.sh
gate cmp-selftest  tests/gguf-cmp-selftest.sh .work/core/fx/tiny.gguf
gate u2            tests/u2-gate.sh
gate u2-lookup     tests/u2-lookup-check.sh
gate u9            tests/u9-gate.sh
gate u9-parity     tests/u9-parity.sh
gate naming        check/u3-naming-check.sh
gate u8-synth      .venv/bin/python tools/u8-synth-check.py .work/t/hf-update .work/llama.cpp .work/t/u8synth
.work/t/quickstart .work/core/fx/tiny.gguf > /dev/null || { echo "FAIL example: examples/quickstart.mojo"; exit 1; }
echo "PASS ci"
