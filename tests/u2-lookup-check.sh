#!/usr/bin/env bash
# U2 lookup gate companion: builds src/u2_lookup.mojo and checks map_tensor_name against gguf-py TensorNameMap.
#   tests/u2-lookup-check.sh   exit 0 only on PASS
set -euo pipefail
source "$(dirname "$0")/gate-lib.sh"
W=.work/u2; mkdir -p "$W"
"$MOJO" build src/u2_lookup.mojo -I src -I lib -o "$W/u2-lookup" > "$W/lookup-build.log" 2>&1 || { echo "FAIL build: see $W/lookup-build.log"; exit 1; }
"$PY" -I tests/u2-lookup-check.py "$GGUFPY" "$W/u2-lookup" "$W/lookup-cases.tsv"
