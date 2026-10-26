#!/usr/bin/env bash
# gguf-cmp must PASS a file against itself and FAIL on one flipped data byte and on one flipped KV byte.
#   tests/gguf-cmp-selftest.sh [REF.gguf]     default: the qwen25-05b Q8_0 oracle
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
REF=${1:-.work/oracle/qwen25-05b/Q8_0.gguf}
[ -s "$REF" ] || { echo "FAIL fixture: $REF missing (tools/mkoracle.sh)"; exit 1; }
W=.work/selftest; mkdir -p "$W"
flip() { cp --reflink=auto "$REF" "$2"; printf "\\x$(printf %02x $(( ($(od -An -tu1 -j"$1" -N1 "$2") + 1) % 256 )))" | dd of="$2" bs=1 seek="$1" conv=notrunc status=none; }
fails=()
"$PY" -I tools/gguf-cmp.py "$REF" "$REF" > "$W/same.txt" || fails+=(same)
flip $(( $(stat -c %s "$REF") - 100 )) "$W/data.gguf"
! "$PY" -I tools/gguf-cmp.py "$REF" "$W/data.gguf" > "$W/data.txt" || fails+=(data-flip)
grep -q "tensor differs" "$W/data.txt" || fails+=(data-named)
flip 200 "$W/kv.gguf"
! "$PY" -I tools/gguf-cmp.py "$REF" "$W/kv.gguf" > "$W/kv.txt" 2>&1 || fails+=(kv-flip)
rm -f "$W/data.gguf" "$W/kv.gguf"
[ ${#fails[@]} -eq 0 ] || { echo "FAIL ${#fails[@]}/4 ${fails[*]} (outputs in $W)"; exit 1; }
echo "PASS 4/4"
