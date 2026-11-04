#!/usr/bin/env bash
# Hermetic negative gate (runs in scripts/ci.sh): inputs a tool must refuse or handle, on a synthetic float-only fixture (tools/mkfloat.py).
#   tests/negative-gate.sh    exit 0 only on PASS n/n
# gguf-rewrite and quantize with OUT the same file as IN (a hard link too): exit non-zero, IN untouched.
source "$(dirname "$0")/gate-lib.sh"
check llama-cpp-pinned pinned
W=$PWD/.work/neg; rm -rf "$W"; mkdir -p "$W"
export LLAMACPP
for t in gguf_rewrite:gguf-rewrite quantize:quantize; do
  "$MOJO" build src/${t%%:*}.mojo -I src -o "$W/${t##*:}" > "$W/build-${t##*:}.log" 2>&1 || { echo "FAIL build: $W/build-${t##*:}.log"; exit 1; }
done
"$PY" tools/mkfloat.py "$W/float.gguf" > "$W/float.log" 2>&1 || { echo "FAIL fixture: $W/float.log"; exit 1; }
T=$W/float.gguf

refuses() { # TOOL ARGS...: run on a copy of the fixture with OUT = IN (or a hard link to it); IN must survive
  local mode=$1; shift; cp "$T" "$W/in.gguf"; cp "$T" "$W/ref.gguf"; rm -f "$W/ln.gguf"
  local out=$W/in.gguf; [ "$mode" = link ] && { ln "$W/in.gguf" "$W/ln.gguf"; out=$W/ln.gguf; }
  local rc=0; "$@" "$W/in.gguf" "$out" > "$W/refuses.log" 2>&1 || rc=$?
  [ "$rc" != 0 ] || { echo "     exit 0: wrote over its own input"; return 1; }
  cmp -s "$W/in.gguf" "$W/ref.gguf" || { echo "     input changed: $(stat -c %s "$W/in.gguf") bytes"; return 1; }
}
check rewrite-same-file   refuses same "$W/gguf-rewrite"
check rewrite-hard-link   refuses link "$W/gguf-rewrite"
quant() { local in=$1 out=$2; shift 2; "$W/quantize" --pure "$in" "$out" Q8_0 "$@"; }
qsame() { local in=$1 out=$2; quant "$in" "$out"; }
check quantize-same-file  refuses same qsame
check quantize-hard-link  refuses link qsame

finish
