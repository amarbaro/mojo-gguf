#!/usr/bin/env bash
# Hermetic negative gate (runs in scripts/ci.sh): inputs a tool must refuse or handle, on a synthetic float-only fixture (tools/mkfloat.py).
#   tests/negative-gate.sh    exit 0 only on PASS n/n
# gguf-rewrite and quantize with OUT the same file as IN (a hard link too): exit non-zero, IN untouched.
# Checkpoint: tensor entries with a negative or reversed offset, a shape that does not match the byte count, an overflowing shape, one offset.
# quantize NTHREADS 0 (llama-quantize: all cores): same bytes as an explicit count, tensors not left zero.
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

nthreads() {
  quant "$T" "$W/q2.gguf" 2 > "$W/q2.log" 2>&1 || return 1
  quant "$T" "$W/q0.gguf" 0 > "$W/q0.log" 2>&1 || { echo "     nthreads 0: $(tail -1 "$W/q0.log")"; return 1; }
  cmp -s "$W/q0.gguf" "$W/q2.gguf" || { echo "     nthreads 0 differs from 2: $(cmp "$W/q0.gguf" "$W/q2.gguf" 2>&1 | head -1)"; return 1; }
}
check quantize-nthreads-0 nthreads

# safetensors tensor entries that must be rejected: offsets outside the data, reversed, or not matching the shape
"$MOJO" build tests/st_open.mojo -I src -o "$W/st_open" > "$W/build-st_open.log" 2>&1 || { echo "FAIL build: $W/build-st_open.log"; exit 1; }
stcase() { # NAME HEADER-JSON DATA-BYTES WANT(ok|error)
  local d=$W/st-$1; rm -rf "$d"; mkdir -p "$d"
  "$PY" -I -c 'import struct, sys; h = sys.argv[1].encode(); open(sys.argv[3] + "/model.safetensors", "wb").write(struct.pack("<Q", len(h)) + h + bytes(int(sys.argv[2])))' "$2" "$3" "$d"
  local got; got=$("$W/st_open" "$d" 2>&1 | head -1)
  [ "${got%%[: ]*}" = "$4" ] || { echo "     $1: wanted $4, got: $got"; return 1; }
}
check st-control     stcase control '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}' 8 ok
check st-negative-lo stcase neglo    '{"a":{"dtype":"F32","shape":[2],"data_offsets":[-8,0]}}' 8 error
check st-reversed    stcase rev      '{"a":{"dtype":"F32","shape":[0],"data_offsets":[8,0]}}' 8 error
check st-shape-size  stcase shape    '{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}' 8 error
check st-overflow    stcase ovf      '{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}' 0 error
finish
