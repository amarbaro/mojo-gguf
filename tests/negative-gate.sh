#!/usr/bin/env bash
# Hermetic negative gate (runs in scripts/ci.sh): inputs a tool must refuse or handle, on a synthetic float-only fixture (tools/mkfloat.py).
#   tests/negative-gate.sh    exit 0 only on PASS n/n
# gguf-rewrite and quantize with OUT the same file as IN (a hard link too): exit non-zero, IN untouched.
# Checkpoint: tensor entries with a negative or reversed offset, a shape that does not match the byte count, an overflowing shape, one offset.
# coexist: a program that seeks a file and also links Gguf, Json and Shard builds and runs.
# quantize NTHREADS 0 (llama-quantize: all cores): same bytes as an explicit count, tensors not left zero.
# quantize F32 and F16 sources (zero, flat, inf, denormal rows) to Q8_0 Q4_0 TQ1_0 TQ2_0: rows equal gguf-py's quantizer; two load-path mutants are caught.
source "$(dirname "$0")/gate-lib.sh"
check llama-cpp-pinned pinned
W=$PWD/.work/neg; rm -rf "$W"; mkdir -p "$W"
export LLAMACPP
for t in gguf_rewrite:gguf-rewrite quantize:quantize gguf_tool:gguf-tool ggml2gguf:ggml2gguf gguf_dequant:gguf-dequant; do
  "$MOJO" build src/${t%%:*}.mojo -I src -I lib -o "$W/${t##*:}" > "$W/build-${t##*:}.log" 2>&1 || { echo "FAIL build: $W/build-${t##*:}.log"; exit 1; }
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

# quantize on the float fixture (F32 and F16 tensors; zero, flat, inf, denormal and near-F16-limit rows) equals gguf-py's own
# quantizer byte for byte; two mutants of the load paths (one per float type) must be caught by the same comparison
qcheck() { local t=$1 q=${2:-$W/quantize} o=$W/qf-$1.gguf
  "$q" --pure "$T" "$o" "$t" > "$W/qf-$t.log" 2>&1 || { echo "     quantize $t: $(tail -1 "$W/qf-$t.log")"; return 1; }
  "$PY" -I tools/quant_check.py "$T" "$o" "$t" > "$W/qf-$t.cmp" 2>&1 || { grep FAIL "$W/qf-$t.cmp" | head -3 | sed 's/^/     /'; return 1; }; }
for t in Q8_0 Q4_0 TQ1_0 TQ2_0; do check "quantize-float-$t" qcheck "$t"; done
qmutant() { # NAME FROM TO: quantize built from a copy of src/ with one load path broken must fail qcheck on that tensor
  local d=$W/mut-$1; rm -rf "$d"; mkdir -p "$d"; cp -r src "$d/src"
  "$PY" -I -c 'import sys; p, a, b = sys.argv[1:]; s = open(p).read(); assert a in s, "mutant does not apply: " + a; open(p, "w").write(s.replace(a, b, 1))' "$d/src/quantize.mojo" "$2" "$3" || return 1
  "$MOJO" build "$d/src/quantize.mojo" -I "$d/src" -I lib -o "$d/quantize" > "$d/build.log" 2>&1 || { echo "     mutant does not build: $d/build.log"; return 1; }
  ! qcheck Q8_0 "$d/quantize" > "$d/out.log" 2>&1 && grep -q "^FAIL blk.0.$1.weight" "$W/qf-Q8_0.cmp"; }  # a comparison failure, not a crash
check mutant-f32-load-caught qmutant f32 'dst[unsafe_offset=i] = bitcast[DType.float32](UInt32(v))
    elif t == T_BF16:' 'dst[unsafe_offset=i] = bitcast[DType.float32](UInt32(v)) * 2.0
    elif t == T_BF16:'
check mutant-f16-load-caught qmutant f16 'dst[unsafe_offset=i] = Float32(bitcast[DType.float16](UInt16(v)))' 'dst[unsafe_offset=i] = Float32(bitcast[DType.float16](UInt16(v))) * 2.0'

# safetensors tensor entries that must be rejected: offsets outside the data, reversed, or not matching the shape
"$MOJO" build tests/st_open.mojo -I src -I lib -o "$W/st_open" > "$W/build-st_open.log" 2>&1 || { echo "FAIL build: $W/build-st_open.log"; exit 1; }
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
check st-one-offset  stcase one      '{"a":{"dtype":"F32","shape":[2],"data_offsets":[5]}}' 8 error
check st-one-offset-nl stcase onenl "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[5]}}"$'\n' 8 error
# \u escapes in a tensor name: a surrogate pair decodes, a lone or mismatched surrogate and a cut-off escape are errors
check st-surrogate-pair   stcase surp  '{"a\ud83d\ude00":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}' 8 ok
check st-lone-high        stcase hi    '{"a\ud800":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}' 8 error
check st-lone-low         stcase lo    '{"a\udc00":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}' 8 error
check st-high-then-letter stcase hiu   '{"a\ud800\u0041":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}' 8 error
check st-short-u-escape   stcase short '{"a\u12":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}' 8 error
# strings from a file are not trusted to be UTF-8: a KV key, a KV value and a legacy-ggml tensor name holding 0xFF bytes come out
# as U+FFFD (like gguf-core), so what the tools print is valid UTF-8 (gguf-py refuses these files outright)
"$PY" -I -c 'import sys; b = open(sys.argv[1], "rb").read(); v = b"mojo-gguf-float"; k = b"general.architecture"
open(sys.argv[2], "wb").write(b.replace(v, v[:4] + b"\xff\xfe" + v[6:], 1).replace(k, k[:3] + b"\xff" + k[4:], 1))' "$T" "$W/badutf.gguf"
badutf_dump() { "$W/gguf-tool" dump "$W/badutf.gguf" > "$W/badutf.out" 2> "$W/badutf.err" || { echo "     dump failed: $(head -c 100 "$W/badutf.err")"; return 1; }
  "$PY" -I -c 'import sys; sys.stdout.buffer.write(b""); open(sys.argv[1], "rb").read().decode("utf-8")' "$W/badutf.out" 2>/dev/null || { echo "     dump printed invalid UTF-8"; return 1; }; }
check utf8-dump badutf_dump
badutf_ggml() { local f=$MAIN/.work/oracle/ggml/ggjt3-f16.bin; [ -s "$f" ] || { echo "     $f missing (tools/mkoracle-ggml.sh)"; return 1; }
  "$PY" -I -c 'import sys; b = open(sys.argv[1], "rb").read(); n = b"output.weight"; assert n in b; open(sys.argv[2], "wb").write(b.replace(n, b"out\xffut.weight", 1))' "$f" "$W/badname.bin" || return 1
  # an unknown tensor name is refused (exit 1) and the refusal names it: that text must be valid UTF-8
  local rc=0; "$W/ggml2gguf" --input "$W/badname.bin" --output "$W/badname.gguf" > "$W/badname.log" 2>&1 || rc=$?
  [ "$rc" = 1 ] && grep -q "Bad name" "$W/badname.log" || { echo "     ggml2gguf exit $rc: $(tail -1 "$W/badname.log")"; return 1; }
  "$PY" -I -c 'import sys; open(sys.argv[1], "rb").read().decode("utf-8")' "$W/badname.log" 2>/dev/null || { echo "     the refusal printed invalid UTF-8"; return 1; }; }
check utf8-ggml2gguf badutf_ggml
# a tensor with no dimensions or with ne0 == 0 (a one-tensor F32 file written by hand): gguf-dequant and quantize refuse it
# (exit 1, a message) instead of indexing dims[0] or dividing by zero
zero_dims() { "$PY" -I -c 'import struct, sys
nd, dims = int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
name = b"t"
info = struct.pack("<Q", len(name)) + name + struct.pack("<I", nd) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, 0)
head = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + info
open(sys.argv[1], "wb").write(head + bytes(-len(head) % 32) + bytes(32))' "$@"; }
refused() { local f=$1; shift; local rc=0; "$@" > "$W/refused.log" 2>&1 || rc=$?
  [ "$rc" = 1 ] && grep -q "^FAIL" "$W/refused.log" || { echo "     $(basename "$f"): exit $rc, $(tail -1 "$W/refused.log" | cut -c1-100)"; return 1; }; }
zero_dims "$W/zero-d.gguf" 0; zero_dims "$W/zero-ne0.gguf" 2 0 3
for f in zero-d zero-ne0; do
  check "dequant-$f"  refused "$W/$f.gguf" "$W/gguf-dequant" "$W/$f.gguf" t 0 1 "$W/$f.f32"
  check "quantize-$f" refused "$W/$f.gguf" "$W/quantize" --pure "$W/$f.gguf" "$W/$f.q8" Q8_0
done
# the library's libc declarations must not clash with the stdlib's (FileHandle.seek declares lseek)
check coexist-builds "$MOJO" build tests/coexist.mojo -I src -I lib -o "$W/coexist"
coexist() { "$W/coexist" "$W/float.gguf" "$W/st-control/model.safetensors" | grep -q '^coexist: ok'; }
check coexist-runs coexist
# py_repr equals Python's repr for every code point (gguf-tool dump text output)
"$MOJO" build tests/repr_all.mojo -I src -I lib -o "$W/repr_all" > "$W/build-repr_all.log" 2>&1 || { echo "FAIL build: $W/build-repr_all.log"; exit 1; }
reprs() { local u; u=$(sed -n 's/^# source: Python [0-9.]*, Unicode \([0-9.]*\):.*/\1/p' src/ggufx/nonprint.mojo)
  "$PY" -I -c 'import sys, unicodedata; v = unicodedata.unidata_version; v == sys.argv[1] or sys.exit(f"     oracle python has Unicode {v}, src/ggufx/nonprint.mojo was generated from {sys.argv[1]}")' "$u" || return 1
  "$W/repr_all" > "$W/repr.mojo" && "$PY" -I -c 'import sys; sys.stdout.reconfigure(encoding="utf-8")
for cp in range(0x110000):
    if not 0xD800 <= cp < 0xE000: print(repr(chr(cp)))' > "$W/repr.py" && cmp "$W/repr.py" "$W/repr.mojo"; }
check py-repr-all-code-points reprs
finish
