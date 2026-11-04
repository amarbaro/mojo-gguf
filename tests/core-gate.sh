#!/usr/bin/env bash
# Hermetic core gate (runs in scripts/ci.sh): gguf-rewrite, gguf-tool and gguf-dequant on synthetic fixtures
# (tools/mkfixture.py, written by the pinned gguf-py) against the pinned gguf-py scripts. No model, no llama.cpp build.
#   tests/core-gate.sh    exit 0 only on PASS n/n
# Rewrite: identical bytes on two fixtures (alignment 32 and 64), summary line equal to gguf-py's, two metadata edits
# equal to gguf_new_metadata.py. Tool: dump (4 layouts), hash (2), set-metadata (2), convert-endian (1 byteswap, 2 refused): stdout,
# exit code and written bytes equal to the scripts'. Corrupt input: a truncated file is an exit 1 with a stderr message from
# every command (EXP is the exit both must return); a both-tools-fail case expecting success is rejected. Dequant: every row of the 8 types gguf-py can also dequantize,
# bit-equal to gguf-py.
source "$(dirname "$0")/gate-lib.sh"
check llama-cpp-pinned pinned
W=$PWD/.work/core; rm -rf "$W"; mkdir -p "$W"; FX=$W/fx
LLAMACPP=$LLAMACPP "$PY" tools/mkfixture.py "$FX" > "$W/fixture.log" 2>&1 || { echo "FAIL fixture: $W/fixture.log"; exit 1; }
for t in gguf_rewrite:gguf-rewrite gguf_tool:gguf-tool gguf_dequant:gguf-dequant; do
  "$MOJO" build src/${t%%:*}.mojo -I src -o "$W/${t##*:}" > "$W/build-${t##*:}.log" 2>&1 || { echo "FAIL build: $W/build-${t##*:}.log"; exit 1; }
done

rewrite() { local src=$1 tag=$2 ref=${3:-$1}; shift 3 || shift $#
  "$W/gguf-rewrite" "$src" "$W/$tag.out" "$@" > "$W/$tag.log" 2>&1 || return 1
  cmp -s "$ref" "$W/$tag.out" || { echo "     $tag: $(cmp "$ref" "$W/$tag.out" 2>&1 | head -1)"; return 1; }; }
summary() { local want; want=$("$PY" -I -c "import sys; sys.path.insert(0, sys.argv[2]); from gguf import GGUFReader; r = GGUFReader(sys.argv[1]); print(f'kv {len(r.fields) - 3} tensors {len(r.tensors)} alignment {r.alignment} data_offset {r.data_offset}')" "$1" "$GGUFPY")
  [ "$(head -1 "$W/$2.log")" = "$want" ] || { echo "     want '$want' got '$(head -1 "$W/$2.log")'"; return 1; }; }
edit() { local src=$1 tag=$2
  (cd "$W" && env PYTHONPATH="$GGUFPY" "$PY" "$GGUFPY/gguf/scripts/gguf_new_metadata.py" --force "$src" "$W/$tag.ref" --general-name edited --remove-metadata tokenizer.ggml.scores > "$W/$tag.ref.log" 2>&1) || return 1
  rewrite "$src" "$tag" "$W/$tag.ref" --name edited --remove tokenizer.ggml.scores; }
for f in tiny tiny-a64; do
  check "rewrite-$f" rewrite "$FX/$f.gguf" "$f"
  check "summary-$f" summary "$FX/$f.gguf" "$f"
  check "edit-$f" edit "$FX/$f.gguf" "edit-$f"
done

T=$FX/tiny.gguf; A=$FX/tiny-a64.gguf
check dump           run_case dump    "$T" dump m.gguf
check dump-no-tensors run_case dump-nt "$A" dump --no-tensors m.gguf
check dump-json      run_case dump-js "$T" dump --json m.gguf
check dump-markdown  run_case dump-md "$A" dump --markdown m.gguf
check hash           run_case hash    "$T" hash m.gguf
check hash-no-layer  run_case hash-nl "$A" hash --no-layer m.gguf
check set-int        run_case set-int "$T" set-metadata m.gguf qwen2.context_length 8192 --force
EXP=1 check set-missing run_case set-miss "$T" set-metadata m.gguf no.such.key 1 --force
# the fixtures hold an I8 tensor, which gguf_convert_endian.py refuses: both tools must refuse it (exit 1, a message);
# the success case is a fixture of byteswappable types only
IN=YES EXP=1 check endian-i8-refused-tiny run_case end-t "$T" convert-endian m.gguf big
IN=YES EXP=1 check endian-i8-refused-a64  run_case end-a "$A" convert-endian m.gguf big
IN=YES check endian-big-supported run_case end-s "$FX/tiny-endian.gguf" convert-endian m.gguf big

# corrupt input: a truncated file (cut inside the tensor infos) must be an error exit with a message on every command,
# from both tools; and a case where both tools fail alike must NOT pass as a success (negative control)
head -c 400 "$FX/tiny.gguf" > "$FX/trunc.gguf"
EXP=1 check dump-truncated        run_case dump-tr    "$FX/trunc.gguf" dump m.gguf
EXP=1 check hash-truncated        run_case hash-tr    "$FX/trunc.gguf" hash m.gguf
EXP=1 check set-truncated         run_case set-tr     "$FX/trunc.gguf" set-metadata m.gguf qwen2.context_length 8192 --force
IN=YES EXP=1 check endian-truncated run_case end-tr   "$FX/trunc.gguf" convert-endian m.gguf big
check mutant-both-fail-rejected   rejects run_case mut-tr "$FX/trunc.gguf" dump m.gguf
errexit() { local rc=0; "$@" > "$W/errexit.err" 2>&1 || rc=$?;  # gguf-rewrite and gguf-dequant print their FAIL line on stdout
  [ "$rc" = 1 ] && [ -s "$W/errexit.err" ]; }
check rewrite-truncated errexit "$W/gguf-rewrite" "$FX/trunc.gguf" "$W/trunc.out"
check dequant-truncated errexit "$W/gguf-dequant" "$FX/trunc.gguf" token_embd.weight 0 1 "$W/trunc.f32"

deq() { "$PY" -I - "$GGUFPY" "$W/gguf-dequant" "$W" "$1" <<'PY'
import subprocess, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
from gguf.quants import dequantize, GGMLQuantizationType as Q
cli, w, path = sys.argv[2:5]
ok = {Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q8_0, Q.TQ1_0}
done, bad = 0, 0
for t in GGUFReader(path).tensors:
    if t.tensor_type not in ok:
        continue
    want = np.asarray(t.data if t.tensor_type in (Q.F32,) else dequantize(t.data, t.tensor_type) if t.tensor_type != Q.F16 else t.data.astype(np.float32), dtype=np.float32).reshape(-1)
    rows = int(t.shape[1])
    subprocess.run([cli, path, t.name, "0", str(rows), w + "/deq.f32"], check=True, capture_output=True)
    got = np.fromfile(w + "/deq.f32", dtype=np.float32)
    done += 1
    if got.shape != want.shape or got.tobytes() != want.tobytes():
        bad += 1
        print("     differs:", t.name, t.tensor_type.name)
print(f"     {done - bad}/{done} types bit-equal")
sys.exit(1 if bad or done != 8 else 0)
PY
}
check dequant-8-types deq "$T"
finish
