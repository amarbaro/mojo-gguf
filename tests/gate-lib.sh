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
script() { case $1 in dump) echo gguf_dump.py;; hash) echo gguf_hash.py;; set-metadata) echo gguf_set_metadata.py;; convert-endian) echo gguf_convert_endian.py;; esac; }
# run_case NAME SRC CMD ARGS...: gguf-py's script and gguf-tool on copies of SRC ("m.gguf" in ARGS), in two directories so printed
# paths agree; stdin is $IN. EXP (default 0) is the exit code BOTH must return, so two tools failing alike do not pass for a
# success; a nonzero EXP also needs a message on both stderrs (a crash or a silent exit is not an error report).
# stdout and the written file must be equal too. Needs W (work dir) with gguf-tool built in it.
run_case() {
  local name=$1 src=$2; shift 2; local cmd=$1; shift
  rm -rf "$W/$name"; mkdir -p "$W/$name/py" "$W/$name/mj"
  cp --reflink=auto "$src" "$W/$name/py/m.gguf"; cp --reflink=auto "$src" "$W/$name/mj/m.gguf"
  local rp=0 rm_=0 exp=${EXP:-0}
  (cd "$W/$name/py" && env PYTHONPATH="$GGUFPY" "$PY" "$GGUFPY/gguf/scripts/$(script "$cmd")" "$@" > out 2> err <<< "${IN:-}") || rp=$?
  (cd "$W/$name/mj" && "$W/gguf-tool" "$cmd" "$@" > out 2> err <<< "${IN:-}") || rm_=$?
  [ "$rp" = "$exp" ] && [ "$rm_" = "$exp" ] || { echo "     $name: exit py $rp, mojo $rm_, want $exp"; return 1; }
  [ "$exp" = 0 ] || { [ -s "$W/$name/py/err" ] && [ -s "$W/$name/mj/err" ] || { echo "     $name: an error exit without a stderr message"; return 1; }; }
  cmp -s "$W/$name/py/out" "$W/$name/mj/out" || { echo "     $name: stdout differs: $(cmp "$W/$name/py/out" "$W/$name/mj/out" 2>&1 | head -1)"; return 1; }
  cmp -s "$W/$name/py/m.gguf" "$W/$name/mj/m.gguf" || { echo "     $name: file differs"; return 1; }
}
rejects() { ! "$@" > /dev/null 2>&1; }  # a negative control: the gate must fail this command
