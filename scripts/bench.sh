#!/usr/bin/env bash
# quantize --pure vs llama-quantize --pure (pinned llama.cpp build) on the Qwen2.5-0.5B bf16 oracle file, Q8_0 Q4_0 Q4_K,
# 3 alternating rounds, wall + peak RSS each; every round checks both outputs are the same bytes.
set -euo pipefail
cd "$(dirname "$0")/.."
source tests/gate-lib.sh
LQ=$LLAMACPP/build/bin/llama-quantize; IN=$ORACLE/bf16.gguf; W=.work/bench; mkdir -p $W
[ -x "$LQ" ] || { echo "FAIL setup: build $LQ (cmake --build build --target llama-quantize in $LLAMACPP)"; exit 1; }
[ -e "$IN" ] || { echo "FAIL setup: no $IN (tools/mkoracle.sh)"; exit 1; }
"$MOJO" build -O3 src/quantize.mojo -I src -I lib -o $W/quantize > $W/build.log 2>&1 || { echo "FAIL build: src/quantize.mojo"; cat $W/build.log; exit 1; }
for r in 1 2 3; do for t in ${TYPES:-Q8_0 Q4_0 Q4_K}; do
  arms=(mojo llama); (( r % 2 == 0 )) && arms=(llama mojo)
  for a in "${arms[@]}"; do
    bin=$W/quantize; [ $a = llama ] && bin=$LQ
    /usr/bin/time -f "%e %M" -o $W/t "$bin" --pure "$IN" "$W/$a-$t.gguf" "$t" > $W/$a.log 2>&1 || { echo "FAIL run: $a $t"; tail -3 $W/$a.log; exit 1; }
    read -r s kb < $W/t; echo "round $r $t $a wall_s $s peak_MB $((kb / 1024))"
  done
  cmp -s "$W/mojo-$t.gguf" "$W/llama-$t.gguf" || { echo "FAIL bytes: $t differs from llama-quantize"; exit 1; }
done; done
