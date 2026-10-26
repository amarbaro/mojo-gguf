#!/usr/bin/env bash
# hfconv.naming vs gguf-py Metadata.get_model_id_components / id_to_title / size_label on a list of model ids.
#   check/u3-naming-check.sh          exit 0 only when every line agrees
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MAIN=$PWD
LLAMACPP=${LLAMACPP:-$MAIN/.work/llama.cpp}
W=.work/u3; mkdir -p "$W"
"$MAIN/.venv/bin/mojo" build check/u3-naming-probe.mojo -I src -o "$W/naming-probe" > "$W/naming-build.log" 2>&1 || { echo "FAIL build: $W/naming-build.log"; exit 1; }
IDS=(qwen2.5-0.5b-instruct-hf Qwen/Qwen2.5-7B-Instruct Llama-3.2-1B-Instruct bloom-7b1 Mixtral-8x7B-Instruct-v0.1
     granite-4.2-3b-bf16 Meta-Llama-3.1-8B phi-2 gemma-2-9b-it Qwen3-30B-A3B tinyllama-1.1b-chat-v1.0 model
     mistral-7b-instruct-v0.2-Q4_K_M SmolLM2-135M deepseek-coder-6.7b-base x-large Falcon3-1B-Base falcon-180B-chat
     MiniCPM-2B-sft-bf16 OLMo-2-0425-1B Hermes-3-Llama-3.1-8B Phi-3.5-mini-instruct "my model" ./local/dir-7b Yi-1.5-34B-Chat-16K)
ARGS=(); for id in "${IDS[@]}"; do for t in 494032768 7615616512 1100048384; do ARGS+=("$id" "$t"); done; done
"$W/naming-probe" "${ARGS[@]}" > "$W/naming-mojo.txt"
"$MAIN/.venv/bin/python" -I - "$LLAMACPP" "$W/naming-mojo.txt" "${ARGS[@]}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/gguf-py")
from gguf.metadata import Metadata
from gguf.utility import size_label
want = []
a = sys.argv[3:]
for id_, t in zip(a[0::2], a[1::2]):
    t = int(t)
    full, org, base, ft, ver, lab = Metadata.get_model_id_components(id_, t)
    n = lambda v: v or ""
    title = Metadata.id_to_title(full) if full else ""
    want.append(f"{id_} {t} | {title} | {n(org)} | {n(base)} | {n(ft)} | {n(ver)} | {n(lab)} | {size_label(t, t, 0, 0)}")
got = open(sys.argv[2]).read().splitlines()
bad = [(w, g) for w, g in zip(want, got) if " ".join(w.split()) != " ".join(g.split())]
for w, g in bad[:10]:
    print("FAIL naming\n  want:", w, "\n  got: ", g)
ok = not bad and len(want) == len(got)
print(("PASS" if ok else "FAIL") + f" {len(want) - len(bad)}/{len(want)} naming lines agree")
sys.exit(0 if ok else 1)
PY
