"""U2 lookup check: map_tensor_name (u2-lookup binary) against gguf-py TensorNameMap, case by case.

usage: u2-lookup-check.py GGUFPY_DIR U2_LOOKUP_BIN CASES_TSV
Cases: every key of TensorNameMap(arch, n).mapping at n=2 and n=5, expected = its gguf-py value; every key seen at
n=5 looked up at n=2 (expected = n=2 value, or unmapped when the key needs a block >= 2); malformed block ids.
Prints PASS k/k or FAIL with the first mismatches; exit 0 only on PASS.
"""
import subprocess, sys

sys.path.insert(0, sys.argv[1])
from gguf import constants as C
from gguf.tensor_mapping import TensorNameMap

ARCHS = ["LLAMA", "QWEN2", "QWEN35", "QWEN35MOE", "QWEN4EXP"]
PRISTINE = dict(TensorNameMap.block_mappings_cfg)


def mapping(arch, n):
    TensorNameMap.block_mappings_cfg = dict(PRISTINE)
    return {k: v[1] for k, v in TensorNameMap(C.MODEL_ARCH[arch], n).mapping.items()}


cases = []
for arch in ARCHS:
    name = C.MODEL_ARCH_NAMES[C.MODEL_ARCH[arch]]
    m2, m5 = mapping(arch, 2), mapping(arch, 5)
    for k, v in m2.items():
        cases.append((name, 2, k, v))
    for k, v in m5.items():
        cases.append((name, 5, k, v))
        cases.append((name, 2, k, m2.get(k, "")))
    for probe in ["model.layers.0.bogus.weight", "model.layers.00.self_attn.q_proj.weight",
                  "model.layers.-1.mlp.up_proj.weight", "model.layers.1x.mlp.up_proj.weight", ""]:
        cases.append((name, 2, probe, ""))
cases.append(("nope", 2, "model.embed_tokens", ""))

with open(sys.argv[3], "w") as f:
    for arch, n, k, _ in cases:
        f.write(f"{arch}\t{n}\t{k}\n")

got = subprocess.run([sys.argv[2], sys.argv[3]], capture_output=True, text=True, check=True).stdout.split("\n")
got = got[:-1] if got and got[-1] == "" else got
if len(got) != len(cases):
    sys.exit(f"FAIL lookup: {len(got)} answers for {len(cases)} cases")
bad = [(c, g) for c, g in zip(cases, got) if c[3] != g]
if bad:
    for (arch, n, k, want), g in bad[:10]:
        print(f"  {arch} n={n} {k!r}: want {want!r} got {g!r}")
    print(f"FAIL {len(bad)}/{len(cases)} lookups differ from gguf-py")
    sys.exit(1)
print(f"PASS {len(cases)}/{len(cases)} lookups equal gguf-py TensorNameMap")
