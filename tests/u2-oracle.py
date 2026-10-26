"""U2 oracle: every table U2 generates, dumped from the pinned gguf-py as one compact JSON (sorted keys).
usage: u2-oracle.py GGUFPY_DIR > ref.json"""
import json, sys
sys.path.insert(0, sys.argv[1])
from gguf import constants as C
from gguf.tensor_mapping import TensorNameMap

def keys(cls, pre=""):
    out = {}
    for k, v in vars(cls).items():
        if k.startswith("_"):
            continue
        if isinstance(v, type):
            out.update(keys(v, pre + k + "."))
        elif isinstance(v, str):
            out[pre + k] = v
    return out

ARCHS = ["LLAMA", "QWEN2", "QWEN35", "QWEN35MOE", "QWEN4EXP"]
o = {
    "value_types": {t.name: int(t) for t in C.GGUFValueType},
    "quant_types": {t.name: int(t) for t in C.GGMLQuantizationType},
    "quant_sizes": {t.name: list(C.GGML_QUANT_SIZES[t]) for t in C.GGML_QUANT_SIZES},
    "arch_names": {a.name: C.MODEL_ARCH_NAMES[a] for a in C.MODEL_ARCH},
    "tensor_names": {t.name: C.TENSOR_NAMES[t] for t in C.MODEL_TENSOR if t in C.TENSOR_NAMES},
    "arch_tensors": {C.MODEL_ARCH_NAMES[a]: [t.name for t in C.MODEL_TENSORS[a]] for a in C.MODEL_TENSORS},
    "keys": keys(C.Keys),
    "name_map": {},
}
for a in ARCHS:
    arch = C.MODEL_ARCH[a]
    m = TensorNameMap(arch, 2)
    o["name_map"][C.MODEL_ARCH_NAMES[arch]] = {k: v[1] for k, v in m.mapping.items()}
json.dump(o, sys.stdout, sort_keys=True, separators=(",", ":"))
