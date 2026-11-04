"""Quantized rows against gguf-py's own quantizer: every tensor of FLOAT.gguf (F32 and F16, tools/mkfloat.py) quantized by the
Mojo `quantize` to TYPE must equal gguf.quants.quantize(float values, TYPE) row by row, so the F32 and F16 load paths and the
all-zero, flat, inf and denormal rows are all compared. TYPE: Q8_0 Q4_0 TQ1_0 TQ2_0 (the ones gguf-py can also quantize).
    quant_check.py FLOAT.gguf QUANT.gguf TYPE      exit 0 only when every row is byte-equal. Needs LLAMACPP (default .work/llama.cpp)."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("LLAMACPP", os.path.join(ROOT, ".work", "llama.cpp")), "gguf-py"))
from gguf import GGUFReader  # noqa: E402
from gguf.quants import quantize, GGMLQuantizationType as Q  # noqa: E402

src, out, t = sys.argv[1], sys.argv[2], Q[sys.argv[3]]
got = {x.name: x for x in GGUFReader(out).tensors}
bad = 0
with np.errstate(all="ignore"):  # the inf row is meant to make NaNs
    for x in GGUFReader(src).tensors:
        want = quantize(np.asarray(x.data, dtype=np.float32), t)
        g = np.asarray(got[x.name].data)
        rows = want.reshape(want.shape[0], -1)
        diff = [i for i in range(len(rows)) if g.reshape(len(rows), -1)[i].tobytes() != rows[i].tobytes()]
        print(("FAIL " if diff else "ok   ") + f"{x.name} {x.tensor_type.name} -> {t.name}: {len(rows) - len(diff)}/{len(rows)} rows equal" + (f", differ {diff}" if diff else ""))
        bad += bool(diff)
sys.exit(1 if bad else 0)
