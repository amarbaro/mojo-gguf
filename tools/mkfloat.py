"""A float-only GGUF for the negative gate (tests/negative-gate.sh), written with the pinned gguf-py: an F32 and an F16
tensor, 4 rows x 64 columns, one row all zero and one flat. Quantizable, unlike tiny.gguf (it has an I8 tensor).
    mkfloat.py OUT.gguf        Oracle only: product code is Mojo. Needs LLAMACPP (default .work/llama.cpp)."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("LLAMACPP", os.path.join(ROOT, ".work", "llama.cpp")), "gguf-py"))
from gguf import GGUFWriter  # noqa: E402

rng = np.random.default_rng(7)
x = (rng.standard_normal((4, 64)) * 0.5).astype(np.float32)
x[1] = 0
x[2] = 0.25
w = GGUFWriter(sys.argv[1], "qwen2")
w.add_name("mojo-gguf-float")
w.add_tensor("blk.0.f32.weight", x)
w.add_tensor("blk.0.f16.weight", x.astype(np.float16))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
