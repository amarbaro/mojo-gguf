"""A float-only GGUF for the negative gate (tests/negative-gate.sh), written with the pinned gguf-py: an F32 and an F16
tensor, 8 rows x 256 columns (a multiple of every quantizer's block, TQ1_0's 256 included): random, all zero, flat, negative flat,
one +inf element, 1e-30 (denormal scales), half zero half flat, 65000 (near the F16 limit). Quantizable, unlike tiny.gguf
(it has an I8 tensor); tools/quant_check.py compares the quantized rows with gguf-py's own quantizer.
    mkfloat.py OUT.gguf        Oracle only: product code is Mojo. Needs LLAMACPP (default .work/llama.cpp)."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("LLAMACPP", os.path.join(ROOT, ".work", "llama.cpp")), "gguf-py"))
from gguf import GGUFWriter  # noqa: E402

rng = np.random.default_rng(7)
x = (rng.standard_normal((8, 256)) * 0.5).astype(np.float32)
x[1] = 0
x[2] = 0.25
x[3] = -3.5
x[4, 5] = np.inf
x[5] = np.float32(1e-30)
x[6, :128] = 0
x[7] = np.float32(65000)
w = GGUFWriter(sys.argv[1], "qwen2")
w.add_name("mojo-gguf-float")
w.add_tensor("blk.0.f32.weight", x)
w.add_tensor("blk.0.f16.weight", x.astype(np.float16))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
