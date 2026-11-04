"""Synthetic GGUF fixtures for the hermetic gates (tests/core-gate.sh), written with the pinned gguf-py.

    mkfixture.py OUT_DIR       writes OUT_DIR/tiny.gguf (alignment 32), OUT_DIR/tiny-a64.gguf (alignment 64) and OUT_DIR/tiny-endian.gguf

Both carry every scalar KV type, arrays of u8/i32/f32/bool/string, and one 8-row tensor per type gguf-py can
quantize (F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 TQ1_0 TQ2_0) after a 33-byte I8 tensor, seeded. Oracle only: product code is Mojo.
Needs LLAMACPP (default .work/llama.cpp) for gguf-py."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("LLAMACPP", os.path.join(ROOT, ".work", "llama.cpp")), "gguf-py"))
from gguf import GGUFWriter, GGUFValueType as V, GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import quantize  # noqa: E402

TYPES = [Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0, Q.TQ1_0, Q.TQ2_0]
COLS, ROWS = 256, 8


def build(path, align, types=TYPES, i8=True):
    rng = np.random.default_rng(align)
    w = GGUFWriter(path, "qwen2")
    w.add_name("mojo-gguf-fixture")
    w.add_context_length(4096)
    if align != 32:
        w.add_custom_alignment(align)
    for k, v, t in [("fx.u8", 200, V.UINT8), ("fx.i8", -100, V.INT8), ("fx.u16", 60000, V.UINT16), ("fx.i16", -30000, V.INT16),
                    ("fx.u32", 4000000000, V.UINT32), ("fx.i32", -2000000000, V.INT32), ("fx.u64", 2 ** 63 + 5, V.UINT64),
                    ("fx.i64", -(2 ** 62), V.INT64), ("fx.f32", 0.1, V.FLOAT32), ("fx.f64", 0.1, V.FLOAT64), ("fx.bool", True, V.BOOL),
                    ("fx.str", "café ☃", V.STRING)]:
        w.add_key_value(k, v, t)
    w.add_key_value("fx.arr_u8", [1, 2, 255], V.ARRAY, V.UINT8)
    w.add_key_value("fx.arr_i32", [-1, 0, 7], V.ARRAY, V.INT32)
    w.add_key_value("fx.arr_f32", [0.5, -2.25, 1e-3], V.ARRAY, V.FLOAT32)
    w.add_key_value("fx.arr_bool", [True, False, True], V.ARRAY, V.BOOL)
    w.add_token_list(["<unk>", "a", "b", "ab", "café", "☃"])
    w.add_token_scores([0.0, -1.0, -2.0, -3.5, -4.0, -5.0])
    w.add_token_types([2, 1, 1, 1, 1, 1])
    if i8:
        w.add_tensor("blk.0.i8.weight", rng.integers(-128, 128, 33, dtype=np.int8))  # 33 bytes: padding to the alignment
    for t in types:
        x = (rng.standard_normal((ROWS, COLS)) * 0.5).astype(np.float32)
        if t == Q.F32:
            w.add_tensor(f"blk.0.{t.name.lower()}.weight", x)
        elif t == Q.F16:
            w.add_tensor(f"blk.0.{t.name.lower()}.weight", x.astype(np.float16))
        else:
            q = quantize(x, t)
            w.add_tensor(f"blk.0.{t.name.lower()}.weight", q, raw_shape=q.shape, raw_dtype=t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    build(os.path.join(out, "tiny.gguf"), 32)
    build(os.path.join(out, "tiny-a64.gguf"), 64)
    # only types gguf_convert_endian.py can byteswap (it refuses I8 and the K-less rest): the success case of convert-endian
    build(os.path.join(out, "tiny-endian.gguf"), 32, [Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q8_0, Q.TQ2_0], False)
    print("wrote", out)
