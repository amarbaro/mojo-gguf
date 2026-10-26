"""Legacy GGML-family fixtures for U9 (no such file exists on this box): a tiny llama-1-shaped model, seeded, in
four containers: GGJTv3 F16, GGJTv3 Q8_0, GGMFv1 F32, GGML (unversioned) F32. Layout follows what the pinned
convert_llama_ggml_to_gguf.py reads: magic, version, 7 x u32 hparams, vocab (len, bytes[, f32 score]), tensors
(n_dims, name_len, type, dims, name, 32-byte pad on GGJT, data).
usage: mk-ggml.py OUT_DIR"""
import os, struct, sys
import numpy as np

V, E, MULT, NH, NL, ROT = 64, 64, 32, 4, 2, 16
FF = ((2 * (4 * E) // 3 + MULT - 1) // MULT) * MULT
out = sys.argv[1]; os.makedirs(out, exist_ok=True)

def tensors():
    yield b"tok_embeddings.weight", (E, V)
    yield b"norm.weight", (E,)
    yield b"output.weight", (E, V)
    for i in range(NL):
        p = b"layers.%d." % i
        for n in (b"attention.wq.weight", b"attention.wk.weight", b"attention.wv.weight", b"attention.wo.weight"):
            yield p + n, (E, E)
        yield p + b"attention_norm.weight", (E,)
        yield p + b"feed_forward.w1.weight", (E, FF)
        yield p + b"feed_forward.w2.weight", (FF, E)
        yield p + b"feed_forward.w3.weight", (E, FF)
        yield p + b"ffn_norm.weight", (E,)

def data(dims, ty, rng):
    n = int(np.prod(dims))
    if ty == 0:
        return (rng.standard_normal(n) * 0.02).astype(np.float32).tobytes()
    if ty == 1:
        return (rng.standard_normal(n) * 0.02).astype(np.float16).tobytes()
    blocks = bytearray()
    for _ in range(n // 32):
        blocks += np.float16(abs(rng.standard_normal()) * 0.01 + 1e-3).tobytes()
        blocks += rng.integers(-127, 128, 32, dtype=np.int8).tobytes()
    return bytes(blocks)

def write(name, magic, version, ftype, wtype, scores, pad):
    rng = np.random.default_rng(len(name))
    b = bytearray(magic)
    if version is not None:
        b += struct.pack("<I", version)
    b += struct.pack("<7I", V, E, MULT, NH, NL, ROT, ftype)
    for i in range(V):
        t = b"<tok%d>" % i if i < 3 else b"t%02d" % i
        b += struct.pack("<I", len(t)) + t
        if scores:
            b += struct.pack("<f", -float(i))
    for n, dims in tensors():
        ty = 0 if len(dims) == 1 else wtype
        b += struct.pack("<3I", len(dims), len(n), ty) + struct.pack(f"<{len(dims)}I", *dims) + n
        if pad:
            b += b"\0" * (((len(b) + 31) & ~31) - len(b))
        b += data(dims, ty, rng)
    open(os.path.join(out, name), "wb").write(b)
    print(name, len(b), "bytes")

write("ggjt3-f16.bin", b"tjgg", 3, 1, 1, True, True)
write("ggjt3-q8_0.bin", b"tjgg", 3, 7, 8, True, True)
write("ggmf1-f32.bin", b"fmgg", 1, 0, 0, True, False)
write("ggml-f32.bin", b"lmgg", None, 0, 0, False, False)
