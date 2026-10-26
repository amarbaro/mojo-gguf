"""Legacy GGML variants for the U9 parity check, beyond the gate fixtures of tests/mk-ggml.py: GGJT v1 and v2,
GGMF with scores, GGML with F16 weights, a 300-token vocab (byte tokens, spaces, empty items, ids past 258), and a
GGJT v2 Q8_0 file the script refuses. Same layout as mk-ggml.py. usage: mk-ggml-variants.py OUT_DIR"""
import os, struct, sys
import numpy as np

out = sys.argv[1]; os.makedirs(out, exist_ok=True)

def tokens(V):
    t = [b"<unk>", b"<s>", b"</s>"]
    for i in range(3, V):
        if i == 3: t.append(b"")  # empty item: control
        elif i == 4: t.append(b"A")  # one byte in 3..258: <0x41>
        elif i == 5: t.append(b"\xff")  # byte 0xFF: <0xFF>
        elif i == 299: t.append(b"Z")  # one byte past 258: kept as is
        elif i % 7 == 0: t.append(b"a b c")  # spaces: U+2581
        else: t.append(b"t%02d" % i)
    return t

def write(name, magic, version, ftype, wtype, scores, pad, V=64, E=64, MULT=32, NH=4, NL=2, ROT=16):
    rng = np.random.default_rng(len(name))
    FF = ((2 * (4 * E) // 3 + MULT - 1) // MULT) * MULT
    b = bytearray(magic)
    if version is not None:
        b += struct.pack("<I", version)
    b += struct.pack("<7I", V, E, MULT, NH, NL, ROT, ftype)
    for i, t in enumerate(tokens(V)):
        b += struct.pack("<I", len(t)) + t
        if scores:
            b += struct.pack("<f", -0.0 if i == 0 else -float(i) + 0.25)
    shapes = [(b"tok_embeddings.weight", (E, V)), (b"norm.weight", (E,)), (b"output.weight", (E, V))]
    for i in range(NL):
        p = b"layers.%d." % i
        shapes += [(p + n, (E, E)) for n in (b"attention.wq.weight", b"attention.wk.weight", b"attention.wv.weight", b"attention.wo.weight")]
        shapes += [(p + b"attention_norm.weight", (E,)), (p + b"feed_forward.w1.weight", (E, FF)),
                   (p + b"feed_forward.w2.weight", (FF, E)), (p + b"feed_forward.w3.weight", (E, FF)),
                   (p + b"ffn_norm.weight", (E,))]
    for n, dims in shapes:
        ty = 0 if len(dims) == 1 else wtype
        b += struct.pack("<3I", len(dims), len(n), ty) + struct.pack(f"<{len(dims)}I", *dims) + n
        if pad:
            b += b"\0" * (((len(b) + 31) & ~31) - len(b))
        k = int(np.prod(dims))
        if ty == 0:
            b += (rng.standard_normal(k) * 0.02).astype(np.float32).tobytes()
        elif ty == 1:
            b += (rng.standard_normal(k) * 0.02).astype(np.float16).tobytes()
        else:
            for _ in range(k // 32):
                b += np.float16(abs(rng.standard_normal()) * 0.01 + 1e-3).tobytes()
                b += rng.integers(-127, 128, 32, dtype=np.int8).tobytes()
    open(os.path.join(out, name), "wb").write(bytes(b))
    print(name, len(b), "bytes")

write("ggjt2-f16.bin", b"tjgg", 2, 1, 1, True, True)
write("ggjt1-f32.bin", b"tjgg", 1, 0, 0, True, True)
write("ggjt2-q8_0.bin", b"tjgg", 2, 7, 8, True, True)
write("ggml-f16.bin", b"lmgg", None, 1, 1, False, False)
write("ggmf1-f16.bin", b"fmgg", 1, 1, 1, True, False)
write("ggjt3-f16-vocab300.bin", b"tjgg", 3, 1, 1, True, True, V=300)
