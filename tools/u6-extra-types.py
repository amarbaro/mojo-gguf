"""Verifier for gguf-dequant beyond the gate's real models: random raw blocks of every supported type (all code
points, incl. the types on no ladder model: Q4_1 Q2_K IQ1_S), compared bit for bit with libggml-base `to_float`
through the CLI; also a multi-row read at an offset, and a 20 MB F32 tensor that forces 2 streaming chunks.
f16 scales with exponent 31 are cleared (NaN payload is not the property under test).
usage: u6-extra-types.py GGUFPY LIBGGML_BASE GGUF_DEQUANT WORKDIR   exit 0 only when every case is equal."""
import ctypes, os, subprocess, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFWriter, GGMLQuantizationType as Q

class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]
lib = ctypes.CDLL(sys.argv[2])
lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
TOF = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
cand, work = sys.argv[3], sys.argv[4]
os.makedirs(work, exist_ok=True)
TYPES = [Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q8_0, Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K, Q.IQ2_XXS,
         Q.IQ1_S, Q.IQ1_M, Q.IQ4_NL, Q.IQ4_XS, Q.TQ1_0, Q.Q2_0]
rng = np.random.default_rng(7)

def reference(t, raw, ne0, nrows):
    if t == Q.F32:
        return raw.view(np.float32).copy()
    tr = lib.ggml_get_type_traits(int(t)).contents
    out = np.zeros(nrows * ne0, np.float32)
    rb = ne0 // tr.blck_size * tr.type_size
    buf = ctypes.create_string_buffer(raw.tobytes(), raw.size)
    for r in range(nrows):
        TOF(tr.to_float)(ctypes.addressof(buf) + r * rb, out.ctypes.data + r * ne0 * 4, ne0)
    return out

def fix_scales(t, raw, tr):  # clear bit 6 of the high byte of every fp16 field so no scale is inf/NaN
    blocks = raw.reshape(-1, tr.type_size)
    offs = {Q.Q4_0: [0], Q.Q4_1: [0, 2], Q.Q5_0: [0], Q.Q8_0: [0], Q.Q2_0: [0], Q.Q2_K: [80, 82], Q.Q3_K: [108],
            Q.Q4_K: [0, 2], Q.Q5_K: [0, 2], Q.Q6_K: [208], Q.IQ2_XXS: [0], Q.IQ1_S: [0], Q.IQ4_NL: [0],
            Q.IQ4_XS: [0], Q.TQ1_0: [52], Q.IQ1_M: [54]}.get(t, [])  # IQ1_M: f16 high byte = bits 8..15 of the scale = top nibbles of u16 2, 3
    for o in offs:
        blocks[:, o + 1] &= 0xBF
    if t in (Q.F16,):
        raw.reshape(-1, 2)[:, 1] &= 0xBF
    if t == Q.BF16:
        raw.reshape(-1, 2)[:, 1] &= 0xBF
    if t == Q.F32:
        raw.reshape(-1, 4)[:, 3] &= 0xBF

def case(name, t, ne0, nrows, row0, n):
    tr = lib.ggml_get_type_traits(int(t)).contents
    rb = ne0 // tr.blck_size * tr.type_size
    raw = rng.integers(0, 256, nrows * rb, dtype=np.uint8)
    fix_scales(t, raw, tr)
    path = os.path.join(work, "x.gguf")
    w = GGUFWriter(path, "llama")
    w.add_tensor("t", raw.reshape(nrows, rb), raw_dtype=t)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    outp = os.path.join(work, "x.f32")
    p = subprocess.run([cand, path, "t", str(row0), str(n), outp], capture_output=True, text=True)
    if p.returncode != 0:
        return f"{name}: exit {p.returncode} {p.stderr.strip()[-100:]}{p.stdout.strip()[-100:]}"
    got = np.fromfile(outp, np.uint8)
    ref = reference(t, raw, ne0, nrows).view(np.uint8).reshape(nrows, ne0 * 4)[row0:row0 + n].reshape(-1)
    if got.size != ref.size:
        return f"{name}: {got.size} bytes, want {ref.size}"
    if not (got == ref).all():
        bad = np.nonzero(got != ref)[0]
        return f"{name}: {len(bad)} bytes differ, first at float {bad[0] // 4}"
    return None

fails = []
for t in TYPES:
    tr = lib.ggml_get_type_traits(int(t)).contents
    ne0 = max(1024, int(tr.blck_size))
    for tag, (nrows, row0, n) in {"all": (8, 0, 8), "offset": (11, 3, 5)}.items():
        e = case(f"{tr.type_name.decode()} {tag}", t, ne0, nrows, row0, n)
        print(("FAIL " + e) if e else f"ok   {tr.type_name.decode()} {tag}", flush=True)
        if e:
            fails.append(e)
e = case("F32 20MB stream", Q.F32, 256, 20000, 7, 19990)
print(("FAIL " + e) if e else "ok   F32 20MB stream", flush=True)
if e:
    fails.append(e)
print(f"FAIL {len(fails)}" if fails else "PASS extra types")
sys.exit(1 if fails else 0)
