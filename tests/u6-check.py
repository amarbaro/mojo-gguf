"""U6 driver: rows from every tensor type of the given GGUFs, dequantized by the pinned llama.cpp (libggml-base
`ggml_get_type_traits(t)->to_float`, ctypes) and by the candidate `gguf-dequant MODEL TENSOR ROW0 N OUT.f32`;
f32 bytes must be equal. Per type: the first and last tensor of that type in each file, rows 0, last and 4 seeded
random ones. Falsifier per type: one flipped byte in the row's first block must change the oracle's output for
that row (proves the comparison can fail). Prints `ok TYPE n` / `FAIL TYPE ...` and a final `PASS t/t` or
`FAIL k/t TYPES`; exit code is the verdict. usage: u6-check.py GGUFPY LIBGGML_BASE CANDIDATE WORKDIR FILE.gguf..."""
import ctypes, os, random, subprocess, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader

class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]

lib = ctypes.CDLL(sys.argv[2])
lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
TOF = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
cand, work, files = sys.argv[3], sys.argv[4], sys.argv[5:]
os.makedirs(work, exist_ok=True)

def oracle(t, raw, ne0):
    tr = lib.ggml_get_type_traits(t).contents
    out = np.zeros(ne0, np.float32)
    if t == 0:
        return np.frombuffer(raw, np.float32).copy()
    buf = ctypes.create_string_buffer(bytes(raw), len(raw))
    TOF(tr.to_float)(buf, out.ctypes.data, ne0)
    return out

res = {}
for f in files:
    r = GGUFReader(f)
    mm = np.memmap(f, np.uint8, "r")
    bytype = {}
    for ti in r.tensors:
        bytype.setdefault(int(ti.tensor_type), []).append(ti)
    for t, ts in sorted(bytype.items()):
        tr = lib.ggml_get_type_traits(t).contents
        name = tr.type_name.decode()
        for ti in {id(x): x for x in (ts[0], ts[-1])}.values():
            ne0 = int(ti.shape[0]); nrows = int(np.prod([int(x) for x in ti.shape[1:]])) if len(ti.shape) > 1 else 1
            rb = ne0 // tr.blck_size * tr.type_size
            rng = random.Random(f"{os.path.basename(f)}:{ti.name}")
            rows = sorted({0, nrows - 1, *[rng.randrange(nrows) for _ in range(4)]})
            for row in rows:
                raw = mm[ti.data_offset + row * rb: ti.data_offset + (row + 1) * rb]
                ref = oracle(t, raw, ne0)
                outp = os.path.join(work, "got.f32")
                p = subprocess.run([cand, f, ti.name, str(row), "1", outp], capture_output=True, text=True)
                got = np.fromfile(outp, np.uint8) if p.returncode == 0 and os.path.exists(outp) else None
                ok = got is not None and got.tobytes() == ref.tobytes()
                if t != 0:
                    bad = bytearray(raw); bad[min(len(bad) - 1, tr.type_size // 2)] ^= 0x5A
                    ok_f = oracle(t, bytes(bad), ne0).tobytes() != ref.tobytes()
                else:
                    ok_f = True
                k = res.setdefault(name, [0, 0, []])
                k[0] += 1
                if ok and ok_f:
                    k[1] += 1
                elif len(k[2]) < 3:
                    why = "falsifier did not move" if not ok_f else (f"exit {p.returncode}: {p.stderr.strip()[-120:]}" if got is None else "bytes differ")
                    k[2].append(f"{os.path.basename(f)} {ti.name} row {row}: {why}")
        print(f"     {os.path.basename(f)}: types {' '.join(lib.ggml_get_type_traits(t).contents.type_name.decode() for t in sorted(bytype))}", flush=True)
bad = [k for k, v in res.items() if v[1] != v[0]]
for k, v in sorted(res.items()):
    print(f"{'ok  ' if v[1] == v[0] else 'FAIL'} {k} {v[1]}/{v[0]}" + "".join("\n     " + e for e in v[2]))
print(f"PASS {len(res)}/{len(res)} types" if not bad else f"FAIL {len(bad)}/{len(res)} {' '.join(sorted(bad))}")
sys.exit(1 if bad else 0)
