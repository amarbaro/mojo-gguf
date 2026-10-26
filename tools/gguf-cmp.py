"""Compare two GGUF files the way the gates need: whole-file bytes first, then KV, tensor table, per-tensor data.

    gguf-cmp.py REF.gguf GOT.gguf [--kv-only] [--max N]

Verifier, not product code: reads both files with the pinned gguf-py (LLAMACPP, default .work/llama.cpp).
Prints one line per difference (capped at --max, default 20), then a summary:
    PASS identical-bytes | PASS same-content (kv K, tensors T) | FAIL <counts>
Exit 0 only on PASS. A same-content PASS means equal KV values and equal tensor bytes with a different
file layout (alignment, order); gates that demand byte identity check the first form.
"""
import hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("LLAMACPP", os.path.join(ROOT, ".work", "llama.cpp")), "gguf-py"))
from gguf import GGUFReader  # noqa: E402


def kv(r):
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        out[k] = (tuple(int(t) for t in f.types), b"".join(bytes(f.parts[i]) for i in f.data))
    return out


def tensors(r):
    return {t.name: (t.tensor_type.name, tuple(int(x) for x in t.shape), hashlib.sha256(bytes(t.data)).hexdigest(), t.n_bytes) for t in r.tensors}


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 2:
        print(__doc__.strip().splitlines()[2].strip())
        sys.exit(2)
    kv_only = "--kv-only" in sys.argv
    cap = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 20
    ref, got = a
    if os.path.getsize(ref) == os.path.getsize(got):
        with open(ref, "rb") as f1, open(got, "rb") as f2:
            same = True
            while True:
                b1, b2 = f1.read(1 << 24), f2.read(1 << 24)
                if b1 != b2:
                    same = False
                    break
                if not b1:
                    break
        if same:
            print("PASS identical-bytes")
            return
    r1, r2 = GGUFReader(ref), GGUFReader(got)
    diffs = []
    k1, k2 = kv(r1), kv(r2)
    for k in sorted(set(k1) | set(k2)):
        if k not in k2:
            diffs.append(f"kv missing: {k}")
        elif k not in k1:
            diffs.append(f"kv extra: {k}")
        elif k1[k] != k2[k]:
            diffs.append(f"kv differs: {k} types {k1[k][0]} vs {k2[k][0]}, {len(k1[k][1])} vs {len(k2[k][1])} bytes")
    nkv = len(diffs)
    t1, t2 = ({}, {}) if kv_only else (tensors(r1), tensors(r2))
    for n in [t.name for t in r1.tensors] + [n for n in t2 if n not in t1]:
        if kv_only:
            break
        if n not in t2:
            diffs.append(f"tensor missing: {n}")
        elif n not in t1:
            diffs.append(f"tensor extra: {n}")
        elif t1[n] != t2[n]:
            (ty1, s1, h1, b1), (ty2, s2, h2, b2) = t1[n], t2[n]
            what = []
            if ty1 != ty2: what.append(f"type {ty1} vs {ty2}")
            if s1 != s2: what.append(f"shape {s1} vs {s2}")
            if h1 != h2: what.append(f"data ({b1} vs {b2} bytes)")
            diffs.append(f"tensor differs: {n} " + ", ".join(what))
    order = [t.name for t in r1.tensors] == [t.name for t in r2.tensors]
    for d in diffs[:cap]:
        print(d)
    if len(diffs) > cap:
        print(f"... {len(diffs) - cap} more")
    if diffs:
        print(f"FAIL kv {nkv} tensors {len(diffs) - nkv} of {len(t1)} (order {'same' if order else 'differs'})")
        sys.exit(1)
    print(f"PASS same-content (kv {len(k1)}, tensors {len(t1)}, order {'same' if order else 'differs'})")


main()
