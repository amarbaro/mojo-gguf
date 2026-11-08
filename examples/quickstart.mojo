"""Open a GGUF file and print its summary, whether a key exists, and the tensor table.

usage: quickstart [FILE.gguf]   (no file: writes and reads a one-key GGUF v3 at /tmp/quickstart.gguf).
"""
from std.sys import argv
from ggufx import Gguf


def main() raises:
    var a = argv()
    var path = String(a[1]) if len(a) > 1 else _tiny("/tmp/quickstart.gguf")
    var g = Gguf(path)
    print(g.summary())  # kv K tensors T alignment A data_offset D
    print("general.architecture present:", g.find("general.architecture") >= 0)
    for i in range(len(g.tensors)):
        print(g.tensors[i].name, "type", g.tensors[i].ttype, "ne0", g.tensors[i].dims[0], "offset", g.tensors[i].offset)


def _le(mut b: List[UInt8], v: UInt64, n: Int):
    for i in range(n):
        b.append(UInt8((v >> UInt64(8 * i)) & 0xFF))


def _str(mut b: List[UInt8], s: String):
    _le(b, UInt64(s.byte_length()), 8)
    for c in s.as_bytes():
        b.append(c)


def _tiny(path: String) raises -> String:
    """GGUF v3, 0 tensors, 1 key: general.architecture = "llama"."""
    var b = List[UInt8]()
    for c in "GGUF".as_bytes():  # magic: 4 raw bytes
        b.append(c)
    _le(b, 3, 4)
    _le(b, 0, 8)
    _le(b, 1, 8)
    _str(b, "general.architecture")
    _le(b, 8, 4)  # value type 8 = string
    _str(b, "llama")
    with open(path, "w") as f:
        f.write_bytes(Span(b))
    return path
