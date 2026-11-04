# Builds only if the library's libc declarations do not clash with the standard library's own
# (FileHandle.seek declares lseek): a program that seeks a file and uses Gguf, Json and Shard must link.
# coexist GGUF SAFETENSORS (tests/negative-gate.sh)
from std.sys import argv
from ggufx import Gguf
from st.json import Json
from st.safetensors import Shard


def main() raises:
    var a = argv()
    with open(String(a[1]), "r") as f:
        _ = f.seek(4)
        _ = f.read_bytes(4)
    var g = Gguf(String(a[1]))
    var j = Json(String(a[1]))
    var s = Shard(String(a[2]))
    print("coexist: ok", len(g.tensors), s.size > 0, j.kind(0) != 0)
