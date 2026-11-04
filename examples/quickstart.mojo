"""Open a GGUF file and print its summary, whether a key exists, and the tensor table. usage: quickstart FILE.gguf"""
from std.sys import argv
from ggufx import Gguf


def main() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("usage: quickstart FILE.gguf")
    var g = Gguf(String(a[1]))
    print(g.summary())  # kv K tensors T alignment A data_offset D
    print("general.architecture present:", g.find("general.architecture") >= 0)
    for i in range(len(g.tensors)):
        print(g.tensors[i].name, "type", g.tensors[i].ttype, "ne0", g.tensors[i].dims[0], "offset", g.tensors[i].offset)
