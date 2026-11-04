"""Gguf-rewrite IN OUT [--name STR] [--remove KEY]...: parse a GGUF, serialize it again."""
from std.sys import argv, exit
from gguf import Gguf
from gguf.model import refuse_same_file

def run() raises:
    var a = argv()
    if len(a) < 3:
        raise Error("usage: gguf-rewrite IN OUT [--name STR] [--remove KEY]...")
    var name = String()
    var remove = List[String]()
    var i = 3
    while i < len(a):
        if (a[i] == "--name" or a[i] == "--remove") and i + 1 < len(a):
            if a[i] == "--name":
                name = String(a[i + 1])
            else:
                remove.append(String(a[i + 1]))
            i += 2
        else:
            raise Error("unknown argument " + String(a[i]))
    refuse_same_file(String(a[1]), String(a[2]))
    var g = Gguf(String(a[1]))
    print(g.summary())
    g.rewrite(String(a[2]), name, remove)

def main():
    try:
        run()
    except e:
        print("FAIL gguf-rewrite:", e)
        exit(1)
