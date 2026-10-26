"""Verifier CLI: u2-lookup CASES runs one map_tensor_name per line of CASES (tab-separated: arch, n_blocks, hf_name); prints one line each,
the GGUF name or an empty line when unmapped. Verifier for tests/u2-lookup-check.py, not a converter entry point."""
from std.sys import argv, exit
from gguf.tensor_lookup import map_tensor_name


def run() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("usage: u2-lookup CASES")
    var out = String()
    for line in open(String(a[1]), "r").read().split("\n"):
        if line.byte_length() == 0:
            continue
        var f = line.split("\t")
        out += map_tensor_name(String(f[0]), Int(String(f[1])), String(f[2])) + "\n"
    print(out, end="")


def main():
    try:
        run()
    except e:
        print("FAIL u2-lookup:", e)
        exit(1)
