# repr_all: py_repr(chr(cp)) for every code point but the surrogates, one line each (tests/negative-gate.sh compares
# with Python's repr).
from ggufx.fmt import py_repr


def main():
    var out = String()
    for cp in range(0x110000):
        if cp >= 0xD800 and cp < 0xE000:
            continue
        out += py_repr(chr(cp)) + "\n"
        if out.byte_length() > 1 << 16:
            print(out, end="")
            out = String()
    print(out, end="")
