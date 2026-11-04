"""Hf-update DIR: the pinned converter's pre-tokenizer fingerprint for a local HF tokenizer directory.

Prints `ntok N chkhsh H pre NAME`: the token count and sha256(str(ids)) of the converter's check text encoded with
DIR/tokenizer.json (see hfconv.hftok), and its tokenizer.ggml.pre name from the table generated out of the pinned
conversion/base.py by tools/gen-hf-table.py (`unknown` when the hash is not in it)."""
from std.sys import argv, exit
from ggufx.sha import Sha256, hexdigest
from hfconv.hftok import encode_chktxt
from hfconv.chk_table import pre_name

def run() raises:
    var a = argv()
    if len(a) != 2:
        raise Error("usage: hf-update DIR")
    var ids = encode_chktxt(String(a[1]))
    var s = String("[")
    for i in range(len(ids)):
        s += (", " if i > 0 else "") + String(ids[i])
    s += "]"
    var sha = Sha256()
    sha.update(List[UInt8](s.as_bytes()), s.byte_length())
    var h = hexdigest(sha.digest())
    print("ntok", len(ids), "chkhsh", h, "pre", pre_name(h))

def main():
    try:
        run()
    except e:
        print("FAIL hf-update:", e)
        exit(1)
