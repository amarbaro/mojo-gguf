# st_open DIR: open a safetensors checkpoint directory, print "ok N" or "error: ..." (tests/negative-gate.sh).
from std.sys import argv
from st.safetensors import Checkpoint


def main():
    try:
        var c = Checkpoint(String(argv()[1]))
        print("ok", len(c.tensors))
    except e:
        print("error:", e)
