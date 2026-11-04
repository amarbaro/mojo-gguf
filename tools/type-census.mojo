# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
# Tensor-type census over GGUF files: reads only the header and tensor table (no tensor data), prints one line
# per file and type (tensor count, bytes from offset gaps) and a total per type over all files. Type names follow
# the pinned llama.cpp ggml.h enum. Scope input for U6 (dequant: every type on disk).
# usage: type-census FILE.gguf...
from std.sys import argv
from ggufx import Gguf


def tname(t: Int) -> String:
    comptime N = ["F32", "F16", "Q4_0", "Q4_1", "?4", "?5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K",
        "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8",
        "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "?31", "?32", "?33", "TQ1_0", "TQ2_0", "?36", "?37", "?38",
        "MXFP4", "NVFP4", "Q1_0", "Q2_0"]
    return materialize[N]()[t] if t >= 0 and t < 43 else "?" + String(t)


def one(path: String, mut tot_n: List[Int], mut tot_b: List[Int], mut files: List[Int]) raises:
    var g = Gguf(path)
    var cnt = List[Int](length=43, fill=0)
    var byt = List[Int](length=43, fill=0)
    var nt = len(g.tensors)
    for i in range(nt):
        var nxt = g.fsize - g.data_offset
        for j in range(nt):
            if g.tensors[j].offset > g.tensors[i].offset and g.tensors[j].offset < nxt:
                nxt = g.tensors[j].offset
        var t = g.tensors[i].ttype if g.tensors[i].ttype < 43 else 42
        cnt[t] += 1
        byt[t] += nxt - g.tensors[i].offset
    var line = path + " :"
    for t in range(43):
        if cnt[t] > 0:
            line += " " + tname(t) + "=" + String(cnt[t]) + "/" + String(byt[t] // 1048576) + "MB"
            tot_n[t] += cnt[t]
            tot_b[t] += byt[t]
            files[t] += 1
    print(line)


def main() raises:
    var a = argv()
    var tot_n = List[Int](length=43, fill=0)
    var tot_b = List[Int](length=43, fill=0)
    var files = List[Int](length=43, fill=0)
    for fi in range(1, len(a)):
        try:
            one(String(a[fi]), tot_n, tot_b, files)
        except e:
            print("skip", a[fi], e)
    print("TOTAL type tensors files GB")
    for t in range(43):
        if tot_n[t] > 0:
            print("TOTAL", tname(t), tot_n[t], files[t], Float64(tot_b[t]) / 1073741824.0)
