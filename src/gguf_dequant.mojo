"""Gguf-dequant MODEL TENSOR ROW0 N OUT.f32: N rows of one tensor as little-endian f32, streamed in row chunks."""
from std.sys import argv, exit
from std.ffi import external_call, c_ssize_t
from ggufx.model import Gguf, open_fd, pread_into
from ggufx.dequant import quant_block, deq_rows

comptime CHUNK = 16 << 20  # f32 output bytes per round trip: bounds memory whatever N is (source is smaller)

def run() raises:
    var a = argv()
    if len(a) != 6:
        raise Error("usage: gguf-dequant MODEL TENSOR ROW0 N OUT.f32")
    var g = Gguf(String(a[1]))
    var ti = -1
    for k in range(len(g.tensors)):
        if g.tensors[k].name == String(a[2]):
            ti = k
            break
    if ti < 0:
        raise Error("no tensor " + String(a[2]))
    var row0 = Int(String(a[3]))
    var n = Int(String(a[4]))
    var info = quant_block(g.tensors[ti].ttype)
    if info[0] == 0:
        raise Error("tensor " + String(a[2]) + ": no dequantizer for ggml type " + String(g.tensors[ti].ttype))
    if len(g.tensors[ti].dims) == 0 or g.tensors[ti].dims[0] <= 0:
        raise Error("tensor " + String(a[2]) + " has no dimensions or a first dimension of 0")
    var ne0 = g.tensors[ti].dims[0]
    var nrows = 1
    for d in range(1, len(g.tensors[ti].dims)):
        nrows *= g.tensors[ti].dims[d]
    if row0 < 0 or n < 0 or row0 + n > nrows:
        raise Error("rows " + String(row0) + ".." + String(row0 + n) + " out of range: tensor has " + String(nrows))
    var rb = ne0 // info[0] * info[1]
    var step = max(1, CHUNK // (ne0 * 4))
    var src = List[UInt8](length=min(n, step) * rb, fill=0)
    var dst = List[Float32](length=min(n, step) * ne0, fill=0)
    var fd = open_fd(String(a[5]), 1 | 0o100 | 0o1000)  # O_WRONLY | O_CREAT | O_TRUNC
    var done = 0
    while done < n:
        var m = min(step, n - done)
        pread_into(g.fd, src, 0, m * rb, g.data_offset + g.tensors[ti].offset + (row0 + done) * rb)
        deq_rows(g.tensors[ti].ttype, src.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), ne0, m,
                 dst.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
        var left = m * ne0 * 4
        var off = done * ne0 * 4
        while left > 0:
            var w = external_call["pwrite", c_ssize_t](fd, dst.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_offset(m * ne0 * 4 - left), left, Int64(off))
            if w <= 0:
                raise Error("write failed")
            left -= Int(w)
            off += Int(w)
        done += m
    _ = external_call["close", Int32](fd)

def main():
    try:
        run()
    except e:
        print("FAIL gguf-dequant:", e)
        exit(1)
