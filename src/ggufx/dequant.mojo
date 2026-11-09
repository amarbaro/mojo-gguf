"""Dequantize rows of GGUF tensors to f32, bit-equal to llama.cpp's `to_float` (ggml-quants.c, pinned 1744c6bde).

Every row function reads `nb` blocks at `s` and writes `nb * block_len` floats at `y`; it allocates nothing, keeps
no state between blocks and touches only its arguments, so it can move into a kernel unchanged. Integer
arithmetic is done in Int and converted once, in the same order as the C reference, so every float op is the same
IEEE f32 op. `deq_rows` splits the rows over threads; codebooks are materialized once per thread chunk.
"""
from std.collections import Array
from std.memory import bitcast
from ggufx.par import parallelize
from ggufx.grids import IQ2XXS_GRID, IQ1S_GRID, KVALUES_IQ4NL

comptime Src = Pointer[UInt8, MutUntrackedOrigin]
comptime Dst = Pointer[Float32, MutUntrackedOrigin]

comptime T_F32 = 0
comptime T_F16 = 1
comptime T_Q4_0 = 2
comptime T_Q4_1 = 3
comptime T_Q5_0 = 6
comptime T_Q8_0 = 8
comptime T_Q2_K = 10
comptime T_Q3_K = 11
comptime T_Q4_K = 12
comptime T_Q5_K = 13
comptime T_Q6_K = 14
comptime T_IQ2_XXS = 16
comptime T_IQ1_S = 19
comptime T_IQ4_NL = 20
comptime T_IQ4_XS = 23
comptime T_IQ1_M = 29
comptime T_BF16 = 30
comptime T_TQ1_0 = 34
comptime T_Q2_0 = 42

def quant_block(t: Int) -> Tuple[Int, Int]:
    """(elements, bytes) of one block of ggml type t; (0, 0) when t has no dequantizer."""
    if t == T_F32:
        return (1, 4)
    if t == T_F16 or t == T_BF16:
        return (1, 2)
    if t == T_Q4_0 or t == T_IQ4_NL:
        return (32, 18)
    if t == T_Q4_1:
        return (32, 20)
    if t == T_Q5_0:
        return (32, 22)
    if t == T_Q8_0:
        return (32, 34)
    if t == T_Q2_0:
        return (64, 18)
    if t == T_Q2_K:
        return (256, 84)
    if t == T_Q3_K:
        return (256, 110)
    if t == T_Q4_K:
        return (256, 144)
    if t == T_Q5_K:
        return (256, 176)
    if t == T_Q6_K:
        return (256, 210)
    if t == T_IQ2_XXS:
        return (256, 66)
    if t == T_IQ1_S:
        return (256, 50)
    if t == T_IQ1_M:
        return (256, 56)
    if t == T_IQ4_XS:
        return (256, 136)
    if t == T_TQ1_0:
        return (256, 54)
    return (0, 0)

@always_inline
def u8(p: Src, i: Int) -> Int:
    return Int(p[unsafe_offset=i])

@always_inline
def i8(p: Src, i: Int) -> Int:
    var v = u8(p, i)
    return v - 256 if v > 127 else v

@always_inline
def u16(p: Src, i: Int) -> Int:
    return u8(p, i) | (u8(p, i + 1) << 8)

@always_inline
def u32(p: Src, i: Int) -> Int:
    return u16(p, i) | (u16(p, i + 2) << 16)

@always_inline
def f16(p: Src, i: Int) -> Float32:
    return Float32(bitcast[DType.float16](UInt16(u16(p, i))))

@always_inline
def gbyte(g: UInt64, j: Int) -> Int:
    return Int((g >> UInt64(8 * j)) & 0xFF)

@always_inline
def sgn(neg: Bool) -> Float32:
    return Float32(-1.0) if neg else Float32(1.0)

@always_inline
def k4_scale(p: Src, base: Int, j: Int) -> Int:
    """6-bit scale j (0..7) of the packed 12-byte scale/min block of Q4_K and Q5_K."""
    if j < 4:
        return u8(p, base + j) & 63
    return (u8(p, base + j + 4) & 15) | ((u8(p, base + j - 4) >> 6) << 4)

@always_inline
def k4_min(p: Src, base: Int, j: Int) -> Int:
    if j < 4:
        return u8(p, base + j + 4) & 63
    return (u8(p, base + j + 4) >> 4) | ((u8(p, base + j) >> 6) << 4)

def deq_f32(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        y[unsafe_offset=i] = bitcast[DType.float32](UInt32(u32(s, 4 * i)))

def deq_f16(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        y[unsafe_offset=i] = f16(s, 2 * i)

def deq_bf16(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        y[unsafe_offset=i] = bitcast[DType.float32](UInt32(u16(s, 2 * i) << 16))

def deq_q4_0(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 18
        var d = f16(s, b)
        for j in range(16):
            var q = u8(s, b + 2 + j)
            y[unsafe_offset=i * 32 + j] = Float32((q & 15) - 8) * d
            y[unsafe_offset=i * 32 + j + 16] = Float32((q >> 4) - 8) * d

def deq_q4_1(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 20
        var d = f16(s, b)
        var m = f16(s, b + 2)
        for j in range(16):
            var q = u8(s, b + 4 + j)
            y[unsafe_offset=i * 32 + j] = Float32(q & 15) * d + m
            y[unsafe_offset=i * 32 + j + 16] = Float32(q >> 4) * d + m

def deq_q5_0(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 22
        var d = f16(s, b)
        var qh = u32(s, b + 2)
        for j in range(16):
            var q = u8(s, b + 6 + j)
            var h0 = ((qh >> j) << 4) & 0x10
            var h1 = (qh >> (j + 12)) & 0x10
            y[unsafe_offset=i * 32 + j] = Float32(((q & 15) | h0) - 16) * d
            y[unsafe_offset=i * 32 + j + 16] = Float32(((q >> 4) | h1) - 16) * d

def deq_q8_0(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 34
        var d = f16(s, b)
        for j in range(32):
            y[unsafe_offset=i * 32 + j] = Float32(i8(s, b + 2 + j)) * d

def deq_q2_0(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 18
        var d = f16(s, b)
        for j in range(64):
            var q = (u8(s, b + 2 + j // 4) >> ((j % 4) * 2)) & 3
            y[unsafe_offset=i * 64 + j] = Float32(q - 1) * d

def deq_q2_k(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 84
        var d = f16(s, b + 80)
        var mn = f16(s, b + 82)
        var qo = b + 16
        var yo = i * 256
        var is_ = 0
        for _ in range(2):
            var shift = 0
            for _ in range(4):
                for h in range(2):
                    var sc = u8(s, b + is_)
                    is_ += 1
                    var dl = d * Float32(sc & 15)
                    var ml = mn * Float32(sc >> 4)
                    for l in range(16):
                        y[unsafe_offset=yo] = dl * Float32((u8(s, qo + l + 16 * h) >> shift) & 3) - ml
                        yo += 1
                shift += 2
            qo += 32

def deq_q3_k(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 110
        var d = f16(s, b + 108)
        var qo = b + 32
        var yo = i * 256
        var is_ = 0
        var m = 1
        for _ in range(2):
            var shift = 0
            for _ in range(4):
                for h in range(2):
                    var sc = ((u8(s, b + 96 + is_ % 8) >> (4 * (is_ // 8))) & 15) | (((u8(s, b + 104 + is_ % 4) >> (2 * (is_ // 4))) & 3) << 4)
                    is_ += 1
                    var dl = d * Float32(sc - 32)
                    for l in range(16):
                        var hi = 0 if (u8(s, b + l + 16 * h) & m) != 0 else 4
                        y[unsafe_offset=yo] = dl * Float32(((u8(s, qo + l + 16 * h) >> shift) & 3) - hi)
                        yo += 1
                shift += 2
                m <<= 1
            qo += 32

def deq_q4_k(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 144
        var d = f16(s, b)
        var mn = f16(s, b + 2)
        var yo = i * 256
        for jj in range(4):
            var d1 = d * Float32(k4_scale(s, b + 4, 2 * jj))
            var m1 = mn * Float32(k4_min(s, b + 4, 2 * jj))
            var d2 = d * Float32(k4_scale(s, b + 4, 2 * jj + 1))
            var m2 = mn * Float32(k4_min(s, b + 4, 2 * jj + 1))
            var qo = b + 16 + 32 * jj
            for l in range(32):
                y[unsafe_offset=yo + l] = d1 * Float32(u8(s, qo + l) & 15) - m1
            for l in range(32):
                y[unsafe_offset=yo + 32 + l] = d2 * Float32(u8(s, qo + l) >> 4) - m2
            yo += 64

def deq_q5_k(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 176
        var d = f16(s, b)
        var mn = f16(s, b + 2)
        var yo = i * 256
        for jj in range(4):
            var d1 = d * Float32(k4_scale(s, b + 4, 2 * jj))
            var m1 = mn * Float32(k4_min(s, b + 4, 2 * jj))
            var d2 = d * Float32(k4_scale(s, b + 4, 2 * jj + 1))
            var m2 = mn * Float32(k4_min(s, b + 4, 2 * jj + 1))
            var qo = b + 48 + 32 * jj
            var u1 = 1 << (2 * jj)
            var u2 = 2 << (2 * jj)
            for l in range(32):
                var hb = 16 if (u8(s, b + 16 + l) & u1) != 0 else 0
                y[unsafe_offset=yo + l] = d1 * Float32((u8(s, qo + l) & 15) + hb) - m1
            for l in range(32):
                var hb = 16 if (u8(s, b + 16 + l) & u2) != 0 else 0
                y[unsafe_offset=yo + 32 + l] = d2 * Float32((u8(s, qo + l) >> 4) + hb) - m2
            yo += 64

def deq_q6_k(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 210
        var d = f16(s, b + 208)
        for n in range(2):
            var ql = b + 64 * n
            var qh = b + 128 + 32 * n
            var sc = b + 192 + 8 * n
            var yo = i * 256 + 128 * n
            for l in range(32):
                var is_ = l // 16
                var lo0 = u8(s, ql + l)
                var lo1 = u8(s, ql + l + 32)
                var h = u8(s, qh + l)
                var q1 = ((lo0 & 15) | (((h >> 0) & 3) << 4)) - 32
                var q2 = ((lo1 & 15) | (((h >> 2) & 3) << 4)) - 32
                var q3 = ((lo0 >> 4) | (((h >> 4) & 3) << 4)) - 32
                var q4 = ((lo1 >> 4) | (((h >> 6) & 3) << 4)) - 32
                y[unsafe_offset=yo + l] = d * Float32(i8(s, sc + is_)) * Float32(q1)
                y[unsafe_offset=yo + l + 32] = d * Float32(i8(s, sc + is_ + 2)) * Float32(q2)
                y[unsafe_offset=yo + l + 64] = d * Float32(i8(s, sc + is_ + 4)) * Float32(q3)
                y[unsafe_offset=yo + l + 96] = d * Float32(i8(s, sc + is_ + 6)) * Float32(q4)

def deq_tq1_0(s: Src, y: Dst, nb: Int):
    for i in range(nb):
        var b = i * 54
        var d = f16(s, b + 52)
        var yo = i * 256
        for part in range(2):  # qs[0:32] in 32-byte groups, qs[32:48] in 16-byte groups, then the 4 qh bytes
            var j0 = 32 * part
            var w = 32 if part == 0 else 16
            var p3 = 1
            for _ in range(5):
                for m in range(w):
                    var xi = (((u8(s, b + j0 + m) * p3) & 255) * 3) >> 8
                    y[unsafe_offset=yo] = Float32(xi - 1) * d
                    yo += 1
                p3 *= 3
        var p3 = 1
        for _ in range(4):
            for j in range(4):
                var xi = (((u8(s, b + 48 + j) * p3) & 255) * 3) >> 8
                y[unsafe_offset=yo] = Float32(xi - 1) * d
                yo += 1
            p3 *= 3

def deq_iq2_xxs(s: Src, y: Dst, nb: Int, grid: Array[UInt64, 256]):
    for i in range(nb):
        var b = i * 66
        var d = f16(s, b)
        var yo = i * 256
        for ib in range(8):
            var a0 = u32(s, b + 2 + 8 * ib)
            var a1 = u32(s, b + 6 + 8 * ib)
            var db = d * (Float32(0.5) + Float32(a1 >> 28)) * Float32(0.25)
            for l in range(4):
                var g = grid[(a0 >> (8 * l)) & 255]
                var sg = (a1 >> (7 * l)) & 127
                var par = sg ^ (sg >> 4)  # ksigns_iq2xs[sg]: bit 7 makes the popcount even
                par ^= par >> 2
                par ^= par >> 1
                var signs = sg | ((par & 1) << 7)
                for j in range(8):
                    y[unsafe_offset=yo] = db * Float32(gbyte(g, j)) * sgn(((signs >> j) & 1) != 0)
                    yo += 1

def deq_iq1_s(s: Src, y: Dst, nb: Int, grid: Array[UInt64, 2048]):
    for i in range(nb):
        var b = i * 50
        var d = f16(s, b)
        var yo = i * 256
        for ib in range(8):
            var qh = u16(s, b + 34 + 2 * ib)
            var dl = d * Float32(2 * ((qh >> 12) & 7) + 1)
            var delta = Float32(-0.125) if (qh & 0x8000) != 0 else Float32(0.125)
            for l in range(4):
                var g = grid[u8(s, b + 2 + 4 * ib + l) | (((qh >> (3 * l)) & 7) << 8)]
                for j in range(8):
                    var gi = gbyte(g, j)
                    y[unsafe_offset=yo] = dl * (Float32(gi - 256 if gi > 127 else gi) + delta)
                    yo += 1

def deq_iq1_m(s: Src, y: Dst, nb: Int, grid: Array[UInt64, 2048]):
    for i in range(nb):
        var b = i * 56
        var sc0 = u16(s, b + 48)
        var sc1 = u16(s, b + 50)
        var sc2 = u16(s, b + 52)
        var sc3 = u16(s, b + 54)
        var sbits = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0) | ((sc2 >> 4) & 0x0F00) | (sc3 & 0xF000)
        var d = Float32(bitcast[DType.float16](UInt16(sbits)))
        var yo = i * 256
        for ib in range(8):
            var sc = u16(s, b + 48 + 2 * (ib // 2))
            var dl1 = d * Float32(2 * ((sc >> (6 * (ib % 2))) & 7) + 1)
            var dl2 = d * Float32(2 * ((sc >> (6 * (ib % 2) + 3)) & 7) + 1)
            var qh0 = u8(s, b + 32 + 2 * ib)
            var qh1 = u8(s, b + 33 + 2 * ib)
            for l in range(4):
                var qh = qh0 if l < 2 else qh1
                var hi = (qh << 8) & 0x700 if l % 2 == 0 else (qh << 4) & 0x700
                var neg = (qh & 0x08) != 0 if l % 2 == 0 else (qh & 0x80) != 0
                var g = grid[u8(s, b + 4 * ib + l) | hi]
                var dl = dl1 if l < 2 else dl2
                var delta = Float32(-0.125) if neg else Float32(0.125)
                for j in range(8):
                    var gi = gbyte(g, j)
                    y[unsafe_offset=yo] = dl * (Float32(gi - 256 if gi > 127 else gi) + delta)
                    yo += 1

def deq_iq4_nl(s: Src, y: Dst, nb: Int, kv: Array[Int32, 16]):
    for i in range(nb):
        var b = i * 18
        var d = f16(s, b)
        for j in range(16):
            var q = u8(s, b + 2 + j)
            y[unsafe_offset=i * 32 + j] = d * Float32(kv[q & 15])
            y[unsafe_offset=i * 32 + j + 16] = d * Float32(kv[q >> 4])

def deq_iq4_xs(s: Src, y: Dst, nb: Int, kv: Array[Int32, 16]):
    for i in range(nb):
        var b = i * 136
        var d = f16(s, b)
        var sh = u16(s, b + 2)
        for ib in range(8):
            var ls = ((u8(s, b + 4 + ib // 2) >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
            var dl = d * Float32(ls - 32)
            for j in range(16):
                var q = u8(s, b + 8 + 16 * ib + j)
                y[unsafe_offset=i * 256 + 32 * ib + j] = dl * Float32(kv[q & 15])
                y[unsafe_offset=i * 256 + 32 * ib + j + 16] = dl * Float32(kv[q >> 4])

def deq_span(t: Int, s: Src, y: Dst, n: Int, nb: Int, rb: Int, ne0: Int):
    """n consecutive rows of type t; codebooks are materialized once per call."""
    if t == T_IQ2_XXS:
        var g = materialize[IQ2XXS_GRID]()
        for r in range(n):
            deq_iq2_xxs(s.unsafe_offset(r * rb), y.unsafe_offset(r * ne0), nb, g)
    elif t == T_IQ1_S:
        var g = materialize[IQ1S_GRID]()
        for r in range(n):
            deq_iq1_s(s.unsafe_offset(r * rb), y.unsafe_offset(r * ne0), nb, g)
    elif t == T_IQ1_M:
        var g = materialize[IQ1S_GRID]()
        for r in range(n):
            deq_iq1_m(s.unsafe_offset(r * rb), y.unsafe_offset(r * ne0), nb, g)
    elif t == T_IQ4_NL or t == T_IQ4_XS:
        var kv = materialize[KVALUES_IQ4NL]()
        for r in range(n):
            if t == T_IQ4_NL:
                deq_iq4_nl(s.unsafe_offset(r * rb), y.unsafe_offset(r * ne0), nb, kv)
            else:
                deq_iq4_xs(s.unsafe_offset(r * rb), y.unsafe_offset(r * ne0), nb, kv)
    else:
        for r in range(n):
            var sr = s.unsafe_offset(r * rb)
            var yr = y.unsafe_offset(r * ne0)
            if t == T_F32:
                deq_f32(sr, yr, nb)
            elif t == T_F16:
                deq_f16(sr, yr, nb)
            elif t == T_BF16:
                deq_bf16(sr, yr, nb)
            elif t == T_Q4_0:
                deq_q4_0(sr, yr, nb)
            elif t == T_Q4_1:
                deq_q4_1(sr, yr, nb)
            elif t == T_Q5_0:
                deq_q5_0(sr, yr, nb)
            elif t == T_Q8_0:
                deq_q8_0(sr, yr, nb)
            elif t == T_Q2_0:
                deq_q2_0(sr, yr, nb)
            elif t == T_Q2_K:
                deq_q2_k(sr, yr, nb)
            elif t == T_Q3_K:
                deq_q3_k(sr, yr, nb)
            elif t == T_Q4_K:
                deq_q4_k(sr, yr, nb)
            elif t == T_Q5_K:
                deq_q5_k(sr, yr, nb)
            elif t == T_Q6_K:
                deq_q6_k(sr, yr, nb)
            else:
                deq_tq1_0(sr, yr, nb)

@fieldwise_init
struct _Span(Copyable):
    var src: Src
    var dst: Dst
    var t: Int
    var nb: Int
    var rb: Int
    var ne0: Int
    var nrows: Int
    var items: Int

def _deq_task(ctx: Int, w: Int):
    var c = Pointer[_Span, MutUntrackedOrigin](unsafe_from_address=ctx)
    var r0 = c[].nrows * w // c[].items
    var r1 = c[].nrows * (w + 1) // c[].items
    deq_span(c[].t, c[].src.unsafe_offset(r0 * c[].rb), c[].dst.unsafe_offset(r0 * c[].ne0), r1 - r0, c[].nb, c[].rb, c[].ne0)

def deq_rows(t: Int, src: Src, ne0: Int, nrows: Int, dst: Dst) raises:
    """Dequantize nrows rows of ne0 elements of ggml type t from src (contiguous rows) into dst (nrows * ne0 f32)."""
    var info = quant_block(t)
    if info[0] == 0:
        raise Error("no dequantizer for ggml type " + String(t))
    if ne0 % info[0] != 0:
        raise Error("row of " + String(ne0) + " elements is not a multiple of the type's block of " + String(info[0]))
    var nb = ne0 // info[0]
    var rb = nb * info[1]
    var items = min(nrows, 256)
    if items <= 0:
        return
    var c: List[_Span] = [_Span(src, dst, t, nb, rb, ne0, nrows, items)]  # heap ctx: a stack slot can die before the threads read it
    parallelize(_deq_task, Int(c.unsafe_ptr()), items)
    _ = len(c)
