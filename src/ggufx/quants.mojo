"""Row quantizers, ported from the `quantize_row_*_ref` paths of ggml-quants.c that llama-quantize takes
without an importance matrix. Float semantics are the C's: single precision, no contraction, fp16 by
round-to-nearest-even, `nearest_int` by the 1.5 * 2^23 trick, `roundf` half away from zero."""
from std.math import sqrt
from std.collections import Array
from std.os import abort
from std.sys.intrinsics import llvm_intrinsic
from std.memory import bitcast

comptime FP = Pointer[Float32, MutAnyOrigin]
comptime BP = Pointer[UInt8, MutAnyOrigin]

comptime T_F32 = 0
comptime T_F16 = 1
comptime T_Q4_0 = 2
comptime T_Q5_0 = 6
comptime T_Q8_0 = 8
comptime T_Q2_K = 10
comptime T_Q4_K = 12
comptime T_Q6_K = 14
comptime T_BF16 = 30
comptime T_TQ1_0 = 34
comptime T_TQ2_0 = 35
comptime T_Q2_0 = 42
comptime QK_K = 256
comptime EPS = Float32(1e-15)

def blck(t: Int) -> Int:
    """Elements per block; 0 for a type this module does not know."""
    if t == T_F32 or t == T_F16 or t == T_BF16:
        return 1
    if t == T_Q4_0 or t == T_Q5_0 or t == T_Q8_0:
        return 32
    if t == T_Q2_0:
        return 64
    if t == T_Q2_K or t == T_Q4_K or t == T_Q6_K or t == T_TQ1_0 or t == T_TQ2_0:
        return QK_K
    return 0

def tsize(t: Int) -> Int:
    """Bytes per block."""
    if t == T_F32:
        return 4
    if t == T_F16 or t == T_BF16:
        return 2
    if t == T_Q4_0 or t == T_Q2_0:
        return 18
    if t == T_Q5_0:
        return 22
    if t == T_Q8_0:
        return 34
    if t == T_Q2_K:
        return 84
    if t == T_Q4_K:
        return 144
    if t == T_Q6_K:
        return 210
    if t == T_TQ1_0:
        return 54
    if t == T_TQ2_0:
        return 66
    return 0

def row_size(t: Int, n: Int) -> Int:
    return n // blck(t) * tsize(t)

@always_inline
def fence(x: Float32) -> Float32:
    """Keeps the producer of x from fusing into a following add: the oracle's C has no FMA, Mojo
    contracts `a*b + c` by default, and the gate's build line has no flag to turn that off."""
    return llvm_intrinsic["llvm.arithmetic.fence", Float32](x)

@always_inline
def m(a: Float32, b: Float32) -> Float32:
    return fence(a * b)

@always_inline
def nint(f: Float32) -> Int:
    return (Int(bitcast[DType.int32](fence(f) + 12582912.0)) & 0x7FFFFF) - 0x400000

@always_inline
def roundf(f: Float32) -> Int:
    var x = fence(f)
    var t = Int(x)
    var d = x - Float32(t)
    if d >= 0.5:
        return t + 1
    if d <= -0.5:
        return t - 1
    return t

@always_inline
def f16_bits(x: Float32) -> Int:
    return Int(bitcast[DType.uint16](Float16(x)))

@always_inline
def put16(y: BP, at: Int, v: Int):
    y[unsafe_offset=at] = UInt8(v & 0xFF)
    y[unsafe_offset=at + 1] = UInt8((v >> 8) & 0xFF)

@always_inline
def vfence[W: Int](x: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    return llvm_intrinsic["llvm.arithmetic.fence", SIMD[DType.float32, W]](x)

@always_inline
def vm[W: Int](a: SIMD[DType.float32, W], b: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
    return vfence(a * b)

@always_inline
def vnint[W: Int](f: SIMD[DType.float32, W]) -> SIMD[DType.int32, W]:
    return (bitcast[DType.int32, W](vfence(f) + 12582912.0) & 0x7FFFFF) - 0x400000

# The K-quant fits are sequential float recurrences over one sub-block, and the sub-blocks of a
# super-block are independent: W of them run in the W lanes of one SIMD vector, element i of every
# sub-block in `xt[i]`. Each lane performs exactly the scalar C's operations in its order, so the
# results are bit for bit the C's; data-dependent `if`s become lane masks.

def qkx2_lanes[W: Int, N: Int, NMAX: Int, NSTEP: Int, USE_MAD: Bool](
        xt: Array[SIMD[DType.float32, W], N], wt: Array[SIMD[DType.float32, W], N],
        mut Lt: Array[SIMD[DType.int32, W], N], mut the_min: SIMD[DType.float32, W],
        rmin: Float32, rdelta: Float32) -> SIMD[DType.float32, W]:
    """make_qkx2_quants over W sub-blocks of N weighted elements: (scale, min, levels 0..NMAX)."""
    comptime F = SIMD[DType.float32, W]
    comptime I = SIMD[DType.int32, W]
    var mn = xt[0]
    var mx = xt[0]
    var sum_w = wt[0]
    var sum_x = vm(sum_w, xt[0])
    for i in range(1, N):
        mn = xt[i].lt(mn).select(xt[i], mn)
        mx = xt[i].gt(mx).select(xt[i], mx)
        sum_w += wt[i]
        sum_x += vm(wt[i], xt[i])
    mn = mn.gt(F(0)).select(F(0), mn)
    var flat = mx.eq(mn)
    var iscale = F(Float32(NMAX)) / (mx - mn)
    var scale = F(1) / iscale
    var best = F(0)
    for i in range(N):
        var l = vnint(vm(iscale, xt[i] - mn)).clamp(I(0), I(NMAX))
        Lt[i] = l
        var diff = vm(scale, l.cast[DType.float32]()) + mn - xt[i]
        comptime if USE_MAD:
            diff = abs(diff)
        else:
            diff = vm(diff, diff)
        best += vm(wt[i], diff)
    var Laux = Array[I, N](uninitialized=True)
    for s in range(NSTEP + 1):
        iscale = F(rmin + m(rdelta, Float32(s)) + Float32(NMAX)) / (mx - mn)
        var sum_l = F(0)
        var sum_l2 = F(0)
        var sum_xl = F(0)
        for i in range(N):
            var l = vnint(vm(iscale, xt[i] - mn)).clamp(I(0), I(NMAX))
            Laux[i] = l
            var lf = l.cast[DType.float32]()
            var wl = vm(wt[i], lf)
            sum_l += wl
            sum_l2 += vm(wl, lf)
            sum_xl += vm(wl, xt[i])
        var D = vm(sum_w, sum_l2) - vm(sum_l, sum_l)
        var this_scale = (vm(sum_w, sum_xl) - vm(sum_x, sum_l)) / D
        var this_min = (vm(sum_l2, sum_x) - vm(sum_l, sum_xl)) / D
        var pos = this_min.gt(F(0))
        this_min = pos.select(F(0), this_min)
        this_scale = pos.select(sum_xl / sum_l2, this_scale)
        var cur = F(0)
        for i in range(N):
            var diff = vm(this_scale, Laux[i].cast[DType.float32]()) + this_min - xt[i]
            comptime if USE_MAD:
                diff = abs(diff)
            else:
                diff = vm(diff, diff)
            cur += vm(wt[i], diff)
        var upd = D.gt(F(0)) & cur.lt(best) & ~flat
        for i in range(N):
            Lt[i] = upd.select(Laux[i], Lt[i])
        best = upd.select(cur, best)
        scale = upd.select(this_scale, scale)
        mn = upd.select(this_min, mn)
    for i in range(N):
        Lt[i] = flat.select(I(0), Lt[i])
    the_min = -mn
    return flat.select(F(0), scale)

def qx_lanes[W: Int, N: Int, NMAX: Int](xt: Array[SIMD[DType.float32, W], N],
                                       mut Lt: Array[SIMD[DType.int32, W], N]) -> SIMD[DType.float32, W]:
    """make_qx_quants (rmse type 1, no importance weights) over W sub-blocks: (scale, levels 0..2*NMAX-1)."""
    comptime F = SIMD[DType.float32, W]
    comptime I = SIMD[DType.int32, W]
    var amax = F(0)
    var mx = F(0)
    for i in range(N):
        var ax = abs(xt[i])
        var gt = ax.gt(amax)
        amax = gt.select(ax, amax)
        mx = gt.select(xt[i], mx)
    var zero = amax.lt(F(EPS))
    var iscale = F(Float32(-NMAX)) / mx
    var sumlx = F(0)
    var suml2 = F(0)
    for i in range(N):
        var l = vnint(vm(iscale, xt[i])).clamp(I(-NMAX), I(NMAX - 1))
        Lt[i] = l + I(NMAX)
        var lf = l.cast[DType.float32]()
        var w = vm(xt[i], xt[i])
        sumlx += vm(vm(w, xt[i]), lf)
        suml2 += vm(vm(w, lf), lf)
    var scale = suml2.ne(F(0)).select(sumlx / suml2, F(0))
    var best = vm(scale, sumlx)
    var Laux = Array[I, N](uninitialized=True)
    for k in range(-9, 10):
        if k == 0:
            continue
        iscale = F(-(Float32(NMAX) + m(0.1, Float32(k)))) / mx
        sumlx = F(0)
        suml2 = F(0)
        for i in range(N):
            var l = vnint(vm(iscale, xt[i])).clamp(I(-NMAX), I(NMAX - 1))
            Laux[i] = l
            var lf = l.cast[DType.float32]()
            var w = vm(xt[i], xt[i])
            sumlx += vm(vm(w, xt[i]), lf)
            suml2 += vm(vm(w, lf), lf)
        var take = suml2.gt(F(0)) & vm(sumlx, sumlx).gt(vm(best, suml2)) & ~zero
        for i in range(N):
            Lt[i] = take.select(Laux[i] + I(NMAX), Lt[i])
        var ns = sumlx / suml2
        scale = take.select(ns, scale)
        best = take.select(vm(ns, sumlx), best)
    for i in range(N):
        Lt[i] = zero.select(I(0), Lt[i])
    return zero.select(F(0), scale)

def q4_0_row(x: FP, y: BP, n: Int, hi: Int):
    """Q4_0 (hi 8, 4-bit) and Q5_0 (hi 16, 5-bit with a qh word): the signed-max scale."""
    var bytes = 18 if hi == 8 else 22
    for i in range(n // 32):
        var amax = Float32(0)
        var mx = Float32(0)
        for j in range(32):
            var v = x[unsafe_offset=32 * i + j]
            if amax < abs(v):
                amax = abs(v)
                mx = v
        var d = mx / Float32(-hi)
        var id = 1.0 / d if d != 0 else Float32(0)
        var o = bytes * i
        put16(y, o, f16_bits(d))
        var qh = 0
        var cap = 2 * hi - 1
        var off = Float32(hi) + 0.5
        for j in range(16):
            var xi0 = min(cap, Int(m(x[unsafe_offset=32 * i + j], id) + off))
            var xi1 = min(cap, Int(m(x[unsafe_offset=32 * i + 16 + j], id) + off))
            y[unsafe_offset=o + bytes - 16 + j] = UInt8((xi0 & 0xF) | ((xi1 & 0xF) << 4))
            qh |= ((xi0 & 0x10) >> 4) << j
            qh |= ((xi1 & 0x10) >> 4) << (j + 16)
        if hi == 16:
            for k in range(4):
                y[unsafe_offset=o + 2 + k] = UInt8((qh >> (8 * k)) & 0xFF)

def q8_0_row(x: FP, y: BP, n: Int):
    for i in range(n // 32):
        var amax = Float32(0)
        for j in range(32):
            amax = max(amax, abs(x[unsafe_offset=32 * i + j]))
        var d = amax / 127.0
        var id = 1.0 / d if d != 0 else Float32(0)
        put16(y, 34 * i, f16_bits(d))
        for j in range(32):
            y[unsafe_offset=34 * i + 2 + j] = UInt8(roundf(x[unsafe_offset=32 * i + j] * id) & 0xFF)

def q2_0_row(x: FP, y: BP, n: Int):
    for i in range(n // 64):
        var amax = Float32(0)
        for j in range(64):
            var a = abs(x[unsafe_offset=64 * i + j])
            if a > amax:
                amax = a
        var id = 1.0 / amax if amax > 0 else Float32(0)
        put16(y, 18 * i, f16_bits(amax))
        for j in range(16):
            y[unsafe_offset=18 * i + 2 + j] = 0
        for j in range(64):
            var q = max(0, min(3, roundf(x[unsafe_offset=64 * i + j] * id) + 1))
            y[unsafe_offset=18 * i + 2 + j // 4] |= UInt8(q << ((j % 4) * 2))

def q2_k_row(x: FP, y: BP, n: Int, sb: BP):
    comptime F = SIMD[DType.float32, 16]
    var xt = Array[F, 16](uninitialized=True)
    var wt = Array[F, 16](uninitialized=True)
    var Lt = Array[SIMD[DType.int32, 16], 16](uninitialized=True)
    var L = sb
    for i in range(n // QK_K):
        var xb = x.unsafe_offset(QK_K * i)
        var yb = y.unsafe_offset(84 * i)
        for l in range(16):
            var v = F(0)
            for j in range(16):
                v[j] = xb[unsafe_offset=16 * j + l]
            xt[l] = v
            wt[l] = abs(v)
        var mins = F(0)
        var scales = qkx2_lanes[16, 16, 3, 15, True](xt, wt, Lt, mins, -0.5, 0.1)
        for l in range(16):
            for j in range(16):
                L[unsafe_offset=16 * j + l] = UInt8(Lt[l][j])
        var max_scale = Float32(0)
        var max_min = Float32(0)
        for j in range(16):
            if scales[j] > max_scale:
                max_scale = scales[j]
            if mins[j] > max_min:
                max_min = mins[j]
        var d16 = Float16(0)
        var dm16 = Float16(0)
        if max_scale > 0:
            var iscale = 15.0 / max_scale
            for j in range(16):
                yb[unsafe_offset=j] = UInt8(nint(m(iscale, scales[j])) & 0xFF)
            d16 = Float16(max_scale / 15.0)
        else:
            for j in range(16):
                yb[unsafe_offset=j] = 0
        if max_min > 0:
            var iscale = 15.0 / max_min
            for j in range(16):
                yb[unsafe_offset=j] |= UInt8((nint(m(iscale, mins[j])) << 4) & 0xFF)
            dm16 = Float16(max_min / 15.0)
        put16(yb, 80, Int(bitcast[DType.uint16](d16)))
        put16(yb, 82, Int(bitcast[DType.uint16](dm16)))
        for j in range(16):
            var d = m(Float32(d16), Float32(Int(yb[unsafe_offset=j]) & 0xF))
            if d == 0:
                continue
            var dm = m(Float32(dm16), Float32(Int(yb[unsafe_offset=j]) >> 4))
            for ii in range(16):
                var l = max(0, min(3, nint((xb[unsafe_offset=16 * j + ii] + dm) / d)))
                L[unsafe_offset=16 * j + ii] = UInt8(l)
        for j in range(0, QK_K, 128):
            for l in range(32):
                yb[unsafe_offset=16 + j // 4 + l] = L[unsafe_offset=j + l] | (L[unsafe_offset=j + l + 32] << 2) | (
                    L[unsafe_offset=j + l + 64] << 4) | (L[unsafe_offset=j + l + 96] << 6)

def q4_k_row(x: FP, y: BP, n: Int, sb: BP):
    comptime F = SIMD[DType.float32, 8]
    var xt = Array[F, 32](uninitialized=True)
    var wt = Array[F, 32](uninitialized=True)
    var Lt = Array[SIMD[DType.int32, 8], 32](uninitialized=True)
    var L = sb
    for i in range(n // QK_K):
        var xb = x.unsafe_offset(QK_K * i)
        var yb = y.unsafe_offset(144 * i)
        var sum_x2 = F(0)
        for l in range(32):
            var v = F(0)
            for j in range(8):
                v[j] = xb[unsafe_offset=32 * j + l]
            xt[l] = v
            sum_x2 += vm[8](v, v)
        var av_x = sqrt(sum_x2 / F(32))
        for l in range(32):
            wt[l] = av_x + abs(xt[l])
        var mins = F(0)
        var scales = qkx2_lanes[8, 32, 15, 20, False](xt, wt, Lt, mins, -1.0, 0.1)
        for l in range(32):
            for j in range(8):
                L[unsafe_offset=32 * j + l] = UInt8(Lt[l][j])
        var max_scale = Float32(0)
        var max_min = Float32(0)
        for j in range(8):
            if scales[j] > max_scale:
                max_scale = scales[j]
            if mins[j] > max_min:
                max_min = mins[j]
        var inv_scale = 63.0 / max_scale if max_scale > 0 else Float32(0)
        var inv_min = 63.0 / max_min if max_min > 0 else Float32(0)
        for j in range(8):
            var ls = min(63, nint(m(inv_scale, scales[j])) & 0xFF)
            var lm = min(63, nint(m(inv_min, mins[j])) & 0xFF)
            if j < 4:
                yb[unsafe_offset=4 + j] = UInt8(ls)
                yb[unsafe_offset=4 + j + 4] = UInt8(lm)
            else:
                yb[unsafe_offset=4 + j + 4] = UInt8((ls & 0xF) | ((lm & 0xF) << 4))
                yb[unsafe_offset=4 + j - 4] |= UInt8((ls >> 4) << 6)
                yb[unsafe_offset=4 + j] |= UInt8((lm >> 4) << 6)
        var d16 = Float16(max_scale / 63.0)
        var dm16 = Float16(max_min / 63.0)
        put16(yb, 0, Int(bitcast[DType.uint16](d16)))
        put16(yb, 2, Int(bitcast[DType.uint16](dm16)))
        for j in range(8):
            var sc: Int
            var mv: Int
            if j < 4:
                sc = Int(yb[unsafe_offset=4 + j]) & 63
                mv = Int(yb[unsafe_offset=4 + j + 4]) & 63
            else:
                sc = (Int(yb[unsafe_offset=4 + j + 4]) & 0xF) | ((Int(yb[unsafe_offset=4 + j - 4]) >> 6) << 4)
                mv = (Int(yb[unsafe_offset=4 + j + 4]) >> 4) | ((Int(yb[unsafe_offset=4 + j]) >> 6) << 4)
            var d = Float32(d16) * Float32(sc)
            if d == 0:
                continue
            var dm = m(Float32(dm16), Float32(mv))
            for ii in range(32):
                var l = max(0, min(15, nint((xb[unsafe_offset=32 * j + ii] + dm) / d)))
                L[unsafe_offset=32 * j + ii] = UInt8(l)
        for k in range(4):
            for l in range(32):
                yb[unsafe_offset=16 + 32 * k + l] = L[unsafe_offset=64 * k + l] | (L[unsafe_offset=64 * k + l + 32] << 4)

def q6_k_row(x: FP, y: BP, n: Int, sb: BP):
    comptime F = SIMD[DType.float32, 16]
    var xt = Array[F, 16](uninitialized=True)
    var Lt = Array[SIMD[DType.int32, 16], 16](uninitialized=True)
    var L = sb
    for i in range(n // QK_K):
        var xb = x.unsafe_offset(QK_K * i)
        var yb = y.unsafe_offset(210 * i)
        for l in range(16):
            var v = F(0)
            for j in range(16):
                v[j] = xb[unsafe_offset=16 * j + l]
            xt[l] = v
        var scales = qx_lanes[16, 16, 32](xt, Lt)
        for l in range(16):
            for j in range(16):
                L[unsafe_offset=16 * j + l] = UInt8(Lt[l][j])
        var max_scale = Float32(0)
        var max_abs = Float32(0)
        for ib in range(16):
            if abs(scales[ib]) > max_abs:
                max_abs = abs(scales[ib])
                max_scale = scales[ib]
        if max_abs < EPS:
            for k in range(210):
                yb[unsafe_offset=k] = 0
            continue
        var iscale = -128.0 / max_scale
        var d16 = Float16(1.0 / iscale)
        put16(yb, 208, Int(bitcast[DType.uint16](d16)))
        for ib in range(16):
            yb[unsafe_offset=192 + ib] = UInt8(min(127, nint(m(iscale, scales[ib]))) & 0xFF)
        for j in range(16):
            var sc = Int(yb[unsafe_offset=192 + j])
            if sc >= 128:
                sc -= 256
            var d = Float32(d16) * Float32(sc)
            if d == 0:
                continue
            for ii in range(16):
                var l = max(-32, min(31, nint(xb[unsafe_offset=16 * j + ii] / d)))
                L[unsafe_offset=16 * j + ii] = UInt8(l + 32)
        for h in range(2):
            var j = 128 * h
            for l in range(32):
                var q1 = L[unsafe_offset=j + l] & 0xF
                var q2 = L[unsafe_offset=j + l + 32] & 0xF
                var q3 = L[unsafe_offset=j + l + 64] & 0xF
                var q4 = L[unsafe_offset=j + l + 96] & 0xF
                yb[unsafe_offset=64 * h + l] = q1 | (q3 << 4)
                yb[unsafe_offset=64 * h + l + 32] = q2 | (q4 << 4)
                yb[unsafe_offset=128 + 32 * h + l] = (L[unsafe_offset=j + l] >> 4) | ((L[unsafe_offset=j + l + 32] >> 4) << 2) | (
                    (L[unsafe_offset=j + l + 64] >> 4) << 4) | ((L[unsafe_offset=j + l + 96] >> 4) << 6)

@always_inline
def tq_amax(x: FP) -> Float32:
    var amax = Float32(0)
    for j in range(QK_K):
        amax = max(amax, abs(x[unsafe_offset=j]))
    return amax

def tq1_0_row(x: FP, y: BP, n: Int):
    for i in range(n // QK_K):
        var xb = x.unsafe_offset(QK_K * i)
        var yb = y.unsafe_offset(54 * i)
        var d = tq_amax(xb)
        var id = 1.0 / d if d != 0 else Float32(0)
        put16(yb, 52, f16_bits(d))
        var b = 0
        # 48 qs bytes: 32 bytes of 5 trits along 32, then 16 bytes along 16; then 4 qh bytes of 4 trits
        for part in range(2):
            var w = 32 if part == 0 else 16
            var base = 0 if part == 0 else 32
            for m in range(w):
                var q = 0
                for k in range(5):
                    q = q * 3 + (roundf(xb[unsafe_offset=b + m + k * w] * id) + 1)
                yb[unsafe_offset=base + m] = UInt8((q * 256 + 242) // 243)
            b += 5 * w
        for j in range(4):
            var q = 0
            for m in range(4):
                q = q * 3 + (roundf(xb[unsafe_offset=b + j + m * 4] * id) + 1)
            q *= 3
            yb[unsafe_offset=48 + j] = UInt8((q * 256 + 242) // 243)

def tq2_0_row(x: FP, y: BP, n: Int):
    for i in range(n // QK_K):
        var xb = x.unsafe_offset(QK_K * i)
        var yb = y.unsafe_offset(66 * i)
        var d = tq_amax(xb)
        var id = 1.0 / d if d != 0 else Float32(0)
        put16(yb, 64, f16_bits(d))
        for j in range(0, 64, 32):
            for m in range(32):
                var q = 0
                for k in range(4):
                    q += ((roundf(xb[unsafe_offset=4 * j + m + k * 32] * id) + 1) & 3) << (2 * k)
                yb[unsafe_offset=j + m] = UInt8(q)

def quantize_row(t: Int, x: FP, y: BP, n: Int, sb: BP):
    """One row of n floats into the row layout of type t. sb: 256 bytes of scratch."""
    if t == T_Q8_0:
        q8_0_row(x, y, n)
    elif t == T_Q4_0:
        q4_0_row(x, y, n, 8)
    elif t == T_Q5_0:
        q4_0_row(x, y, n, 16)
    elif t == T_Q2_0:
        q2_0_row(x, y, n)
    elif t == T_Q2_K:
        q2_k_row(x, y, n, sb)
    elif t == T_Q4_K:
        q4_k_row(x, y, n, sb)
    elif t == T_Q6_K:
        q6_k_row(x, y, n, sb)
    elif t == T_TQ1_0:
        tq1_0_row(x, y, n)
    elif t == T_TQ2_0:
        tq2_0_row(x, y, n)
    elif t == T_F16:
        for i in range(n):
            put16(y, 2 * i, f16_bits(x[unsafe_offset=i]))
    else:
        abort("no quantizer for ggml type " + String(t))
