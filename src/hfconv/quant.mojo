"""Tensor payload converters, bit-exact with gguf-py quants.py for BF16, F32 and Q8_0 output from
BF16 / F16 / F32 rows. Each takes the source tensor's base address, an element range and a
destination address, and writes the output bytes for that range:

  BF16  float32 bits, NaN forced quiet, round to nearest even (a BF16 source passes through)
  F32   exact widening
  Q8_0  per block of 32: d = max|x| / 127 in f32, q = round-half-away(x * (1/d)), d stored as f16
"""
from std.memory import bitcast
from std.math import floor, copysign
from st.safetensors import ST_F32, ST_F16, ST_BF16

comptime QK8_0 = 32
comptime Q8_0_BYTES = 34

@always_inline
def _src[T: DType](addr: Int) -> Pointer[Scalar[T], ImmutAnyOrigin]:
    return Pointer[Scalar[T], ImmutAnyOrigin](unsafe_from_address=addr)

@always_inline
def _dst[T: DType](addr: Int) -> Pointer[Scalar[T], MutAnyOrigin]:
    return Pointer[Scalar[T], MutAnyOrigin](unsafe_from_address=addr)

@always_inline
def load_f32(dtype: Int, base: Int, i: Int) -> Float32:
    if dtype == ST_BF16:
        return bitcast[DType.float32, 1](UInt32(_src[DType.uint16](base)[unsafe_offset=i]) << 16)
    if dtype == ST_F16:
        return _src[DType.float16](base)[unsafe_offset=i].cast[DType.float32]()
    return _src[DType.float32](base)[unsafe_offset=i]

@always_inline
def f32_to_bf16(x: Float32) -> UInt16:
    var n = bitcast[DType.uint32, 1](x)
    if (n & 0x7FFFFFFF) > 0x7F800000:
        return UInt16(((n & 0xFFFF0000) | 0x400000) >> 16)
    return UInt16((UInt64(n) + 0x7FFF + UInt64((n >> 16) & 1)) >> 16)

def emit_f32(dtype: Int, base: Int, start: Int, count: Int, dst: Int):
    var o = _dst[DType.float32](dst)
    for i in range(count):
        o[unsafe_offset=i] = load_f32(dtype, base, start + i)

def emit_bf16(dtype: Int, base: Int, start: Int, count: Int, dst: Int):
    var o = _dst[DType.uint16](dst)
    var i = 0
    if dtype == ST_BF16:
        var s = _src[DType.uint16](base).unsafe_offset(start)
        while i + 32 <= count:
            var v = s.unsafe_load[width=32](i)
            var quiet = (v & 0x7FFF).gt(0x7F80)
            o.unsafe_offset(i).unsafe_store(quiet.select(v | 0x40, v))
            i += 32
    while i < count:
        o[unsafe_offset=i] = f32_to_bf16(load_f32(dtype, base, start + i))
        i += 1

@always_inline
def _load_block(dtype: Int, base: Int, at: Int) -> SIMD[DType.float32, QK8_0]:
    if dtype == ST_BF16:
        var h = _src[DType.uint16](base).unsafe_load[width=QK8_0](at)
        return bitcast[DType.float32, QK8_0](h.cast[DType.uint32]() << 16)
    if dtype == ST_F16:
        return _src[DType.float16](base).unsafe_load[width=QK8_0](at).cast[DType.float32]()
    return _src[DType.float32](base).unsafe_load[width=QK8_0](at)

def emit_q8_0(dtype: Int, base: Int, start: Int, count: Int, dst: Int):
    """count is a multiple of QK8_0."""
    var d16 = _dst[DType.float16](dst)
    var q8 = _dst[DType.int8](dst)
    for b in range(count // QK8_0):
        var x = _load_block(dtype, base, start + b * QK8_0)
        var d = abs(x).reduce_max() / 127.0
        var id = Float32(0.0) if d == 0.0 else Float32(1.0) / d
        var y = x * id
        var a = abs(y)
        var fl = floor(a)
        var q = copysign(fl + floor(2.0 * (a - fl)), y).cast[DType.int8]()
        var at = b * Q8_0_BYTES
        d16.unsafe_offset(at // 2).unsafe_store(d.cast[DType.float16]())
        q8.unsafe_offset(at + 2).unsafe_store(q)
