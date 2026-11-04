"""Reading GGUF values in place and rendering them the way gguf-py renders them.

Three dialects, one walker:
  M_PY    text dump and json-free lists: Python str() of numbers (floats as float64 repr), repr() of strings
  M_NP    markdown: numpy str() of numbers (float32 as its shortest float32 repr), raw strings
  M_JSON  json.dump: numbers as json writes them, strings escaped, NaN/Infinity spelled out
"""
from std.math import isinf, isnan
from std.memory import bitcast
from gguf.model import Gguf, T_ARRAY, T_STRING, fixed_size, pread_into
from gguf.fmt import json_str, np_float32, py_repr

comptime T_INT8 = 1
comptime T_INT16 = 3
comptime T_INT32 = 5
comptime T_FLOAT32 = 6
comptime T_BOOL = 7
comptime T_INT64 = 11
comptime T_FLOAT64 = 12

comptime M_PY = 0
comptime M_NP = 1
comptime M_JSON = 2


def type_name(t: Int) raises -> String:
    """GGUFValueType name."""
    var names = ["UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32", "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64"]
    if t < 0 or t >= len(names):
        raise Error("unknown value type " + String(t))
    return names[t]


def peek(g: Gguf, at: Int, n: Int) raises -> List[UInt8]:
    var buf = List[UInt8](length=n, fill=0)
    if n > 0:
        pread_into(g.fd, buf, 0, n, at)
    return buf^


def uint_at(g: Gguf, at: Int, n: Int) raises -> UInt64:
    var b = peek(g, at, n)
    var v: UInt64 = 0
    for k in range(n):
        var sh = 8 * (n - 1 - k) if g.big else 8 * k
        v |= UInt64(b[k]) << UInt64(sh)
    return v


def read_str(g: Gguf, at: Int) raises -> String:
    var n = Int(uint_at(g, at, 8))
    var b = peek(g, at + 8, n)
    return String(from_utf8_lossy=Span(b))


def end_at(g: Gguf, at: Int, t: Int) raises -> Int:
    """Offset just past the payload of type t starting at `at`."""
    if t == T_STRING:
        return at + 8 + Int(uint_at(g, at, 8))
    if t != T_ARRAY:
        return at + fixed_size(t)
    var et = Int(uint_at(g, at, 4))
    var n = Int(uint_at(g, at + 4, 8))
    var w = fixed_size(et)
    if w > 0:
        return at + 12 + n * w
    var p = at + 12
    for _ in range(n):
        p = end_at(g, p, et)
    return p


def float_text(d: Float64, mode: Int) -> String:
    if mode == M_JSON:
        if isnan(d):
            return "NaN"
        if isinf(d):
            return "Infinity" if d > 0.0 else "-Infinity"
    return String(d)


def leaf(g: Gguf, at: Int, t: Int, mode: Int) raises -> String:
    """One scalar or string payload at `at` of type t."""
    if t == T_STRING:
        var s = read_str(g, at)
        if mode == M_PY:
            return py_repr(s)
        if mode == M_JSON:
            return json_str(s)
        return s
    var w = fixed_size(t)
    var v = uint_at(g, at, w)
    if t == T_BOOL:
        var on = v != 0
        if mode == M_JSON:
            return "true" if on else "false"
        return "True" if on else "False"
    if t == T_FLOAT32:
        var f = bitcast[DType.float32, 1](SIMD[DType.uint32, 1](UInt32(v)))[0]
        if mode == M_NP:
            return np_float32(f)
        return float_text(Float64(f), mode)
    if t == T_FLOAT64:
        return float_text(bitcast[DType.float64, 1](SIMD[DType.uint64, 1](v))[0], mode)
    if t == T_INT8 or t == T_INT16 or t == T_INT32:
        var bits = 8 * w
        var x = Int(v)
        if x >= (1 << (bits - 1)):
            x -= 1 << bits
        return String(x)
    if t == T_INT64:
        return String(bitcast[DType.int64, 1](SIMD[DType.uint64, 1](v))[0])
    return String(v)


def emit(g: Gguf, at: Int, t: Int, mode: Int, limit: Int, mut out: List[String]) raises:
    """Append the leaves of the payload at `at` (type t) to `out`, flattening nested arrays, until
    `out` holds `limit` items: gguf-py's `contents(slice(limit))`."""
    if t != T_ARRAY:
        out.append(leaf(g, at, t, mode))
        return
    var et = Int(uint_at(g, at, 4))
    var n = Int(uint_at(g, at + 4, 8))
    var p = at + 12
    for _ in range(n):
        if len(out) >= limit:
            return
        emit(g, p, et, mode, limit, out)
        p = end_at(g, p, et)
