"""Python-compatible text: exact `%.Nf`, round-half-even, str.title, repr(str), json strings, code-point widths.

Float repr needs no helper: Mojo's String(Float64) and String(Float32) already print Python's and numpy's
shortest repr (checked against both, 2026-10-12).
"""
from std.math import floor, isinf, isnan
from std.memory import bitcast


def rep(ch: String, n: Int) -> String:
    var s = String()
    for _ in range(n):
        s += ch
    return s


def cp_starts(s: String) -> List[Int]:
    """Byte offset of every code point, plus the byte length as a final entry."""
    var out = List[Int]()
    var i = 0
    var n = s.byte_length()
    var b = s.as_bytes()
    while i < n:
        out.append(i)
        var c = Int(b[i])
        i += 1 if c < 0x80 else 2 if c >> 5 == 6 else 3 if c >> 4 == 14 else 4
    out.append(n)
    return out^


def cp_slice(s: String, a: Int, z: Int) -> String:
    """Code points [a, z) of s (Python's s[a:z])."""
    var at = cp_starts(s)
    var lo = at[min(a, len(at) - 1)]
    var hi = at[min(max(z, a), len(at) - 1)]
    return String(unsafe_from_utf8=s.as_bytes()[lo:hi])


def cp_len(s: String) -> Int:
    return len(cp_starts(s)) - 1


def ljust(s: String, w: Int) -> String:
    return s + rep(" ", w - cp_len(s))


def rjust(s: String, w: Int) -> String:
    return rep(" ", w - cp_len(s)) + s


def code_point(s: String, i: Int) -> Int:
    """Code point starting at byte i (no validation: the input is GGUF text)."""
    var b = s.as_bytes()
    var c = Int(b[i])
    if c < 0x80:
        return c
    if c >> 5 == 6:
        return ((c & 0x1F) << 6) | (Int(b[i + 1]) & 0x3F)
    if c >> 4 == 14:
        return ((c & 0x0F) << 12) | ((Int(b[i + 1]) & 0x3F) << 6) | (Int(b[i + 2]) & 0x3F)
    return ((c & 0x07) << 18) | ((Int(b[i + 1]) & 0x3F) << 12) | ((Int(b[i + 2]) & 0x3F) << 6) | (Int(b[i + 3]) & 0x3F)


def hexn(v: Int, n: Int) -> String:
    """v as n lowercase hex digits."""
    var digits = "0123456789abcdef"
    var s = String()
    for k in range(n):
        var sh = 4 * (n - 1 - k)
        s += String(digits[byte=(v >> sh) & 0xF])
    return s


def sub_str(s: String, a: Int, z: Int) -> String:
    """Bytes [a, z) of an ASCII-only string (digit strings here)."""
    return String(unsafe_from_utf8=s.as_bytes()[a:z])


def py_title(s: String) -> String:
    """str.title() for ASCII text: a letter starts a word unless the one before it is a letter."""
    var out = List[UInt8]()
    var cased = False
    for c in s.as_bytes():
        var ch = Int(c)
        var is_alpha = (ch >= 65 and ch <= 90) or (ch >= 97 and ch <= 122)
        var upper = ch >= 65 and ch <= 90
        if is_alpha and cased and upper:
            ch += 32
        elif is_alpha and not cased and not upper:
            ch -= 32
        out.append(UInt8(ch))
        cased = is_alpha
    return String(unsafe_from_utf8=Span(out))


def py_repr(s: String) -> String:
    """repr() of a Python str: quote choice, \\ \\' \\t \\n \\r and \\xNN for control bytes; other text as is."""
    var q = "'"
    if s.find("'") >= 0 and s.find('"') < 0:
        q = '"'
    var out = q
    var b = s.as_bytes()
    var i = 0
    while i < len(b):
        var cp = code_point(s, i)
        var step = 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4
        if cp == 92:
            out += "\\\\"
        elif cp == Int(ord(q)):
            out += "\\" + q
        elif cp == 9:
            out += "\\t"
        elif cp == 10:
            out += "\\n"
        elif cp == 13:
            out += "\\r"
        elif cp < 0x20 or (cp >= 0x7F and cp < 0xA0):
            out += "\\x" + hexn(cp, 2)
        else:
            out += String(unsafe_from_utf8=b[i : i + step])
        i += step
    return out + q


def json_str(s: String) -> String:
    """json.dumps(s) with ensure_ascii, the way gguf_dump.py's json output writes strings."""
    var out = String('"')
    var b = s.as_bytes()
    var i = 0
    while i < len(b):
        var cp = code_point(s, i)
        var step = 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4
        i += step
        if cp == 34:
            out += '\\"'
        elif cp == 92:
            out += "\\\\"
        elif cp == 10:
            out += "\\n"
        elif cp == 13:
            out += "\\r"
        elif cp == 9:
            out += "\\t"
        elif cp == 8:
            out += "\\b"
        elif cp == 12:
            out += "\\f"
        elif cp < 0x20:
            out += "\\u" + hexn(cp, 4)
        elif cp < 0x80:
            out += String(unsafe_from_utf8=b[i - step : i])
        elif cp < 0x10000:
            out += "\\u" + hexn(cp, 4)
        else:
            var v = cp - 0x10000
            out += "\\u" + hexn(0xD800 + (v >> 10), 4) + "\\u" + hexn(0xDC00 + (v & 0x3FF), 4)
    return out + '"'


def round_half_even(x: Float64) -> Int:
    """Python round(float) for x >= 0."""
    var f = floor(x)
    var r = x - f
    var n = Int(f)
    if r > 0.5 or (r == 0.5 and n % 2 == 1):
        n += 1
    return n


comptime LIMB = 1000000000


def _mul(mut limbs: List[UInt64], m: UInt64):
    var carry: UInt64 = 0
    for i in range(len(limbs)):
        var v = limbs[i] * m + carry
        limbs[i] = v % UInt64(LIMB)
        carry = v // UInt64(LIMB)
    while carry > 0:
        limbs.append(carry % UInt64(LIMB))
        carry = carry // UInt64(LIMB)


def _decimal(limbs: List[UInt64]) -> String:
    var s = String(limbs[len(limbs) - 1])
    for i in range(len(limbs) - 2, -1, -1):
        var part = String(limbs[i])
        s += rep("0", 9 - part.byte_length()) + part
    return s


def fixed(x: Float64, nd: Int) raises -> String:
    """`'%.{nd}f' % x`: the exact binary value rounded half-even, not the shortest repr rounded."""
    if isnan(x):
        return "nan"
    var neg = x < 0.0
    var a = -x if neg else x
    var sign = "-" if neg else ""
    if isinf(a):
        return sign + "inf"
    var bits = bitcast[DType.uint64, 1](SIMD[DType.float64, 1](a))[0]
    var expo = Int((bits >> 52) & 0x7FF)
    var frac = UInt64(bits & ((UInt64(1) << 52) - 1))
    var m = frac
    var e = -1074
    if expo != 0:
        m = frac | (UInt64(1) << 52)
        e = expo - 1075
    # a = m * 2^e = (m * 5^-e) / 10^-e when e < 0, else the integer m * 2^e
    var limbs = List[UInt64]()
    limbs.append(m % UInt64(LIMB))
    limbs.append((m // UInt64(LIMB)) % UInt64(LIMB))
    limbs.append(m // (UInt64(LIMB) * UInt64(LIMB)))
    var k = 0
    if e < 0:
        k = -e
        var rest = k
        while rest >= 13:
            _mul(limbs, 1220703125)
            rest -= 13
        var p: UInt64 = 1
        for _ in range(rest):
            p *= 5
        _mul(limbs, p)
    else:
        var rest = e
        while rest >= 29:
            _mul(limbs, UInt64(1) << 29)
            rest -= 29
        _mul(limbs, UInt64(1) << UInt64(rest))
    var digits = _decimal(limbs)
    if digits.byte_length() < k + 1:
        digits = rep("0", k + 1 - digits.byte_length()) + digits
    var n_int = digits.byte_length() - k
    # keep = integer digits + the first nd fraction digits; decide the rounding from the rest
    var keep = sub_str(digits, 0, n_int + min(k, nd))
    var rest_frac = sub_str(digits, n_int + min(k, nd), digits.byte_length())
    keep += rep("0", nd - min(k, nd))
    var up = False
    if rest_frac.byte_length() > 0:
        var first = Int(rest_frac.as_bytes()[0])
        if first > 53:
            up = True
        elif first == 53:
            var tail_nonzero = False
            for c in rest_frac.as_bytes()[1:]:
                if c != 48:
                    tail_nonzero = True
            var last = Int(keep.as_bytes()[keep.byte_length() - 1]) - 48
            up = tail_nonzero or last % 2 == 1
    if up:
        var b = keep.as_bytes()
        var out = List[UInt8]()
        for c in b:
            out.append(c)
        var i = len(out) - 1
        var carry = True
        while carry and i >= 0:
            if out[i] == 57:
                out[i] = 48
                i -= 1
            else:
                out[i] += 1
                carry = False
        if carry:
            out.insert(0, 49)
        keep = String(unsafe_from_utf8=Span(out))
    var cut = keep.byte_length() - nd
    var text = sub_str(keep, 0, cut)
    if nd > 0:
        text += "." + sub_str(keep, cut, keep.byte_length())
    return sign + text


def np_float32(f: Float32) -> String:
    """numpy's str() of a float32 scalar: the shortest float32 digits, scientific below 1e-4 and from 1e6
    (Python's float repr moves to scientific at 1e16 instead)."""
    if f == 0.0 or isnan(f) or isinf(f):
        return String(f)
    var s = String(f)
    var neg = s.startswith("-")
    if neg:
        s = sub_str(s, 1, s.byte_length())
    var mant = s
    var ex = 0
    var epos = s.find("e")
    if epos >= 0:
        mant = sub_str(s, 0, epos)
        var e_neg = sub_str(s, epos + 1, epos + 2) == "-"
        for c in s.as_bytes()[epos + 2 :]:
            ex = ex * 10 + Int(c) - 48
        if e_neg:
            ex = -ex
    var dot = mant.find(".")
    var int_len = dot if dot >= 0 else mant.byte_length()
    var raw_digits = mant.replace(".", "")
    # value = int(digits) * 10^exp10, digits without leading or trailing zeros
    var exp10 = ex + int_len - raw_digits.byte_length()
    var digits = raw_digits
    while digits.byte_length() > 1 and sub_str(digits, 0, 1) == "0":
        digits = sub_str(digits, 1, digits.byte_length())
    while digits.byte_length() > 1 and sub_str(digits, digits.byte_length() - 1, digits.byte_length()) == "0":
        digits = sub_str(digits, 0, digits.byte_length() - 1)
        exp10 += 1
    var n = digits.byte_length()
    var out = String()
    var a = abs(Float64(f))
    if a < 1e-4 or a >= 1e6:
        var sci = exp10 + n - 1
        out = sub_str(digits, 0, 1)
        if n > 1:
            out += "." + sub_str(digits, 1, n)
        var mag = abs(sci)
        out += "e" + ("-" if sci < 0 else "+") + ("0" if mag < 10 else "") + String(mag)
    elif exp10 >= 0:
        out = digits + rep("0", exp10) + ".0"
    else:
        var k = -exp10
        if n > k:
            out = sub_str(digits, 0, n - k) + "." + sub_str(digits, n - k, n)
        else:
            out = "0." + rep("0", k - n) + digits
    return ("-" if neg else "") + out
