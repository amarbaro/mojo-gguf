"""gguf-py Metadata naming heuristics, ASCII only: a model directory name ("qwen2.5-0.5b-instruct-hf")
becomes name / basename / finetune / version / size_label exactly as `Metadata.get_model_id_components`
and `id_to_title` do, and `size_label` rounds a parameter count the way `size_label()` does."""
from std.math import floor, pow

comptime BASENAME = 1
comptime SIZE = 2
comptime FINETUNE = 4
comptime VERSION = 8
comptime TYPE = 16

def _digit(c: UInt8) -> Bool:
    return c >= 48 and c <= 57

def _alpha(c: UInt8) -> Bool:
    return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)

def _lower(s: String) -> String:
    var out = List[UInt8]()
    for c in s.as_bytes():
        out.append(c + 32 if c >= 65 and c <= 90 else c)
    return String(unsafe_from_utf8=Span(out))

def _digits(b: Span[UInt8, _], i: Int) -> Int:
    """Index after the run of digits starting at i."""
    var k = i
    while k < len(b) and _digit(b[k]):
        k += 1
    return k

def is_version(s: String) -> Bool:
    """(v|iter)?\\d+([.]\\d+)*"""
    var b = _lower(s).as_bytes()
    var i = 0
    if len(b) >= 4 and b[0] == 105 and b[1] == 116 and b[2] == 101 and b[3] == 114:
        i = 4
    elif len(b) >= 1 and b[0] == 118:
        i = 1
    var k = _digits(b, i)
    if k == i:
        return False
    while k < len(b):
        if b[k] != 46:
            return False
        var e = _digits(b, k + 1)
        if e == k + 1:
            return False
        k = e
    return True

def is_quant(s: String) -> Bool:
    """i?q\\d(_\\w)*|b?fp?(16|32)"""
    var l = _lower(s)
    var b = l.as_bytes()
    var i = 1 if len(b) > 0 and b[0] == 105 else 0
    if i + 1 < len(b) and b[i] == 113 and _digit(b[i + 1]):
        var k = i + 2
        var ok = True
        while k < len(b):
            if b[k] != 95 or k + 1 >= len(b) or not (_alpha(b[k + 1]) or _digit(b[k + 1]) or b[k + 1] == 95):
                ok = False
                break
            k += 2
        if ok:
            return True
    var j = 1 if len(b) > 0 and b[0] == 98 else 0
    if j < len(b) and b[j] == 102:
        j += 1
        if j < len(b) and b[j] == 112:
            j += 1
        return len(b) - j == 2 and ((b[j] == 49 and b[j + 1] == 54) or (b[j] == 51 and b[j + 1] == 50))
    return False

def _is_size_tail(b: Span[UInt8, _], i: Int) -> Bool:
    """\\d+([._]\\d+)?[KMBT][\\d]? from i, anchored at the end."""
    var k = _digits(b, i)
    if k == i:
        return False
    if k < len(b) and (b[k] == 46 or b[k] == 95):
        var e = _digits(b, k + 1)
        if e > k + 1:
            k = e
    if k >= len(b) or not (b[k] == 107 or b[k] == 109 or b[k] == 98 or b[k] == 116):
        return False
    k += 1
    if k < len(b) and _digit(b[k]):
        k += 1
    return k == len(b)

def is_size(s: String) -> Bool:
    """(([A]|\\d+[x])?\\d+([._]\\d+)?[KMBT][\\d]?|small|mini|medium|large|x?xl)"""
    var l = _lower(s)
    if l == "small" or l == "mini" or l == "medium" or l == "large" or l == "xl" or l == "xxl":
        return True
    var b = l.as_bytes()
    if _is_size_tail(b, 0):
        return True
    if len(b) > 0 and b[0] == 97 and _is_size_tail(b, 1):
        return True
    var k = _digits(b, 0)
    return k > 0 and k < len(b) and b[k] == 120 and _is_size_tail(b, k + 1)

def _all_alpha(s: String) -> Bool:
    for c in s.as_bytes():
        if not _alpha(c):
            return False
    return True

def _has_digit(s: String) -> Bool:
    for c in s.as_bytes():
        if _digit(c):
            return True
    return False

def id_to_title(s: String) -> String:
    """Title-case every all-lowercase word that is not a version or does not start with a digit."""
    var words = List[String]()
    for w in s.replace("-", " ").split():
        var word = String(w)
        var b = word.as_bytes()
        var has_lower = False
        var has_upper = False
        for c in b:
            has_lower = has_lower or (c >= 97 and c <= 122)
            has_upper = has_upper or (c >= 65 and c <= 90)
        var keep = _digit(b[0]) or (b[0] == 118 and is_version(String(word.removeprefix("v"))) and _digit(b[1]) if len(b) > 1 else False)
        if has_lower and not has_upper and not keep:
            var out = List[UInt8]()
            var cased_run = False
            for c in b:
                if c >= 97 and c <= 122:
                    out.append(c - 32 if not cased_run else c)
                    cased_run = True
                else:
                    out.append(c)
                    cased_run = _alpha(c)
            word = String(unsafe_from_utf8=Span(out))
        words.append(word^)
    var res = String()
    for k in range(len(words)):
        res += (" " if k > 0 else "") + words[k]
    return res

@fieldwise_init
struct Comps(Copyable, Movable):
    var full: String
    var org: String
    var basename: String
    var finetune: String
    var version: String
    var size_label: String

def components(model_id: String, total_params: Int) raises -> Comps:
    """Empty string means None."""
    if model_id.byte_length() == 0:
        return Comps(String(), String(), String(), String(), String(), String())
    if model_id.find(" ") >= 0:
        return Comps(model_id.copy(), String(), String(), String(), String(), String())
    var full = model_id.copy()
    var org = String()
    var slash = model_id.find("/")
    if slash >= 0:
        org = String(unsafe_from_utf8=Span(model_id.as_bytes())[0:slash])
        full = String(unsafe_from_utf8=Span(model_id.as_bytes())[slash + 1 : model_id.byte_length()])
        if org.byte_length() > 0 and org.as_bytes()[0] == 46:
            org = String()
    var parts = List[String]()
    for p in full.split("-"):
        if p.byte_length() > 0:
            parts.append(String(p))
    var types = List[Int]()
    for _ in range(len(parts)):
        types.append(0)
    for i in range(len(parts)):
        var part = parts[i].copy()
        if is_version(part):
            types[i] |= VERSION
        elif is_quant(part):
            types[i] |= TYPE
            parts[i] = part.upper()
        elif i > 0 and is_size(part):
            var pb = List[UInt8](Span(part.replace("_", ".").as_bytes()))
            var n = len(pb)
            if _digit(pb[n - 1]):  # bloom-style "7b1" -> "7.1b"
                var tail = pb[n - 2]
                pb[n - 2] = 46
                pb.append(tail)
                n += 1
            var last = pb[n - 1]
            if n > 1 and _digit(pb[n - 2]) and (last == 107 or last == 109 or last == 98 or last == 116):
                pb[n - 1] -= 32
            if total_params != 0:
                var ok = True
                var value = Float64(0)
                try:
                    value = Float64(String(unsafe_from_utf8=Span(pb)[0 : n - 1]))
                except:
                    ok = False
                if ok:
                    var unit = String(" KMBT").find(String(unsafe_from_utf8=Span(pb)[n - 1 : n]))
                    var label = value * pow(Float64(1000), Float64(unit))
                    var tp = Float64(total_params)
                    var far = (label - tp if label > tp else tp - label) > Float64(7 * total_params // 8)
                    if (total_params < 0 and label < Float64(-total_params // 8)) or (total_params > 0 and far):
                        types[i] |= FINETUNE
                        if pb[n - 1] >= 65 and pb[n - 1] <= 90:
                            pb[n - 1] += 32
            part = String(unsafe_from_utf8=Span(pb))
            if types[i] == 0:
                types[i] |= SIZE
            parts[i] = part^
        elif i > 0:
            var l = _lower(part)
            if l == "chat" or l == "instruct" or l == "vision" or l == "lora":
                if total_params < 0 and l == "lora":
                    types[i] |= TYPE
                else:
                    types[i] |= FINETUNE
    var any_digit_size = False
    for i in range(len(parts)):
        if types[i] & SIZE != 0 and _has_digit(parts[i]):
            any_digit_size = True
    if any_digit_size:
        for i in range(len(parts)):
            if types[i] & SIZE != 0 and _all_alpha(parts[i]):
                types[i] &= ~SIZE
    var at_start = True
    for i in range(len(parts)):
        if at_start and ((types[i] == 0 and _alpha(parts[i].as_bytes()[0])) or types[i] & VERSION != 0):
            types[i] |= BASENAME
        else:
            if at_start:
                at_start = False
            if types[i] == 0:
                types[i] |= FINETUNE
    var r = len(parts) - 1
    while r >= 0:
        if types[r] & BASENAME != 0 and types[r] != BASENAME:
            types[r] &= ~BASENAME
            r -= 1
        else:
            break
    var basename = String()
    var size_label = String()
    var finetune = String()
    var version = String()
    for i in range(len(parts)):
        if types[i] & BASENAME != 0:
            basename += ("-" if basename.byte_length() > 0 else "") + parts[i]
        if types[i] & SIZE != 0:
            var dup = False
            for k in range(i):
                dup = dup or (types[k] & SIZE != 0 and parts[k] == parts[i])
            if not dup:
                size_label += ("-" if size_label.byte_length() > 0 else "") + parts[i]
        if types[i] & FINETUNE != 0:
            finetune += ("-" if finetune.byte_length() > 0 else "") + parts[i]
        if types[i] & VERSION != 0 and types[i] & BASENAME == 0:
            version += ("-" if version.byte_length() > 0 else "") + parts[i]
    if size_label.byte_length() == 0 and finetune.byte_length() == 0 and version.byte_length() == 0:
        basename = String()
    return Comps(full^, org^, basename^, finetune^, version^, size_label^)

def _round_half_even(x: Float64) -> Int:
    var f = floor(x)
    var d = x - f
    var i = Int(f)
    if d > 0.5 or (d == 0.5 and i % 2 == 1):
        i += 1
    return i

def size_label(total_params: Int) -> String:
    """model_weight_count_rounded_notation(total, min_digits=2)."""
    var n = Float64(total_params)
    var scaled: Float64
    var suffix: String
    if n > 1e12:
        scaled = n * 1e-12
        suffix = "T"
    elif n > 1e9:
        scaled = n * 1e-9
        suffix = "B"
    elif n > 1e6:
        scaled = n * 1e-6
        suffix = "M"
    else:
        scaled = n * 1e-3
        suffix = "K"
    var whole = String(_round_half_even(scaled))
    var shown = 0 if whole == "0" else whole.byte_length()
    var fix = max(2 - shown, 0)
    var scale = 1
    for _ in range(fix):
        scale *= 10
    var v = _round_half_even(scaled * Float64(scale))
    var text = String(v // scale)
    if fix > 0:
        var frac = String(v % scale)
        for _ in range(fix - frac.byte_length()):
            frac = "0" + frac
        text += "." + frac
    return text + suffix
