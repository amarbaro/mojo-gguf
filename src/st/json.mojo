"""JSON reader over one byte buffer. Nothing is built: a value is a byte offset, and the accessors
decode only what the caller asks for. `member` / `element` walk a container and skip each value
they hand back, so a 6 MB tokenizer.json costs its bytes plus the strings actually decoded."""
from std.ffi import external_call
from gguf.model import open_fd, pread_into

comptime QUOTE = 34
comptime BACKSLASH = 92

def is_space(c: UInt8) -> Bool:
    return c == 32 or c == 10 or c == 13 or c == 9

def push_utf8(mut out: List[UInt8], cp: Int):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))

struct Json(Movable):
    var b: List[UInt8]

    def __init__(out self, var bytes: List[UInt8]):
        self.b = bytes^

    def __init__(out self, path: String) raises:
        var fd = open_fd(path, 0)
        var size = Int(external_call["lseek64", Int](fd, Int64(0), Int32(2)))
        self.b = List[UInt8](length=size, fill=0)
        pread_into(fd, self.b, 0, size, 0)
        _ = external_call["close", Int32](fd)

    def ws(self, p: Int) -> Int:
        """First non-space at or after p; past the end for the -1 of an absent key (a negative index would wrap)."""
        if p < 0:
            return len(self.b)
        var i = p
        while i < len(self.b) and is_space(self.b[i]):
            i += 1
        return i

    def kind(self, p: Int) -> Int:
        """First byte of the value at p: '{' '[' '"' 'n' 't' 'f', a digit or '-'; 0 past the end."""
        var i = self.ws(p)
        return Int(self.b[i]) if i < len(self.b) else 0

    def str_end(self, p: Int) raises -> Int:
        """Index of the closing quote of the string whose opening quote is at p."""
        var i = p + 1
        while i < len(self.b):
            var c = self.b[i]
            if c == QUOTE:
                return i
            i += 2 if c == BACKSLASH else 1
        raise Error("unterminated JSON string at byte " + String(p))

    def skip(self, p: Int) raises -> Int:
        """Offset just past the value at p."""
        var i = self.ws(p)
        if i >= len(self.b):
            raise Error("JSON ends where a value was expected")
        var c = self.b[i]
        if c == QUOTE:
            return self.str_end(i) + 1
        if c == 123 or c == 91:
            var depth = 0
            while i < len(self.b):
                c = self.b[i]
                if c == QUOTE:
                    i = self.str_end(i) + 1
                    continue
                if c == 123 or c == 91:
                    depth += 1
                elif c == 125 or c == 93:
                    depth -= 1
                    if depth == 0:
                        return i + 1
                i += 1
            raise Error("unterminated JSON container")
        while i < len(self.b) and not is_space(self.b[i]) and self.b[i] != 44 and self.b[i] != 125 and self.b[i] != 93:
            i += 1
        return i

    def _hex4(self, at: Int) raises -> Int:
        var v = 0
        for k in range(4):
            var c = Int(self.b[at + k])
            if c >= 48 and c <= 57:
                v = v * 16 + c - 48
            elif c >= 97 and c <= 102:
                v = v * 16 + c - 87
            elif c >= 65 and c <= 70:
                v = v * 16 + c - 55
            else:
                raise Error("bad \\u escape at byte " + String(at))
        return v

    def string(self, p: Int) raises -> String:
        var i = self.ws(p)
        if i >= len(self.b) or self.b[i] != QUOTE:
            raise Error("expected a JSON string at byte " + String(i))
        var e = self.str_end(i)
        var plain = True
        for k in range(i + 1, e):
            if self.b[k] == BACKSLASH:
                plain = False
                break
        if plain:
            return String(unsafe_from_utf8=Span(self.b)[i + 1 : e])
        var out = List[UInt8](capacity=e - i)
        var k = i + 1
        while k < e:
            var c = self.b[k]
            if c != BACKSLASH:
                out.append(c)
                k += 1
                continue
            var d = Int(self.b[k + 1])
            k += 2
            if d == 110:
                out.append(10)
            elif d == 116:
                out.append(9)
            elif d == 114:
                out.append(13)
            elif d == 98:
                out.append(8)
            elif d == 102:
                out.append(12)
            elif d == 34 or d == 92 or d == 47:
                out.append(UInt8(d))
            elif d == 117:
                var cp = self._hex4(k)
                k += 4
                if cp >= 0xD800 and cp < 0xDC00 and self.b[k] == BACKSLASH and self.b[k + 1] == 117:
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (self._hex4(k + 2) - 0xDC00)
                    k += 6
                push_utf8(out, cp)
            else:
                raise Error("bad escape \\" + chr(d) + " at byte " + String(k))
        return String(unsafe_from_utf8=Span(out))

    def number(self, p: Int) raises -> Float64:
        var i = self.ws(p)
        var e = self.skip(i)
        return Float64(String(unsafe_from_utf8=Span(self.b)[i:e]))

    def integer(self, p: Int) raises -> Int:
        var i = self.ws(p)
        var neg = i < len(self.b) and self.b[i] == 45
        if neg:
            i += 1
        var v = 0
        var start = i
        while i < len(self.b) and self.b[i] >= 48 and self.b[i] <= 57:
            v = v * 10 + Int(self.b[i]) - 48
            i += 1
        if i == start:
            raise Error("expected a JSON integer at byte " + String(start))
        return -v if neg else v

    def boolean(self, p: Int) -> Bool:
        return self.kind(p) == 116

    def open(self, p: Int) raises -> Int:
        """Cursor just inside the object or array at p."""
        var i = self.ws(p)
        if i >= len(self.b) or (self.b[i] != 123 and self.b[i] != 91):
            raise Error("expected a JSON object or array at byte " + String(i))
        return i + 1

    def element(self, mut cur: Int) raises -> Int:
        """Next array element's offset (and cur moves past it), or -1 at the end."""
        var i = self.ws(cur)
        if i < len(self.b) and self.b[i] == 44:
            i = self.ws(i + 1)
        if i >= len(self.b):
            raise Error("unterminated JSON array")
        if self.b[i] == 93:
            cur = i + 1
            return -1
        cur = self.skip(i)
        return i

    def member(self, mut cur: Int, mut key: String) raises -> Int:
        """Next object member: key is set, its value's offset is returned, cur moves past it; -1 at the end."""
        var i = self.ws(cur)
        if i < len(self.b) and self.b[i] == 44:
            i = self.ws(i + 1)
        if i >= len(self.b):
            raise Error("unterminated JSON object")
        if self.b[i] == 125:
            cur = i + 1
            return -1
        key = self.string(i)
        var c = self.ws(self.str_end(i) + 1)
        if c >= len(self.b) or self.b[c] != 58:
            raise Error("expected ':' at byte " + String(c))
        var v = self.ws(c + 1)
        cur = self.skip(v)
        return v

    def get(self, obj: Int, key: String) raises -> Int:
        """Offset of obj[key], or -1 when obj is not an object or has no such key."""
        if self.kind(obj) != 123:
            return -1
        var cur = self.open(obj)
        var k = String()
        while True:
            var v = self.member(cur, k)
            if v < 0:
                return -1
            if k == key:
                return v

    def get_string(self, obj: Int, key: String) raises -> String:
        """obj[key] when it is a string, else empty."""
        var v = self.get(obj, key)
        return self.string(v) if v >= 0 and self.kind(v) == QUOTE else String()
