"""GGUF v3 little-endian reader and writer.

Nothing but small windows of the file is ever in memory: the reader walks the header through a
WINDOW-byte sliding read buffer and records, per KV, its key, type and file span (all 13 value types
and nested arrays are validated by the walk). `rewrite` streams KV payloads and tensor data from the
source through one CHUNK-byte staging buffer.
"""
from std.ffi import external_call, c_ssize_t
from std.os import stat
from std.os.path import exists

comptime MAGIC = 0x46554747
comptime WINDOW = 256 << 10
comptime CHUNK = 1 << 20
comptime T_STRING = 8
comptime T_ARRAY = 9

def fixed_size(t: Int) -> Int:
    """Byte size of a fixed-width value type; 0 for string, array and unknown."""
    if t == 0 or t == 1 or t == 7:
        return 1
    if t == 2 or t == 3:
        return 2
    if t == 4 or t == 5 or t == 6:
        return 4
    if t == 10 or t == 11 or t == 12:
        return 8
    return 0

def pad_to(n: Int, align: Int) -> Int:
    return (n + align - 1) // align * align

comptime CREATE = 1 | 0o100 | 0o1000  # O_WRONLY | O_CREAT | O_TRUNC

def open_fd(path: String, flags: Int) raises -> Int32:
    # open64 takes 2 arguments here, as in mojo-gguf-core and mojo-safetensors: one program cannot link two
    # signatures of one libc symbol. Creating a file is creat64 (open with CREATE, mode 0644).
    if flags & 0o100 and flags != CREATE:
        raise Error("open_fd: O_CREAT only as CREATE")
    var p = path.copy()
    var fd: Int32
    if flags == CREATE:
        fd = external_call["creat64", Int32](p.as_c_string_span().ptr(), Int32(0o644))
    else:
        fd = external_call["open64", Int32](p.as_c_string_span().ptr(), Int32(flags))
    if fd < 0:
        raise Error("cannot open " + path)
    return fd

def pread_into(fd: Int32, mut dst: List[UInt8], at: Int, n: Int, off: Int) raises:
    var got = 0
    while got < n:
        var r = external_call["pread", c_ssize_t](fd, dst.unsafe_ptr().unsafe_offset(at + got), n - got, Int64(off + got))
        if r <= 0:
            raise Error("short read at byte " + String(off + got))
        got += Int(r)

def refuse_same_file(inp: String, dst: String) raises:
    """Out truncates its file on open: an output that is the input (or a link to it) would erase it mid-read."""
    if exists(dst) and stat(inp).st_ino == stat(dst).st_ino and stat(inp).st_dev == stat(dst).st_dev:
        raise Error("output " + dst + " is the input file")


struct Out(Movable):
    """Output file behind one CHUNK-byte staging buffer; every write goes through `buf`."""
    var fd: Int32
    var off: Int  # file offset of buf[0]
    var used: Int
    var buf: List[UInt8]

    def __init__(out self, path: String) raises:
        self.fd = open_fd(path, CREATE)
        self.off = 0
        self.used = 0
        self.buf = List[UInt8](length=CHUNK, fill=0)

    def flush(mut self) raises:
        var done = 0
        while done < self.used:
            var r = external_call["pwrite", c_ssize_t](self.fd, self.buf.unsafe_ptr().unsafe_offset(done), self.used - done, Int64(self.off + done))
            if r <= 0:
                raise Error("write failed at byte " + String(self.off + done))
            done += Int(r)
        self.off += self.used
        self.used = 0

    def put(mut self, v: Int, width: Int) raises:
        if self.used + width > CHUNK:
            self.flush()
        for k in range(width):
            self.buf[self.used + k] = UInt8((v >> (8 * k)) & 0xFF)
        self.used += width

    def put_str(mut self, s: String) raises:
        self.put(s.byte_length(), 8)
        for b in s.as_bytes():
            self.put(Int(b), 1)

    def copy(mut self, src: Int32, at: Int, n: Int) raises:
        var left = n
        var pos = at
        while left > 0:
            if self.used == CHUNK:
                self.flush()
            var k = min(left, CHUNK - self.used)
            pread_into(src, self.buf, self.used, k, pos)
            self.used += k
            pos += k
            left -= k

    def pad(mut self, align: Int) raises:
        for _ in range(pad_to(self.off + self.used, align) - self.off - self.used):
            self.put(0, 1)

    def close(mut self) raises:
        self.flush()
        if external_call["close", Int32](self.fd) != 0:
            raise Error("close failed")

@fieldwise_init
struct KV(Copyable, Movable):
    var key: String
    var vtype: Int
    var start: Int  # value payload: file offsets [start, end)
    var end: Int
    var at: Int  # file offset of the key length (the record's offset)
    var types: List[Int]  # type chain as gguf-py keeps it: [ARRAY, ..., leaf], or [vtype]
    var count: Int  # leaf elements (gguf-py's len(field.data)): 1 for scalars and strings

@fieldwise_init
struct TensorInfo(Copyable, Movable):
    var name: String
    var dims: List[Int]
    var ttype: Int
    var offset: Int  # relative to the data section
    var at: Int  # file offset of the tensor info record

struct Gguf(Movable):
    var fd: Int32
    var fsize: Int
    var win: List[UInt8]
    var wbase: Int
    var wlen: Int
    var pos: Int
    var kvs: List[KV]
    var tensors: List[TensorInfo]
    var alignment: Int
    var data_offset: Int
    var big: Bool  # file written for big-endian hosts

    def __init__(out self, path: String) raises:
        self.fd = open_fd(path, 0)
        self.fsize = Int(external_call["lseek64", Int](self.fd, Int64(0), Int32(2)))
        self.win = List[UInt8](length=WINDOW, fill=0)
        self.wbase = 0
        self.wlen = 0
        self.pos = 0
        self.kvs = List[KV]()
        self.tensors = List[TensorInfo]()
        self.alignment = 32
        self.data_offset = 0
        self.big = False
        if self._int(4) != MAGIC:
            raise Error(path + " is not a GGUF file")
        var version = self._int(4)
        if (version & 0xFFFF) == 0:  # low half zero: the version was written in the other byte order
            self.big = True
            self.pos = 4
            version = self._int(4)
        if version != 3:
            raise Error("only GGUF v3 is supported, got v" + String(version))
        var n_tensors = self._int(8)
        var n_kv = self._int(8)
        for _ in range(n_kv):
            var at = self.pos
            var key = self._str()
            var vtype = self._int(4)
            var start = self.pos
            var types = List[Int]()
            var count = self._skip_value(vtype, types)
            self.kvs.append(KV(key^, vtype, start, self.pos, at, types^, count))
        var a = self.find("general.alignment")
        if a >= 0:
            if self.kvs[a].vtype != 4:
                raise Error("general.alignment is not uint32")
            var end = self.pos
            self.pos = self.kvs[a].start
            self.alignment = self._int(4)
            self.pos = end
        if self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise Error("alignment " + String(self.alignment) + " is not a power of two")
        for _ in range(n_tensors):
            var at = self.pos
            var name = self._str()
            var dims = List[Int]()
            for _ in range(self._int(4)):
                dims.append(self._int(8))
            var ttype = self._int(4)
            self.tensors.append(TensorInfo(name^, dims^, ttype, self._int(8), at))
        self.data_offset = pad_to(self.pos, self.alignment)
        var prev = 0
        for t in range(len(self.tensors)):
            var off = self.tensors[t].offset
            if off < prev or self.data_offset + off > self.fsize:
                raise Error("tensor " + self.tensors[t].name + ": offset " + String(off) + " out of order or past EOF")
            prev = off

    def __deinit__(deinit self):
        _ = external_call["close", Int32](self.fd)

    def _need(mut self, n: Int) raises:
        if n < 0 or self.pos + n > self.fsize:
            raise Error("truncated GGUF: need " + String(n) + " bytes at " + String(self.pos) + " of " + String(self.fsize))
        if self.pos >= self.wbase and self.pos + n <= self.wbase + self.wlen:
            return
        if n > len(self.win):
            self.win = List[UInt8](length=n, fill=0)
        self.wlen = min(len(self.win), self.fsize - self.pos)
        self.wbase = self.pos
        pread_into(self.fd, self.win, 0, self.wlen, self.pos)

    def _int(mut self, width: Int) raises -> Int:
        self._need(width)
        var v = 0
        for k in range(width):
            var sh = 8 * (width - 1 - k) if self.big else 8 * k
            v |= Int(self.win[self.pos - self.wbase + k]) << sh
        self.pos += width
        return v

    def _str(mut self) raises -> String:
        var n = self._int(8)
        self._need(n)
        var at = self.pos - self.wbase
        self.pos += n
        return String(from_utf8_lossy=Span(self.win)[at : at + n])

    def _skip(mut self, n: Int) raises:
        if n < 0 or self.pos + n > self.fsize:
            raise Error("truncated GGUF: value of " + String(n) + " bytes at " + String(self.pos))
        self.pos += n

    def _skip_value(mut self, vtype: Int, mut types: List[Int]) raises -> Int:
        """Skip one value payload. Records its type chain in `types` (the first element's chain for an
        array, as gguf-py does) and returns its leaf count."""
        types.append(vtype)
        var w = fixed_size(vtype)
        if w > 0:
            self._skip(w)
            return 1
        if vtype == T_STRING:
            self._skip(self._int(8))
            return 1
        if vtype == T_ARRAY:
            var et = self._int(4)
            var n = self._int(8)
            var ew = fixed_size(et)
            if et > 12:
                raise Error("unknown array element type " + String(et))
            if ew > 0:
                if n > self.fsize // ew:
                    raise Error("truncated GGUF: array of " + String(n))
                self._skip(n * ew)
                if n > 0:
                    types.append(et)
                return n
            var count = 0
            for i in range(n):
                if i == 0:
                    count += self._skip_value(et, types)
                else:
                    var scratch = List[Int]()
                    count += self._skip_value(et, scratch)
            return count
        raise Error("unknown value type " + String(vtype))

    def find(self, key: String) -> Int:
        for k in range(len(self.kvs)):
            if self.kvs[k].key == key:
                return k
        return -1

    def summary(self) -> String:
        return String("kv ", len(self.kvs), " tensors ", len(self.tensors), " alignment ", self.alignment,
                      " data_offset ", self.data_offset)

    def rewrite(self, out_path: String, name: String, remove: List[String]) raises:
        """Write the model to out_path. With no edit the layout is the source's; with an edit it is
        what gguf_new_metadata.py (GGUFWriter) produces: architecture KV first, `general.name`
        replaced in place or appended, removed keys dropped, no padding after an empty tensor list."""
        if self.big:
            raise Error("rewrite: big-endian input is not supported")
        var edited = name.byte_length() > 0 or len(remove) > 0
        var arch = self.find("general.architecture")
        if edited and arch < 0:
            raise Error("cannot edit: no general.architecture KV")
        var order = List[Int]()
        var has_name = False
        if edited:
            order.append(arch)
        for i in range(len(self.kvs)):
            if edited and (i == arch or self.kvs[i].key in remove):
                continue
            order.append(i)
        for i in order:
            has_name = has_name or self.kvs[i].key == "general.name"
        var rename = name.byte_length() > 0
        var out = Out(out_path)
        out.put(MAGIC, 4)
        out.put(3, 4)
        out.put(len(self.tensors), 8)
        out.put(len(order) + Int(rename and not has_name), 8)
        for i in order:
            ref kv = self.kvs[i]
            out.put_str(kv.key)
            out.put(T_STRING if rename and kv.key == "general.name" else kv.vtype, 4)
            if rename and kv.key == "general.name":
                out.put_str(name)
            else:
                out.copy(self.fd, kv.start, kv.end - kv.start)
        if rename and not has_name:
            out.put_str("general.name")
            out.put(T_STRING, 4)
            out.put_str(name)
        var total = self.fsize - self.data_offset
        var new_off = 0
        for t in range(len(self.tensors)):
            ref info = self.tensors[t]
            out.put_str(info.name)
            out.put(len(info.dims), 4)
            for d in info.dims:
                out.put(d, 8)
            out.put(info.ttype, 4)
            out.put(new_off, 8)
            var nxt = self.tensors[t + 1].offset if t + 1 < len(self.tensors) else total
            new_off += pad_to(nxt - info.offset, self.alignment)
        if len(self.tensors) > 0 or not edited:
            out.pad(self.alignment)
        for t in range(len(self.tensors)):
            var nxt = self.tensors[t + 1].offset if t + 1 < len(self.tensors) else total
            out.copy(self.fd, self.data_offset + self.tensors[t].offset, nxt - self.tensors[t].offset)
            out.pad(self.alignment)
        out.close()
