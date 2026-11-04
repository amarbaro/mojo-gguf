"""Safetensors checkpoint directory: every shard mmapped read-only, tensors in the order the
Hugging Face converter visits them (shards sorted by file name, tensors sorted by name inside a
shard), so a streaming consumer reproduces its tensor order. Nothing is copied: a tensor is
(shard, absolute offset, byte count)."""
from std.ffi import external_call
from std.os import listdir
from std.os.path import exists
from gguf.model import open_fd, pread_into
from .json import Json

comptime ST_F32 = 0
comptime ST_F16 = 1
comptime ST_BF16 = 2

@fieldwise_init
struct StTensor(Copyable, Movable):
    var name: String
    var dtype: Int
    var shape: List[Int]
    var shard: Int
    var begin: Int  # absolute file offset
    var nbytes: Int

    def numel(self) -> Int:
        var n = 1
        for d in self.shape:
            n *= d
        return n

struct Shard(Copyable, Movable):
    var path: String
    var base: Int  # address of the read-only mapping of the whole file
    var size: Int

    def __init__(out self, path: String) raises:
        var fd = open_fd(path, 0)
        self.path = path.copy()
        self.size = Int(external_call["lseek", Int](fd, Int64(0), Int32(2)))
        # PROT_READ, MAP_PRIVATE
        self.base = external_call["mmap", Int](Int(0), self.size, Int32(1), Int32(2), fd, Int64(0))
        _ = external_call["close", Int32](fd)
        if self.base == -1:
            raise Error("cannot mmap " + path)

def dtype_code(s: String) raises -> Int:
    if s == "BF16":
        return ST_BF16
    if s == "F16":
        return ST_F16
    if s == "F32":
        return ST_F32
    raise Error("unsupported safetensors dtype " + s + " (BF16, F16, F32 only)")

def check_tensor(path: String, name: String, dtype: Int, shape: List[Int], lo: Int, hi: Int, data_start: Int, size: Int) raises:
    """Offsets inside the data, in order, and exactly shape * element size bytes (no wrap): what the converters read unchecked."""
    var esz = 4 if dtype == ST_F32 else 2
    var n = 1
    for d in shape:
        if d < 0 or (d != 0 and n > Int.MAX // d // esz):
            raise Error(path + ": tensor " + name + " has a shape out of range")
        n *= d
    if lo < 0 or hi < lo or hi > size - data_start:
        raise Error(path + ": tensor " + name + " has data_offsets outside the file")
    if n * esz != hi - lo:
        raise Error(path + ": tensor " + name + " shape does not match its data_offsets")

def _sort_in(mut names: List[String], s: String):
    var at = len(names)
    for k in range(len(names)):
        if s < names[k]:
            at = k
            break
    names.insert(at, s.copy())

struct Checkpoint(Movable):
    var shards: List[Shard]
    var tensors: List[StTensor]

    def __init__(out self, dir: String) raises:
        self.shards = List[Shard]()
        self.tensors = List[StTensor]()
        var parts = List[String]()
        var index = dir + "/model.safetensors.index.json"
        var from_index = List[String]()
        if exists(index):
            var j = Json(index)
            var wm = j.get(0, "weight_map")
            if wm < 0:
                raise Error("model.safetensors.index.json has no weight_map")
            var cur = j.open(wm)
            var key = String()
            while True:
                var v = j.member(cur, key)
                if v < 0:
                    break
                from_index.append(key.copy())
                var f = j.string(v)
                var seen = False
                for p in parts:
                    seen = seen or p == f
                if not seen:
                    _sort_in(parts, f)
        else:
            for f in listdir(dir):
                if f.startswith("model") and f.endswith(".safetensors"):
                    _sort_in(parts, f)
        if len(parts) == 0:
            raise Error("no model*.safetensors in " + dir)
        for pi in range(len(parts)):
            var shard = Shard(dir + "/" + parts[pi])
            var fd = open_fd(shard.path, 0)
            var hdr = List[UInt8](length=8, fill=0)
            pread_into(fd, hdr, 0, 8, 0)
            var n = 0
            for k in range(8):
                n |= Int(hdr[k]) << (8 * k)
            if n <= 0 or 8 + n > shard.size:
                raise Error(shard.path + ": bad safetensors header length " + String(n))
            var buf = List[UInt8](length=n, fill=0)
            pread_into(fd, buf, 0, n, 8)
            _ = external_call["close", Int32](fd)
            var j = Json(buf^)
            var cur = j.open(0)
            var name = String()
            var mine = List[StTensor]()
            while True:
                var v = j.member(cur, name)
                if v < 0:
                    break
                if name == "__metadata__":
                    continue
                var shape = List[Int]()
                var sc = j.open(j.get(v, "shape"))
                while True:
                    var e = j.element(sc)
                    if e < 0:
                        break
                    shape.append(j.integer(e))
                var off = j.open(j.get(v, "data_offsets"))
                var lo = j.integer(j.element(off))
                var hi = j.integer(j.element(off))
                var dt = dtype_code(j.get_string(v, "dtype"))
                check_tensor(shard.path, name, dt, shape, lo, hi, 8 + n, shard.size)
                var t = StTensor(name.copy(), dt, shape^, pi, 8 + n + lo, hi - lo)
                var at = len(mine)
                for k in range(len(mine)):
                    if t.name < mine[k].name:
                        at = k
                        break
                mine.insert(at, t^)
            for t in mine:
                self.tensors.append(t.copy())
            self.shards.append(shard^)
        if len(from_index) > 0 and len(from_index) != len(self.tensors):
            raise Error("weight_map names " + String(len(from_index)) + " tensors, the shards hold " + String(len(self.tensors)))
