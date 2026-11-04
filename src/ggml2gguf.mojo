"""Ggml2gguf --input IN --output OUT [--name S] [--desc S] [--gqa N] [--eps E] [--context-length N]: a legacy
GGML file (lmgg, fmgg v1, tjgg v1 to v3) as GGUF v3, byte-identical to `convert_llama_ggml_to_gguf.py` with the
same argv (pinned llama.cpp 1744c6bde). Every input the script refuses is refused with FAIL and exit 1, before
the output file exists. --model-metadata-dir and --vocab-dir print "not supported" and exit 2.

Kept on purpose because the script writes them: the stored dims are the legacy dims with the first two swapped,
then the last one divided by the type's element size (gguf-py quant_shape_from_byte_shape applied to the byte
array, F32 and F16 included), and the stored byte count is the whole legacy tensor.

Not ported: a tensor whose data runs past the end of the input (the script writes the short slice; this fails).
"""
from std.sys import argv, exit
from std.ffi import external_call
from std.memory import bitcast
from gguf.model import Out, open_fd, pread_into, pad_to, refuse_same_file
from gguf.quant import row, tensor_bytes
from gguf.tensor_lookup import map_tensor_name

comptime MAGIC = 0x46554747
comptime ALIGN = 32
comptime N_KV = 20  # architecture, name, description, file_type, 8 hparams, 8 tokenizer keys
comptime GGML = 0
comptime GGMF = 1
comptime GGJT = 2
comptime T_U32 = 4
comptime T_I32 = 5
comptime T_F32 = 6
comptime T_STR = 8
comptime T_ARR = 9
comptime TOK_NORMAL = 1
comptime TOK_UNK = 2
comptime TOK_CONTROL = 3
comptime TOK_BYTE = 6


def ftype_name(t: Int) -> String:
    """GGMLFType member name for t; empty when the enum has no such value."""
    var names: List[String] = [
        "ALL_F32", "MOSTLY_F16", "MOSTLY_Q4_0", "MOSTLY_Q4_1", "MOSTLY_Q4_1_SOME_F16", "", "",
        "MOSTLY_Q8_0", "MOSTLY_Q5_0", "MOSTLY_Q5_1", "MOSTLY_Q2_K", "MOSTLY_Q3_K_S", "MOSTLY_Q3_K_M",
        "MOSTLY_Q3_K_L", "MOSTLY_Q4_K_S", "MOSTLY_Q4_K_M", "MOSTLY_Q5_K_S", "MOSTLY_Q5_K_M", "MOSTLY_Q6_K",
    ]
    if t < 0 or t >= len(names):
        return ""
    return names[t]


def fmt_name(fmt: Int) -> String:
    if fmt == GGML:
        return "GGML"
    if fmt == GGMF:
        return "GGMF"
    return "GGJT"


def check_conversion(fmt: Int, ver: Int, ftype: Int) raises:
    """Legacy Quantizations are refused where the script refuses them (validate_conversion)."""
    var bad = False
    if fmt < GGJT or ver < 2:
        bad = ftype != 0 and ftype != 1
    elif fmt == GGJT and ver == 2:
        bad = ftype == 2 or ftype == 3 or ftype == 4 or ftype == 7
    if bad:
        raise Error("quantization " + ftype_name(ftype) + " is not eligible for conversion from " + fmt_name(fmt) + "v" + String(ver))


def basename(p: String) -> String:
    var b = p.as_bytes()
    var cut = 0
    for i in range(len(b)):
        if b[i] == 47:
            cut = i + 1
    return String(unsafe_from_utf8=b[cut : len(b)])


def gguf_name(hf: String, n_layer: Int) raises -> String:
    """Gguf-py TensorNameMap.get_name(hf, try_suffixes=(".weight", ".bias")) for llama; empty when unmapped."""
    var r = map_tensor_name("llama", n_layer, hf)
    if r.byte_length() > 0:
        return r
    for suffix in [".weight", ".bias"]:
        if hf.endswith(suffix):
            var base = String(unsafe_from_utf8=Span(hf.as_bytes())[0 : hf.byte_length() - suffix.byte_length()])
            var g = map_tensor_name("llama", n_layer, base)
            if g.byte_length() > 0:
                return g + suffix
    return ""


def bytes_of(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    for b in s.as_bytes():
        out.append(b)
    return out^


def hex_upper(v: Int) -> UInt8:
    return UInt8(48 + v) if v < 10 else UInt8(55 + v)


@fieldwise_init
struct Cfg(Copyable, Movable):
    var input: String
    var output: String
    var name: String
    var has_name: Bool
    var desc: String
    var has_desc: Bool
    var gqa: Int
    var eps: String
    var ctx: Int


struct Src(Movable):
    """The input file read in exact-size pieces: the header and vocab are small, tensor data is copied by fd."""
    var fd: Int32
    var size: Int
    var pos: Int

    def __init__(out self, path: String) raises:
        self.fd = open_fd(path, 0)
        self.size = Int(external_call["lseek", Int](self.fd, Int64(0), Int32(2)))
        self.pos = 0

    def __deinit__(deinit self):
        _ = external_call["close", Int32](self.fd)

    def take(mut self, n: Int) raises -> List[UInt8]:
        if n < 0 or self.pos + n > self.size:
            raise Error("truncated input: " + String(n) + " bytes at " + String(self.pos))
        var buf = List[UInt8](length=n, fill=0)
        pread_into(self.fd, buf, 0, n, self.pos)
        self.pos += n
        return buf^

    def uint(mut self, n: Int) raises -> Int:
        var b = self.take(n)
        var v = 0
        for k in range(n):
            v |= Int(b[k]) << (8 * k)
        return v

    def skip(mut self, n: Int) raises:
        _ = self.take(n)


@fieldwise_init
struct Tn(Copyable, Movable):
    var name: String  # legacy name, then the GGUF name
    var dims: List[Int]  # legacy dims, then the shape gguf-py writes
    var dtype: Int
    var start: Int  # tensor data offset in the input
    var nbytes: Int


@fieldwise_init
struct Legacy(Movable):
    var fmt: Int
    var ver: Int
    var ftype: Int
    var n_embd: Int
    var n_head: Int
    var n_layer: Int
    var n_ff: Int
    var tokens: List[List[UInt8]]
    var scores: List[Int]  # float32 bit patterns, 0 for GGML (no scores)
    var tensors: List[Tn]


def read_legacy(mut src: Src) raises -> Legacy:
    var magic = src.take(4)
    var m = String(unsafe_from_utf8=Span(magic))
    if m == "GGUF":
        raise Error("File is already in GGUF format.")
    var fmt = GGML
    var ver = 1
    if m == "fmgg":
        ver = src.uint(4)
        if ver != 1:
            raise Error("Cannot handle unexpected GGMF file version " + String(ver))
        fmt = GGMF
    elif m == "tjgg":
        ver = src.uint(4)
        if ver < 1 or ver > 3:
            raise Error("Cannot handle unexpected GGJT file version " + String(ver))
        fmt = GGJT
    elif m != "lmgg":
        raise Error("Unexpected file magic " + m + "! This doesn't look like a GGML format file.")

    var n_vocab = src.uint(4)
    var n_embd = src.uint(4)
    _ = src.uint(4)  # n_mult
    var n_head = src.uint(4)
    var n_layer = src.uint(4)
    _ = src.uint(4)  # n_rot
    var ftype = src.uint(4)
    if ftype_name(ftype) == "":
        raise Error("Invalid ftype " + String(ftype))
    check_conversion(fmt, ver, ftype)

    var tokens = List[List[UInt8]]()
    var scores = List[Int]()
    for _ in range(n_vocab):
        var n = src.uint(4)
        if n >= 4096:
            raise Error("Absurd vocab item length")
        tokens.append(src.take(n))
        if fmt == GGML:
            scores.append(0)
        else:
            scores.append(src.uint(4))

    var tensors = List[Tn]()
    while src.pos < src.size:
        var n_dims = src.uint(4)
        var name_len = src.uint(4)
        var dtype = src.uint(4)
        if n_dims > 4:
            raise Error("Invalid tensor dimensions " + String(n_dims))
        if name_len >= 4096:
            raise Error("Absurd tensor name length")
        _ = row(dtype)
        var dims = List[Int]()
        var n_elems = 1
        for _ in range(n_dims):
            var d = src.uint(4)
            dims.append(d)
            n_elems *= d
        var name_bytes = src.take(name_len)
        var name = String(unsafe_from_utf8=Span(name_bytes))
        if fmt == GGJT:
            src.skip(pad_to(src.pos, ALIGN) - src.pos)
        var nbytes = tensor_bytes(dtype, n_elems)
        var start = src.pos
        src.skip(nbytes)
        tensors.append(Tn(name^, dims^, dtype, start, nbytes))

    var ff = -1
    for t in range(len(tensors)):
        if tensors[t].name == "layers.0.feed_forward.w1.weight":
            ff = t
    if ff < 0:
        raise Error("Missing layer 0 FF tensor")
    if len(tensors[ff].dims) < 2:
        raise Error("layer 0 FF tensor has fewer than 2 dims")
    return Legacy(fmt, ver, ftype, n_embd, n_head, n_layer, tensors[ff].dims[1], tokens^, scores^, tensors^)


def kv_key(mut out: Out, key: String, t: Int) raises:
    out.put_str(key)
    out.put(t, 4)


def kv_str(mut out: Out, key: String, v: String) raises:
    kv_key(out, key, T_STR)
    out.put_str(v)


def kv_u32(mut out: Out, key: String, v: Int) raises:
    kv_key(out, key, T_U32)
    out.put(v, 4)


def emit(src: Src, cfg: Cfg, L: Legacy) raises:
    if L.n_head == 0:
        raise Error("n_head is 0: no rope dimension count")
    var n_kv_head = L.n_head
    if cfg.gqa != 1:
        n_kv_head = -1
        var gqa = Float64(cfg.gqa)
        for x in range(1, 256):
            if Float64(L.n_head) / Float64(x) == gqa:
                n_kv_head = x
        if n_kv_head < 0:
            raise Error("Couldn't determine n_kv_head from GQA param")
    if cfg.ctx < 0 or cfg.ctx > 4294967295:
        raise Error("context length " + String(cfg.ctx) + " does not fit uint32")
    var eps = Float64(cfg.eps)
    var eps32 = Float32(eps)
    if eps32 > 3.4e38 or eps32 < -3.4e38:
        raise Error("rms eps " + cfg.eps + " does not fit float32")
    var eps_bits = Int(bitcast[DType.uint32, 1](eps32))

    var plan = List[Tn]()
    for t in range(len(L.tensors)):
        ref x = L.tensors[t]
        var g = gguf_name(x.name, L.n_layer)
        if g.byte_length() == 0:
            raise Error("Bad name " + x.name)
        for k in range(len(plan)):
            if plan[k].name == g:
                raise Error("Duplicated tensor name " + g)
        var shape = x.dims.copy()
        if len(shape) == 0:
            raise Error("tensor " + x.name + " has no dims")
        if len(shape) > 1:
            var tmp = shape[1]
            shape[1] = shape[0]
            shape[0] = tmp
        var q = row(x.dtype)
        var last = shape[len(shape) - 1]
        if last % q.size != 0:
            raise Error("Quantized tensor bytes per row (" + String(last) + ") is not a multiple of " + q.name + " type size (" + String(q.size) + ")")
        shape[len(shape) - 1] = last // q.size * q.block
        plan.append(Tn(g^, shape^, x.dtype, x.start, x.nbytes))

    var name = cfg.name if cfg.has_name else basename(cfg.input)
    var desc = cfg.desc if cfg.has_desc else "converted from legacy " + fmt_name(L.fmt) + "v" + String(L.ver) + " " + ftype_name(L.ftype) + " format"
    # gguf-py add_string skips an empty value: an empty name or description is not written
    var n_kv = N_KV - Int(name.byte_length() == 0) - Int(desc.byte_length() == 0)
    var ar = "llama."
    refuse_same_file(cfg.input, cfg.output)
    var out = Out(cfg.output)
    out.put(MAGIC, 4)
    out.put(3, 4)
    out.put(len(plan), 8)
    out.put(n_kv, 8)
    kv_str(out, "general.architecture", "llama")
    if name.byte_length() > 0:
        kv_str(out, "general.name", name)
    if desc.byte_length() > 0:
        kv_str(out, "general.description", desc)
    kv_u32(out, "general.file_type", L.ftype)
    kv_u32(out, ar + "context_length", cfg.ctx)
    kv_u32(out, ar + "embedding_length", L.n_embd)
    kv_u32(out, ar + "block_count", L.n_layer)
    kv_u32(out, ar + "feed_forward_length", L.n_ff)
    kv_u32(out, ar + "rope.dimension_count", L.n_embd // L.n_head)
    kv_u32(out, ar + "attention.head_count", L.n_head)
    kv_u32(out, ar + "attention.head_count_kv", n_kv_head)
    kv_key(out, ar + "attention.layer_norm_rms_epsilon", T_F32)
    out.put(eps_bits, 4)

    kv_str(out, "tokenizer.ggml.model", "llama")
    kv_str(out, "tokenizer.ggml.pre", "default")
    var toks = List[List[UInt8]]()
    var types = List[Int]()
    for tokid in range(len(L.tokens)):
        var tb = L.tokens[tokid].copy()
        var tt = TOK_NORMAL
        if tokid <= 2:
            tt = TOK_UNK if tokid == 0 else TOK_CONTROL
            tb = bytes_of("<unk>" if tokid == 0 else ("<s>" if tokid == 1 else "</s>"))
        elif len(tb) == 0:
            tt = TOK_CONTROL
        elif tokid <= 258 and len(tb) == 1:
            tb = [60, 48, 120, hex_upper(Int(tb[0]) >> 4), hex_upper(Int(tb[0]) & 15), 62]
            tt = TOK_BYTE
        else:
            var sp = List[UInt8]()
            for b in tb:
                if b == 32:
                    sp.extend([0xE2, 0x96, 0x81])
                else:
                    sp.append(b)
            tb = sp^
        toks.append(tb^)
        types.append(tt)
    kv_key(out, "tokenizer.ggml.tokens", T_ARR)
    out.put(T_STR, 4)
    out.put(len(toks), 8)
    for tb in toks:
        out.put(len(tb), 8)
        for b in tb:
            out.put(Int(b), 1)
    kv_key(out, "tokenizer.ggml.scores", T_ARR)
    out.put(T_F32, 4)
    out.put(len(L.scores), 8)
    for bits in L.scores:
        out.put(bits, 4)
    kv_key(out, "tokenizer.ggml.token_type", T_ARR)
    out.put(T_I32, 4)
    out.put(len(types), 8)
    for tt in types:
        out.put(tt, 4)
    kv_u32(out, "tokenizer.ggml.unknown_token_id", 0)
    kv_u32(out, "tokenizer.ggml.bos_token_id", 1)
    kv_u32(out, "tokenizer.ggml.eos_token_id", 2)

    var off = 0
    for t in range(len(plan)):
        ref p = plan[t]
        out.put_str(p.name)
        out.put(len(p.dims), 4)
        for k in range(len(p.dims)):
            out.put(p.dims[len(p.dims) - 1 - k], 8)
        out.put(p.dtype, 4)
        out.put(off, 8)
        off += pad_to(p.nbytes, ALIGN)
    out.pad(ALIGN)
    for t in range(len(plan)):
        ref p = plan[t]
        out.copy(src.fd, p.start, p.nbytes)
        out.pad(ALIGN)
    out.close()
    print("wrote", cfg.output, len(plan), "tensors", n_kv, "kv", out.off, "bytes")


def usage(msg: String):
    print("usage: ggml2gguf --input IN --output OUT [--name S] [--desc S] [--gqa N] [--eps E] [--context-length N]")
    print("ggml2gguf: error:", msg)
    exit(2)


def parse_int(flag: String, v: String) -> Int:
    try:
        return Int(v)
    except:
        usage("argument " + flag + ": invalid int value: '" + v + "'")
        return 0


def run() raises:
    var a = argv()
    var cfg = Cfg(String(), String(), String(), False, String(), False, 1, String("5.0e-06"), 2048)
    var i = 1
    while i < len(a):
        var arg = String(a[i])
        if arg == "--verbose":
            i += 1
            continue
        var v = String()
        var eq = arg.find("=")
        if arg.startswith("--") and eq > 0:
            var b = arg.as_bytes()
            var head = String(unsafe_from_utf8=b[0 : eq])
            v = String(unsafe_from_utf8=b[eq + 1 : len(b)])
            arg = head^
        elif i + 1 < len(a):
            v = String(a[i + 1])
            i += 1
        else:
            usage("argument " + arg + ": expected one value")
        i += 1
        if arg == "--input" or arg == "-i":
            cfg.input = v
        elif arg == "--output" or arg == "-o":
            cfg.output = v
        elif arg == "--name":
            cfg.name = v
            cfg.has_name = True
        elif arg == "--desc":
            cfg.desc = v
            cfg.has_desc = True
        elif arg == "--gqa":
            cfg.gqa = parse_int(arg, v)
        elif arg == "--eps":
            cfg.eps = v
        elif arg == "--context-length" or arg == "-c":
            cfg.ctx = parse_int(arg, v)
        elif arg == "--vocabtype":
            pass
        elif arg == "--model-metadata-dir" or arg == "-m" or arg == "--vocab-dir":
            print("FAIL ggml2gguf:", arg, "not supported")
            exit(2)
        else:
            usage("unrecognized argument: " + arg)
    if cfg.input.byte_length() == 0 or cfg.output.byte_length() == 0:
        usage("the following arguments are required: --input/-i, --output/-o")
    var src = Src(cfg.input)
    var L = read_legacy(src)
    emit(src, cfg, L)


def main():
    try:
        run()
    except e:
        print("FAIL ggml2gguf:", e)
        exit(1)
