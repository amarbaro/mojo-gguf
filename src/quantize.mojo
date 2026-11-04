"""Quantize --pure IN OUT TYPE [NTHREADS]: llama-quantize's pure mode, byte for byte.

One type for every weight tensor (with llama-quantize's block-size fallbacks), written in the
loader's tensor order. Tensor data is streamed in slabs of rows through two fixed buffers and
quantized on all cores; norm and bias tensors are copied.
"""
from std.ffi import external_call, c_ssize_t
from std.memory import bitcast
from std.sys import argv, exit, num_logical_cores
from gguf.par import parallelize
from gguf import Gguf
from gguf.model import Out, T_STRING, pad_to, pread_into, refuse_same_file
from gguf.quants import (FP, BP, T_F32, T_F16, T_BF16, T_Q4_0, T_Q5_0, T_Q8_0, T_Q2_K, T_Q4_K, T_Q6_K,
                         T_TQ1_0, T_TQ2_0, T_Q2_0, QK_K, blck, row_size, quantize_row)

comptime SLAB = 16 << 20
comptime QUANT_VERSION = 2
comptime DROPPED_KEYS = ["general.quantization_version", "general.file_type", "split.no", "split.count",
                         "split.tensors.count"]
comptime NOT_WEIGHTS = ["_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "altup", "laurel",
                        "per_layer_model_proj", "ssm_conv1d", "shortconv.conv.weight", "indexer.k_proj.weight",
                        "indexer.q_proj.weight", "time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight",
                        "time_mix_w2.weight", "time_mix_v0.weight", "time_mix_v1.weight", "time_mix_v2.weight",
                        "time_mix_a0.weight", "time_mix_a1.weight", "time_mix_a2.weight", "time_mix_g1.weight",
                        "time_mix_g2.weight", "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
                        "time_mix_lerp_fused.weight", "attn_rel_b.weight", ".position_embd", "sam.pos_embd",
                        "sam.neck.", "sam.net_", ".rel_pos", ".patch_embd", ".patch_merger", "a.rvq.codebook",
                        "mm.a.code_embd"]

def parse_type(name: String) raises -> Tuple[Int, Int]:
    """(llama_ftype, ggml type) for a llama-quantize type name this tool implements."""
    if name == "Q8_0":
        return (7, T_Q8_0)
    if name == "Q4_0":
        return (2, T_Q4_0)
    if name == "Q4_K" or name == "Q4_K_M":
        return (15, T_Q4_K)
    if name == "Q6_K":
        return (18, T_Q6_K)
    if name == "Q2_K":
        return (10, T_Q2_K)
    if name == "Q2_0":
        return (41, T_Q2_0)
    if name == "TQ1_0":
        return (36, T_TQ1_0)
    if name == "TQ2_0":
        return (37, T_TQ2_0)
    raise Error("unsupported type " + name)

def fallback(target: Int, ne0: Int) -> Int:
    """The demotion llama-quantize applies when a row is not a whole number of blocks."""
    if ne0 % blck(target) == 0:
        return target
    var t = target
    if target == T_Q4_K:
        t = T_Q5_0
    elif target == T_Q6_K:
        t = T_Q8_0
    elif target == T_Q2_0 or target == T_Q2_K or target == T_TQ1_0 or target == T_TQ2_0:
        t = T_Q4_0
    return t if ne0 % blck(t) == 0 else T_F16

def allows_quantization(name: String, n_dims: Int) -> Bool:
    if n_dims < 2 or not name.endswith("weight"):
        return False
    if name == "position_embd.weight" or name == "token_types.weight":
        return False
    for s in materialize[NOT_WEIGHTS]():
        if s in name:
            return False
    return True

def layer_of(name: String) -> Int:
    """`blk.N.` layer number as sscanf reads it, -1 for tensors outside any block."""
    if not name.startswith("blk."):
        return -1
    var b = name.as_bytes()
    var i = 4
    var v = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        v = v * 10 + Int(b[i]) - 48
        i += 1
    return v if i > 4 else -1

struct Plan(Copyable, Movable):
    var src_type: Int
    var dst_type: Int
    var ne0: Int
    var nrows: Int
    var src_off: Int  # absolute file offset
    var dst_off: Int  # relative to the output data section
    var dst_bytes: Int

    def __init__(out self, src_type: Int, dst_type: Int, ne0: Int, nrows: Int, src_off: Int):
        self.src_type = src_type
        self.dst_type = dst_type
        self.ne0 = ne0
        self.nrows = nrows
        self.src_off = src_off
        self.dst_off = 0
        self.dst_bytes = nrows * row_size(dst_type, ne0)

def pwrite_all(fd: Int32, buf: List[UInt8], n: Int, off: Int) raises:
    var done = 0
    while done < n:
        var r = external_call["pwrite", c_ssize_t](fd, buf.unsafe_ptr().unsafe_offset(done), n - done, Int64(off + done))
        if r <= 0:
            raise Error("write failed at byte " + String(off + done))
        done += Int(r)

def load_row(t: Int, src: BP, at: Int, ne0: Int, dst: FP):
    """Row of ne0 elements of float type t at byte `at` of src, widened to f32."""
    if t == T_F32:
        for i in range(ne0):
            var b = at + 4 * i
            var v = Int(src[unsafe_offset=b]) | (Int(src[unsafe_offset=b + 1]) << 8) | (
                Int(src[unsafe_offset=b + 2]) << 16) | (Int(src[unsafe_offset=b + 3]) << 24)
            dst[unsafe_offset=i] = bitcast[DType.float32](UInt32(v))
    elif t == T_BF16:
        for i in range(ne0):
            var b = at + 2 * i
            var v = (Int(src[unsafe_offset=b]) << 16) | (Int(src[unsafe_offset=b + 1]) << 24)
            dst[unsafe_offset=i] = bitcast[DType.float32](UInt32(v))
    else:
        for i in range(ne0):
            var b = at + 2 * i
            var v = Int(src[unsafe_offset=b]) | (Int(src[unsafe_offset=b + 1]) << 8)
            dst[unsafe_offset=i] = Float32(bitcast[DType.float16](UInt16(v)))

@fieldwise_init
struct _Rows:
    var src: Int
    var dst: Int
    var st: Int
    var dt: Int
    var ne0: Int
    var nr: Int
    var items: Int
    var src_row: Int
    var dst_row: Int

def _quant_task(ctx: Int, w: Int):
    var c = Pointer[_Rows, MutUntrackedOrigin](unsafe_from_address=ctx)
    var rowf = List[Float32](length=c[].ne0, fill=0)
    var sb = List[UInt8](length=QK_K, fill=0)
    var rp = rowf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var bp = sb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var sp = BP(unsafe_from_address=c[].src)
    var dp = BP(unsafe_from_address=c[].dst)
    for r in range(w * c[].nr // c[].items, (w + 1) * c[].nr // c[].items):
        load_row(c[].st, sp, r * c[].src_row, c[].ne0, rp)
        quantize_row(c[].dt, rp, dp.unsafe_offset(r * c[].dst_row), c[].ne0, bp)
    _ = len(rowf) + len(sb)

def quantize_tensor(fd: Int32, ofd: Int32, p: Plan, out_at: Int, nthreads: Int) raises:
    var src_row = row_size(p.src_type, p.ne0)
    var dst_row = row_size(p.dst_type, p.ne0)
    var step = max(1, SLAB // max(src_row, dst_row))
    var sbuf = List[UInt8](length=step * src_row, fill=0)
    var dbuf = List[UInt8](length=step * dst_row, fill=0)
    var sp = sbuf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var dp = dbuf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var ne0 = p.ne0
    var st = p.src_type
    var dt = p.dst_type
    for r0 in range(0, p.nrows, step):
        var nr = min(step, p.nrows - r0)
        pread_into(fd, sbuf, 0, nr * src_row, p.src_off + r0 * src_row)
        var items = min(nr, nthreads * 8)
        var c = _Rows(Int(sp), Int(dp), st, dt, ne0, nr, items, src_row, dst_row)
        parallelize(_quant_task, Int(Pointer(to=c)), items, nthreads)
        _ = c.items
        pwrite_all(ofd, dbuf, nr * dst_row, out_at + r0 * dst_row)

def copy_tensor(fd: Int32, ofd: Int32, src_off: Int, n: Int, out_at: Int) raises:
    var buf = List[UInt8](length=min(n, SLAB), fill=0)
    var done = 0
    while done < n:
        var k = min(len(buf), n - done)
        pread_into(fd, buf, 0, k, src_off + done)
        pwrite_all(ofd, buf, k, out_at + done)
        done += k

def run() raises:
    var a = argv()
    var pure = False
    var pos = List[String]()
    for i in range(1, len(a)):
        if String(a[i]) == "--pure":
            pure = True
        else:
            pos.append(String(a[i]))
    if not pure or len(pos) < 3:
        raise Error("usage: quantize --pure IN OUT TYPE [NTHREADS] (only --pure is implemented)")
    var nthreads = Int(pos[3]) if len(pos) > 3 else 0
    if nthreads < 1:  # llama-quantize: 0 or less means all cores
        nthreads = num_logical_cores()
    refuse_same_file(pos[0], pos[1])
    var ft = parse_type(pos[2])
    var g = Gguf(pos[0])
    var n = len(g.tensors)
    var plans = List[Plan]()
    for t in range(n):
        ref info = g.tensors[t]
        var n_dims = len(info.dims)
        while n_dims > 1 and info.dims[n_dims - 1] == 1:
            n_dims -= 1
        var nrows = 1
        for d in range(1, len(info.dims)):
            nrows *= info.dims[d]
        var ne0 = info.dims[0]
        var dst = info.ttype
        if allows_quantization(info.name, n_dims):
            dst = fallback(ft[1], ne0)
            var is_float = info.ttype == T_F32 or info.ttype == T_F16 or info.ttype == T_BF16
            if dst != info.ttype and not is_float:
                raise Error("requantizing tensor " + info.name + " from type " + String(info.ttype) + " is not supported")
        elif blck(info.ttype) == 0:
            raise Error("unsupported type " + String(info.ttype) + " in tensor " + info.name)
        plans.append(Plan(info.ttype, dst, ne0, nrows, g.data_offset + info.offset))
    # the loader's order: by layer number (non-block tensors first), then by name
    var order = List[Int]()
    for t in range(n):
        order.append(t)
    for i in range(1, n):
        var cur = order[i]
        var j = i - 1
        while j >= 0:
            var o = order[j]
            var la = layer_of(g.tensors[o].name)
            var lb = layer_of(g.tensors[cur].name)
            if la < lb or (la == lb and g.tensors[o].name < g.tensors[cur].name):
                break
            order[j + 1] = o
            j -= 1
        order[j + 1] = cur
    var running = 0
    for k in order:
        plans[k].dst_off = running
        running += pad_to(plans[k].dst_bytes, g.alignment)

    var out = Out(pos[1])
    var kept = List[Int]()
    for k in range(len(g.kvs)):
        var drop = False
        for key in materialize[DROPPED_KEYS]():
            drop = drop or g.kvs[k].key == key
        if not drop:
            kept.append(k)
    out.put(0x46554747, 4)
    out.put(3, 4)
    out.put(n, 8)
    out.put(len(kept) + 2, 8)
    for k in kept:
        ref kv = g.kvs[k]
        out.put_str(kv.key)
        out.put(kv.vtype, 4)
        out.copy(g.fd, kv.start, kv.end - kv.start)
    out.put_str("general.quantization_version")
    out.put(4, 4)
    out.put(QUANT_VERSION, 4)
    out.put_str("general.file_type")
    out.put(4, 4)
    out.put(ft[0], 4)
    for k in order:
        ref info = g.tensors[k]
        out.put_str(info.name)
        out.put(len(info.dims), 4)
        for d in info.dims:
            out.put(d, 8)
        out.put(plans[k].dst_type, 4)
        out.put(plans[k].dst_off, 8)
    out.pad(g.alignment)
    out.flush()
    var base = out.off
    for k in order:
        ref p = plans[k]
        if p.src_type == p.dst_type:
            copy_tensor(g.fd, out.fd, p.src_off, p.dst_bytes, base + p.dst_off)
        else:
            quantize_tensor(g.fd, out.fd, p, base + p.dst_off, nthreads)
        var pad = pad_to(p.dst_bytes, g.alignment) - p.dst_bytes
        if pad > 0:
            pwrite_all(out.fd, List[UInt8](length=pad, fill=0), pad, base + p.dst_off + p.dst_bytes)
    out.close()

def main():
    try:
        run()
    except e:
        print("FAIL quantize:", e)
        exit(1)
