"""gguf-tool dump|hash|set-metadata|convert-endian ARGS

The four gguf-py scripts (gguf_dump.py, gguf_hash.py, gguf_set_metadata.py, gguf_convert_endian.py) as one CLI.
Each subcommand takes the script's argv and must match it: stdout byte for byte, exit code, and every byte a
command writes. Logging that the scripts send to stderr is not reproduced.
"""
from std.ffi import c_ssize_t, external_call
from std.io.file_descriptor import FileDescriptor
from std.memory import bitcast
from std.sys import argv, exit
from gguf.fmt import (
    cp_len, cp_slice, fixed, json_str, ljust, py_repr, py_title, rep, rjust, round_half_even, sub_str,
)
from gguf.model import Gguf, open_fd, pread_into, T_ARRAY, T_STRING, fixed_size
from gguf.quant import row, tensor_bytes
from gguf.sha import Sha1, Sha256, hexdigest
from gguf.values import M_JSON, M_NP, M_PY, emit, end_at, leaf, read_str, type_name, uint_at

comptime CHUNK = 1 << 20
comptime T_UINT64 = 10
comptime T_FLOAT32 = 6
comptime T_FLOAT64 = 12
comptime T_BF16 = 30
comptime T_F32 = 0
comptime T_F16 = 1


def say(s: String):
    print(s, end="")


def has(items: List[String], s: String) -> Bool:
    for x in items:
        if x == s:
            return True
    return False


def usage_error(cmd: String, msg: String):
    print("gguf-tool " + cmd + ": error: " + msg, file=FileDescriptor(2))
    exit(2)


struct Opts(Movable):
    var flags: List[String]
    var pos: List[String]

    def __init__(out self, var flags: List[String], var pos: List[String]):
        self.flags = flags^
        self.pos = pos^


def parse(cmd: String, rest: List[String], known: List[String], npos: Int) -> Opts:
    """argparse's job for these scripts: known flags anywhere, exactly npos positionals, else exit 2."""
    var flags = List[String]()
    var pos = List[String]()
    for a in rest:
        var negative_number = a.byte_length() > 1 and a.as_bytes()[1] >= 48 and a.as_bytes()[1] <= 57
        if a.startswith("-") and not negative_number:
            if not has(known, a):
                usage_error(cmd, "unrecognized arguments: " + a)
            flags.append(a)
        else:
            pos.append(a)
    if len(pos) != npos:
        usage_error(cmd, "expected " + String(npos) + " positional argument(s), got " + String(len(pos)))
    return Opts(flags^, pos^)


def n_elems(dims: List[Int]) -> Int:
    var n = 1
    for d in dims:
        n *= d
    return n


def dims4(dims: List[Int]) -> List[Int]:
    """list(shape) + [1] * (4 - len(shape)): the four dimensions gguf-py prints."""
    var d = dims.copy()
    while len(d) < 4:
        d.append(1)
    return d^


def hexs(v: Int) -> String:
    """'{:x}' of a non-negative int."""
    if v == 0:
        return "0"
    var digits = "0123456789abcdef"
    var s = String()
    var x = v
    while x > 0:
        s = String(digits[byte=x & 0xF]) + s
        x >>= 4
    return s


def fixed_hex16(v: Int) -> String:
    """'{:#16x}' of an int: 0x prefix, right-aligned to 16."""
    return rjust("0x" + hexs(v), 16)


def rounded(count: Int) -> String:
    """gguf-py's element_count_rounded_notation."""
    var x = Float64(count)
    var scaled = x
    var suffix = ""
    if x > 1e15:
        scaled = x * 1e-15
        suffix = "Q"
    elif x > 1e12:
        scaled = x * 1e-12
        suffix = "T"
    elif x > 1e9:
        scaled = x * 1e-9
        suffix = "B"
    elif x > 1e6:
        scaled = x * 1e-6
        suffix = "M"
    elif x > 1e3:
        scaled = x * 1e-3
        suffix = "K"
    return ("~" if count > 1000 else "") + String(round_half_even(scaled)) + suffix


def pretty_type(types: List[Int]) raises -> String:
    if types[0] == T_ARRAY:
        var nest = len(types) - 1
        return rep("[", nest) + type_name(types[len(types) - 1]) + rep("]", nest)
    return type_name(types[len(types) - 1])


struct Rec(Copyable, Movable):
    """One metadata field as gguf-py's reader sees it: the three header fields, then every KV."""
    var name: String
    var types: List[Int]
    var count: Int
    var at: Int  # record offset (gguf-py's field.offset)
    var start: Int  # value payload

    def __init__(out self, name: String, t: Int, count: Int, at: Int, start: Int):
        self.name = name
        self.types = List[Int]()
        self.types.append(t)
        self.count = count
        self.at = at
        self.start = start


def records(g: Gguf) -> List[Rec]:
    var result = List[Rec]()
    result.append(Rec("GGUF.version", 4, 1, 4, 4))
    result.append(Rec("GGUF.tensor_count", T_UINT64, 1, 8, 8))
    result.append(Rec("GGUF.kv_count", T_UINT64, 1, 16, 16))
    for kv in g.kvs:
        var r = Rec(kv.key, kv.vtype, kv.count, kv.at, kv.start)
        r.types = kv.types.copy()
        result.append(r^)
    return result^


def endian_name(g: Gguf) -> String:
    return "BIG" if g.big else "LITTLE"


def dump_text(g: Gguf, no_tensors: Bool) raises -> String:
    var recs = records(g)
    var s = "* File is " + endian_name(g) + " endian, script is running on a LITTLE endian host.\n"
    s += "* Dumping " + String(len(recs)) + " key/value pair(s)\n"
    for i in range(len(recs)):
        var r = recs[i].copy()
        var line = "  " + rjust(String(i + 1), 5) + ": " + ljust(pretty_type(r.types), 10)
        line += " | " + rjust(String(r.count), 8) + " | " + r.name
        var t = r.types[0]
        if t == T_STRING:
            var v = read_str(g, r.start)
            if cp_len(v) > 60:
                v = cp_slice(v, 0, 57) + "..."
            line += " = " + py_repr(v)
        elif t != T_ARRAY:
            line += " = " + leaf(g, r.start, t, M_PY)
        else:
            var items = List[String]()
            emit(g, r.start, T_ARRAY, M_PY, 6, items)
            var joined = ", ".join(items)
            line += " = [" + joined + (", ..." if r.count > 6 else "") + "]"
        s += line + "\n"
    if no_tensors:
        return s
    s += "* Dumping " + String(len(g.tensors)) + " tensor(s)\n"
    for i in range(len(g.tensors)):
        var t = g.tensors[i].copy()
        var dims = List[String]()
        for d in dims4(t.dims):
            dims.append(rjust(String(d), 5))
        s += "  " + rjust(String(i + 1), 5) + ": " + rjust(String(n_elems(t.dims)), 10)
        s += " | " + ", ".join(dims) + " | " + ljust(row(t.ttype).name, 7) + " | " + t.name + "\n"
    return s


def dump_json(g: Gguf, model: String, no_tensors: Bool, json_array: Bool) raises -> String:
    var recs = records(g)
    var s = '{"filename": ' + json_str(model) + ', "endian": "' + endian_name(g) + '", "metadata": {'
    for i in range(len(recs)):
        var r = recs[i].copy()
        if i > 0:
            s += ", "
        s += json_str(r.name) + ': {"index": ' + String(i) + ', "type": "' + type_name(r.types[0])
        s += '", "offset": ' + String(r.at)
        if r.types[0] == T_ARRAY:
            var names = List[String]()
            for k in range(1, len(r.types)):
                names.append('"' + type_name(r.types[k]) + '"')
            s += ', "array_types": [' + ", ".join(names) + "]"
            if json_array:
                var items = List[String]()
                emit(g, r.start, T_ARRAY, M_JSON, r.count, items)
                s += ', "value": [' + ", ".join(items) + "]"
        else:
            s += ', "value": ' + leaf(g, r.start, r.types[0], M_JSON)
        s += "}"
    s += '}, "tensors": {'
    if not no_tensors:
        for i in range(len(g.tensors)):
            var t = g.tensors[i].copy()
            if i > 0:
                s += ", "
            var shape = List[String]()
            for d in t.dims:
                shape.append(String(d))
            s += json_str(t.name) + ': {"index": ' + String(i) + ', "shape": [' + ", ".join(shape)
            s += '], "type": "' + row(t.ttype).name + '", "offset": ' + String(t.at) + "}"
    return s + "}}"


def esc(s: String) -> String:
    """gguf-py's escape_markdown_inline_code: a backtick fence one longer than the longest run."""
    var longest = 0
    var run = 0
    for c in s.as_bytes():
        if c == 96:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    var fence = rep("`", longest + 1)
    var body = s
    if s.startswith("`") or s.endswith("`"):
        body = " " + s + " "
    return fence + body + fence


def clip(s: String, half: Int) -> String:
    """head + '...' + tail when s is longer than 2 * half code points, else s escaped."""
    var n = cp_len(s)
    if n > 2 * half:
        return esc(cp_slice(s, 0, half)) + "..." + esc(cp_slice(s, n - half, n))
    return esc(s)


def md_value(g: Gguf, r: Rec) raises -> String:
    var t = r.types[0]
    if len(r.types) == 1:
        if t == T_STRING:
            return clip(read_str(g, r.start), 30)
        if t == T_ARRAY:
            return ""
        return leaf(g, r.start, t, M_NP)
    var items = List[String]()
    var sub = r.types[1]
    if sub == T_STRING:
        var raw = List[String]()
        emit(g, r.start, T_ARRAY, M_NP, 5, raw)
        for v in raw:
            items.append(clip(v, 15))
    elif sub != T_ARRAY:
        emit(g, r.start, T_ARRAY, M_NP, 7, items)
    return "[ " + ", ".join(items) + (", ..." if r.count > len(items) else "") + " ]"


def md_align(cell: String, w: Int, align: String) -> String:
    if align == "right":
        return rjust(cell, w - 1) + " "
    return " " + ljust(cell, w - 1)


def md_table(names: List[String], aligns: List[String], body: List[List[String]]) -> String:
    """gguf-py's markdown_table_with_alignment_support: column width from the widest cell, header included."""
    var widths = List[Int]()
    for j in range(len(names)):
        var pad = 0
        for r in body:
            pad = max(pad, cp_len(r[j]))
        var head = cp_len(names[j]) + 2
        widths.append(head if pad + 2 <= head else pad + 2)
    var lines = List[String]()
    var head_cells = List[String]()
    var dash_cells = List[String]()
    for j in range(len(names)):
        head_cells.append(md_align(names[j], widths[j], aligns[j]))
        var dashes = rep("-", widths[j] - 1)
        dash_cells.append(dashes + ":" if aligns[j] == "right" else ":" + dashes)
    lines.append("|" + "|".join(head_cells) + "|\n")
    lines.append("|" + "|".join(dash_cells) + "|\n")
    for r in body:
        var cells = List[String]()
        for j in range(len(names)):
            cells.append(md_align(r[j], widths[j], aligns[j]))
        lines.append("|" + "|".join(cells) + "|\n")
    return "".join(lines)


def split_dot(s: String) -> List[String]:
    """Python's str.split('.'), empty pieces kept."""
    var parts = List[String]()
    var cur = List[UInt8]()
    for c in s.as_bytes():
        if c == 46:
            parts.append(String(unsafe_from_utf8=Span(cur)))
            cur = List[UInt8]()
        else:
            cur.append(c)
    parts.append(String(unsafe_from_utf8=Span(cur)))
    return parts^


def lower_ascii(s: String) -> String:
    var b = List[UInt8]()
    for c in s.as_bytes():
        b.append(c + 32 if c >= 65 and c <= 90 else c)
    return String(unsafe_from_utf8=Span(b))


def abbreviation(word: String) -> String:
    """The dictionary in gguf-py's translate_tensor_name; empty when the word is not in it."""
    var keys: List[String] = [
        "token_embd", "pos_embd", "output_norm", "output", "attn_norm", "attn_norm_2", "attn_qkv", "attn_q", "attn_k",
        "attn_v", "attn_output", "ffn_norm", "ffn_up", "ffn_gate", "ffn_down", "ffn_gate_inp", "ffn_gate_exp",
        "ffn_down_exp", "ffn_up_exp", "ssm_in", "ssm_conv1d", "ssm_x", "ssm_a", "ssm_d", "ssm_dt", "ssm_out", "blk",
        "enc", "dec",
    ]
    var vals: List[String] = [
        "Token embedding", "Position embedding", "Output normalization", "Output", "Attention normalization",
        "Attention normalization", "Attention query-key-value", "Attention query", "Attention key", "Attention value",
        "Attention output", "Feed-forward network normalization", 'Feed-forward network "up"',
        'Feed-forward network "gate"', 'Feed-forward network "down"',
        "Expert-routing layer for the Feed-forward network in Mixture of Expert models",
        'Feed-forward network "gate" layer per expert in Mixture of Expert models',
        'Feed-forward network "down" layer per expert in Mixture of Expert models',
        'Feed-forward network "up" layer per expert in Mixture of Expert models',
        "State space model input projections", "State space model rolling/shift",
        "State space model selective parametrization", "State space model state compression",
        "State space model skip connection", "State space model time step", "State space model output projection",
        "Block", "Encoder", "Decoder",
    ]
    for i in range(len(keys)):
        if keys[i] == word:
            return vals[i]
    return ""


def translate(name: String) -> String:
    var words = List[String]()
    for w in split_dot(name):
        var found = abbreviation(lower_ascii(String(w.strip())))
        words.append(py_title(found if found != "" else w))
    return " ".join(words)


def group_of(name: String) raises -> String:
    """gguf-py's tensor group name for the markdown sections."""
    var p = split_dot(name)
    if p[0] == "blk":
        return p[0] + "." + p[1]
    if p[0] == "enc" or p[0] == "dec":
        if p[1] == "blk":
            return p[0] + "." + p[1] + "." + p[2]
        return p[0]
    return "base"


def dump_markdown(g: Gguf, model: String, no_tensors: Bool) raises -> String:
    var recs = records(g)
    var s = "# " + model + " - GGUF Internal File Dump\n\n"
    s += "- Endian: " + endian_name(g) + " endian\n\n"
    s += "## Key Value Metadata Store\n\n"
    s += "There are " + String(len(recs)) + " key-value pairs in this file\n\n"
    var kv_rows = List[List[String]]()
    for i in range(len(recs)):
        var r = recs[i].copy()
        var cells = List[String]()
        cells.append(String(i + 1))
        cells.append(pretty_type(r.types))
        cells.append(String(r.count))
        cells.append(r.name)
        cells.append(md_value(g, r))
        kv_rows.append(cells^)
    s += md_table(
        ["POS", "TYPE", "Count", "Key", "Value"],
        ["right", "left", "right", "left", "left"],
        kv_rows,
    )
    s += "\n"
    if no_tensors:
        return s + "Total BPW for " + model + ": undefined (no elements)"
    return s + dump_tensor_sections(g, model)


def dump_tensor_sections(g: Gguf, model: String) raises -> String:
    var total = 0
    var groups = List[String]()
    for t in g.tensors:
        total += n_elems(t.dims)
    var s = ""
    var by_group = List[List[Int]]()
    for i in range(len(g.tensors)):
        var grp = group_of(g.tensors[i].name)
        var idx = -1
        for k in range(len(groups)):
            if groups[k] == grp:
                idx = k
        if idx < 0:
            groups.append(grp)
            by_group.append(List[Int]())
            idx = len(groups) - 1
        by_group[idx].append(i)
    s += "## Tensors Overview " + rounded(total) + " Elements\n\n"
    s += "Total number of elements in all tensors: " + String(total) + " Elements\n\n"
    for k in range(len(groups)):
        var gel = 0
        for i in by_group[k]:
            gel += n_elems(g.tensors[i].dims)
        s += "- [" + translate(groups[k]) + " Tensor Group - " + rounded(gel) + " Elements](#"
        s += groups[k].replace(".", "_") + ")\n"
    s += "\n### Tensor Data Offset\n\n"
    s += "This table contains the offset and data segment relative to start of file\n\n"
    var map_rows = List[List[String]]()
    for i in range(len(g.tensors)):
        var t = g.tensors[i].copy()
        var cells = List[String]()
        cells.append(String(i))
        cells.append(t.name)
        cells.append(fixed_hex16(g.data_offset + t.offset))
        cells.append(fixed_hex16(tensor_bytes(t.ttype, n_elems(t.dims))))
        map_rows.append(cells^)
    s += md_table(
        ["T_ID", "Tensor Layer Name", "Data Offset (B)", "Data Size (B)"],
        ["right", "left", "right", "right"],
        map_rows,
    )
    s += "\n"
    var model_bytes = 0
    var model_elems = 0
    for k in range(len(groups)):
        var grp = groups[k]
        var gel = 0
        var gbytes = 0
        for i in by_group[k]:
            gel += n_elems(g.tensors[i].dims)
            gbytes += tensor_bytes(g.tensors[i].ttype, n_elems(g.tensors[i].dims))
        if total == 0:
            raise Error("percentage of total elements: division by zero")
        s += '### <a name="' + grp.replace(".", "_") + '">' + translate(grp) + " Tensor Group : "
        s += rounded(gel) + " Elements</a>\n\n"
        var est_w = 1
        var cnt_w = 1
        var dim_w = List[Int](length=4, fill=1)
        for i in by_group[k]:
            var t = g.tensors[i].copy()
            var n = n_elems(t.dims)
            est_w = max(est_w, cp_len(rounded(n)))
            cnt_w = max(cnt_w, String(n).byte_length())
            var d4 = dims4(t.dims)
            for j in range(4):
                dim_w[j] = max(dim_w[j], String(d4[j]).byte_length())
        var tbody = List[List[String]]()
        for i in by_group[k]:
            var t = g.tensors[i].copy()
            var n = n_elems(t.dims)
            var nb = tensor_bytes(t.ttype, n)
            var human = translate(t.name.replace(".weight", ".(W)").replace(".bias", ".(B)"))
            var d4 = dims4(t.dims)
            var shape = List[String]()
            for j in range(4):
                shape.append(rjust(String(d4[j]), dim_w[j]))
            var count = "(" + rjust(rounded(n), est_w) + ") " + rjust(String(n), cnt_w)
            var bpw = fixed(Float64(nb * 8) / Float64(n), 4) if n > 0 else "nan"
            var cells = List[String]()
            cells.append(String(i))
            cells.append(t.name)
            cells.append(human)
            cells.append(count)
            cells.append(" x ".join(shape))
            cells.append(row(t.ttype).name)
            cells.append(bpw)
            tbody.append(cells^)
        s += md_table(
            ["T_ID", "Tensor Layer Name", "Human Friendly Tensor Layer Name", "Elements", "Shape", "Type", "BPW"],
            ["right", "left", "left", "left", "left", "left", "right"],
            tbody,
        )
        s += "\n"
        s += "- Total elements in " + grp + ": (" + rjust(rounded(gel), 4) + ") " + String(gel) + "\n"
        var pct = Float64(gel) / Float64(total) * 100.0
        s += "- Percentage of total elements: " + fixed(pct, 2) + "%\n"
        if gel > 0:
            s += "- Bits per Weight (BPW) for " + grp + ": " + fixed(Float64(gbytes * 8) / Float64(gel), 4) + " bits\n"
        else:
            s += "- Bits per Weight (BPW) for " + grp + ": undefined (no elements)\n"
        s += "\n\n"
        model_bytes += gbytes
        model_elems += gel
    if model_elems > 0:
        return s + "Total BPW for " + model + ": " + fixed(Float64(model_bytes * 8) / Float64(model_elems), 4) + " bits"
    return s + "Total BPW for " + model + ": undefined (no elements)"


def dump(rest: List[String]) raises:
    var o = parse("dump", rest, ["--no-tensors", "--json", "--json-array", "--data-offset",
        "--data-alignment", "--markdown", "--verbose"], 1)
    var model = o.pos[0]
    var g = Gguf(model)
    var no_tensors = has(o.flags, "--no-tensors")
    if has(o.flags, "--json"):
        say(dump_json(g, model, no_tensors, has(o.flags, "--json-array")))
    elif has(o.flags, "--markdown"):
        print(dump_markdown(g, model, no_tensors))
    elif has(o.flags, "--data-offset"):
        print(g.data_offset)
    elif has(o.flags, "--data-alignment"):
        print(g.alignment)
    else:
        say(dump_text(g, no_tensors))


# ---- hash -----------------------------------------------------------------------------------------

def skip_hash(name: String) -> Bool:
    return name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".rotary_emb.inv_freq")


def read_chunk(fd: Int32, mut buf: List[UInt8], at: Int, n: Int) raises:
    if n > 0:
        pread_into(fd, buf, 0, n, at)


def hash_cmd(rest: List[String]) raises:
    var o = parse("hash", rest, ["--no-layer", "--verbose", "--progressbar"], 1)
    var model = o.pos[0]
    var no_layer = has(o.flags, "--no-layer")
    var g = Gguf(model)
    var all_sha1 = Sha1()
    var all_sha256 = Sha256()
    var uuid_sha1 = Sha1()
    var namespace: List[UInt8] = [0xEF, 0x00, 0x12, 0x06, 0xDA, 0xDC, 0x5F, 0x6D, 0xA1, 0x5F, 0x33, 0x59, 0xE5, 0x77, 0xD4, 0xE5]
    uuid_sha1.update(namespace, 16)
    var buf = List[UInt8](length=CHUNK, fill=0)
    for t in g.tensors:
        if skip_hash(t.name):
            continue
        var nb = tensor_bytes(t.ttype, n_elems(t.dims))
        var layer_sha1 = Sha1()
        var layer_sha256 = Sha256()
        var done = 0
        while done < nb:
            var m = min(CHUNK, nb - done)
            read_chunk(g.fd, buf, g.data_offset + t.offset + done, m)
            if not no_layer:
                layer_sha1.update(buf, m)
                layer_sha256.update(buf, m)
            all_sha1.update(buf, m)
            all_sha256.update(buf, m)
            uuid_sha1.update(buf, m)
            done += m
        if not no_layer:
            say("sha1      " + hexdigest(layer_sha1.digest()) + "  " + model + ":" + t.name + "\n")
            say("sha256    " + hexdigest(layer_sha256.digest()) + "  " + model + ":" + t.name + "\n")
    say("sha1      " + hexdigest(all_sha1.digest()) + "  " + model + "\n")
    say("sha256    " + hexdigest(all_sha256.digest()) + "  " + model + "\n")
    var u = uuid_sha1.digest()
    u[6] = (u[6] & 0x0F) | 0x50  # version 5
    u[8] = (u[8] & 0x3F) | 0x80  # RFC 4122 variant
    var h = hexdigest(u)
    var grouped = sub_str(h, 0, 8) + "-" + sub_str(h, 8, 12) + "-" + sub_str(h, 12, 16) + "-" + sub_str(h, 16, 20)
    say("uuid      " + grouped + "-" + sub_str(h, 20, 32) + "  " + model + "\n")


# ---- set-metadata ---------------------------------------------------------------------------------

def parse_int(s: String, bits: Int, signed: Bool) raises -> UInt64:
    """Python int(s) stored in an integer field: two's complement in `bits`, OverflowError outside the type."""
    var t = String(s.strip())
    var neg = t.startswith("-")
    if neg or t.startswith("+"):
        t = sub_str(t, 1, t.byte_length())
    if t.byte_length() == 0:
        raise Error("invalid integer literal " + s)
    var mag: UInt64 = 0
    for c in t.as_bytes():
        if c < 48 or c > 57:
            raise Error("invalid integer literal " + s)
        var d = UInt64(c - 48)
        if mag > (UInt64(0) - 1 - d) // 10:
            raise Error("Python int too large to convert: " + s)
        mag = mag * 10 + d
    var top: UInt64 = (UInt64(0) - 1) if bits == 64 else (UInt64(1) << UInt64(bits)) - 1
    if neg:
        if not signed and mag != 0:
            raise Error("OverflowError: negative value for an unsigned field: " + s)
        if signed and mag > (top >> 1) + 1:
            raise Error("OverflowError: value out of range for the field: " + s)
        return (UInt64(0) - mag) & top
    if mag > (top >> 1 if signed else top):
        raise Error("OverflowError: value out of range for the field: " + s)
    return mag


def set_metadata(rest: List[String]) raises:
    var o = parse("set-metadata", rest, ["--dry-run", "--force", "--verbose"], 3)
    var model = o.pos[0]
    var key = o.pos[1]
    var value = o.pos[2]
    var g = Gguf(model)
    var found = g.find(key)
    if found < 0:
        raise Error("field " + key + " not found")
    ref kv = g.kvs[found]
    var t = kv.vtype
    if t == T_ARRAY or t == T_STRING:
        raise Error("this tool only supports changing simple values, " + key + " has type " + type_name(t))
    var width = fixed_size(t)
    var new_bits: UInt64
    if t == 7:  # numpy: np.bool_(text) is True for any non-empty text
        new_bits = UInt64(1) if value.byte_length() > 0 else UInt64(0)
    elif t == T_FLOAT32:
        var f = Float32(Float64(value.strip()))
        new_bits = UInt64(bitcast[DType.uint32, 1](SIMD[DType.float32, 1](f))[0])
    elif t == T_FLOAT64:
        new_bits = bitcast[DType.uint64, 1](SIMD[DType.float64, 1](Float64(value.strip())))[0]
    else:
        var signed = t == 1 or t == 3 or t == 5 or t == 11
        new_bits = parse_int(value, 8 * width, signed)
    var cur = uint_at(g, kv.start, width)
    if cur == new_bits:
        return
    if has(o.flags, "--dry-run"):
        return
    if not has(o.flags, "--force"):
        var response = input("YES, I am sure> ")
        if response != "YES":
            return
    var fd = open_fd(model, 2)
    var b = List[UInt8](length=width, fill=0)
    for k in range(width):
        var byte = UInt8((new_bits >> UInt64(8 * k)) & 0xFF)
        b[width - 1 - k if g.big else k] = byte
    pwrite_all(fd, b, width, kv.start)
    close_fd(fd)


def pwrite_all(fd: Int32, mut buf: List[UInt8], n: Int, off: Int) raises:
    var done = 0
    while done < n:
        var r = external_call["pwrite", c_ssize_t](fd, buf.unsafe_ptr().unsafe_offset(done), n - done, Int64(off + done))
        if r <= 0:
            raise Error("write failed at byte " + String(off + done))
        done += Int(r)


def close_fd(fd: Int32) raises:
    if external_call["close", Int32](fd) != 0:
        raise Error("close failed")


# ---- convert-endian -------------------------------------------------------------------------------

def swap_units(fd: Int32, at: Int, count: Int, w: Int) raises:
    """Reverse every w-byte unit of `count` consecutive units at `at`, a chunk at a time."""
    var total = count * w
    var off = 0
    while off < total:
        var m = min(CHUNK, total - off)
        var buf = List[UInt8](length=m, fill=0)
        pread_into(fd, buf, 0, m, at + off)
        for i in range(0, m, w):
            for k in range(w // 2):
                var a = buf[i + k]
                buf[i + k] = buf[i + w - 1 - k]
                buf[i + w - 1 - k] = a
        pwrite_all(fd, buf, m, at + off)
        off += m


def swap_payload(g: Gguf, fd: Int32, at: Int, t: Int) raises:
    """Byte-swap one value payload in place, the way gguf-py's convert swaps every header part."""
    if t == T_STRING:
        swap_units(fd, at, 1, 8)
        return
    if t != T_ARRAY:
        swap_units(fd, at, 1, fixed_size(t))
        return
    var et = Int(uint_at(g, at, 4))
    var n = Int(uint_at(g, at + 4, 8))
    swap_units(fd, at, 1, 4)
    swap_units(fd, at + 4, 1, 8)
    var p = at + 12
    var ew = fixed_size(et)
    if ew > 0:
        swap_units(fd, p, n, ew)
        return
    for _ in range(n):
        var nxt = end_at(g, p, et)
        swap_payload(g, fd, p, et)
        p = nxt


def block_swap_offsets(t: Int) -> List[Int]:
    """Byte offsets of the f16 fields in one block of each quantized type (gguf-py's byteswap_* functions)."""
    if t == 2 or t == 8 or t == 41:  # Q4_0, Q8_0, Q1_0: delta at 0
        return [0]
    if t == 12:  # Q4_K: two f16 at 0 and 2
        return [0, 2]
    if t == 14:  # Q6_K: f16 at 208
        return [208]
    if t == 35:  # TQ2_0: f16 at 64
        return [64]
    return []


def swap_tensor(fd: Int32, at: Int, nbytes: Int, t: Int) raises:
    if t == T_F32:
        swap_units(fd, at, nbytes // 4, 4)
        return
    if t == T_F16 or t == T_BF16:
        swap_units(fd, at, nbytes // 2, 2)
        return
    var blk = row(t).size
    var fields = block_swap_offsets(t)
    if len(fields) == 0:
        return
    var blocks_per_chunk = CHUNK // blk
    var nblocks = nbytes // blk
    var done = 0
    while done < nblocks:
        var m = min(blocks_per_chunk, nblocks - done)
        var buf = List[UInt8](length=m * blk, fill=0)
        pread_into(fd, buf, 0, m * blk, at + done * blk)
        for b in range(m):
            for off in fields:
                var p = b * blk + off
                var a = buf[p]
                buf[p] = buf[p + 1]
                buf[p + 1] = a
        pwrite_all(fd, buf, m * blk, at + done * blk)
        done += m


def convert_endian(rest: List[String]) raises:
    var o = parse("convert-endian", rest, ["--dry-run", "--verbose"], 2)
    var model = o.pos[0]
    var order_arg = o.pos[1]
    if order_arg != "big" and order_arg != "little" and order_arg != "native":
        usage_error("convert-endian", "argument order: invalid choice: '" + order_arg + "'")
    var g = Gguf(model)
    var file_endian = endian_name(g)
    var order = "LITTLE" if order_arg == "native" else order_arg.upper()
    if file_endian == order:
        return
    for t in g.tensors:
        var tt = t.ttype
        if not (tt == T_F32 or tt == T_F16 or tt == T_BF16 or len(block_swap_offsets(tt)) > 0 or tt == 39 or tt == 40):
            raise Error("cannot handle type " + row(tt).name + " for tensor " + t.name)
    if has(o.flags, "--dry-run"):
        return
    var response = input("YES, I am sure> ")
    if response != "YES":
        return
    var fd = open_fd(model, 2)
    swap_units(fd, 4, 1, 4)  # GGUF.version
    swap_units(fd, 8, 1, 8)  # tensor count
    swap_units(fd, 16, 1, 8)  # kv count
    for kv in g.kvs:
        swap_units(fd, kv.at, 1, 8)  # key length
        swap_units(fd, kv.at + 8 + kv.key.byte_length(), 1, 4)  # value type
        swap_payload(g, fd, kv.start, kv.vtype)
    for ti in g.tensors:
        swap_units(fd, ti.at, 1, 8)  # name length
        var p = ti.at + 8 + ti.name.byte_length()
        swap_units(fd, p, 1, 4)  # n_dims
        swap_units(fd, p + 4, len(ti.dims), 8)  # dims
        p += 4 + 8 * len(ti.dims)
        swap_units(fd, p, 1, 4)  # ggml type
        swap_units(fd, p + 4, 1, 8)  # offset
    for ti in g.tensors:
        var n = n_elems(ti.dims)
        swap_tensor(fd, g.data_offset + ti.offset, tensor_bytes(ti.ttype, n), ti.ttype)
    close_fd(fd)


def run() raises:
    var a = argv()
    if len(a) < 2:
        say("usage: gguf-tool dump|hash|set-metadata|convert-endian ARGS\n")
        return
    var cmd = String(a[1])
    var rest = List[String]()
    for i in range(2, len(a)):
        rest.append(String(a[i]))
    if cmd == "dump":
        dump(rest)
    elif cmd == "hash":
        hash_cmd(rest)
    elif cmd == "set-metadata":
        set_metadata(rest)
    elif cmd == "convert-endian":
        convert_endian(rest)
    else:
        usage_error("", "invalid choice: '" + cmd + "' (choose from dump, hash, set-metadata, convert-endian)")


def main():
    try:
        run()
    except e:
        print("FAIL gguf-tool: " + String(e), file=FileDescriptor(2))
        exit(1)
