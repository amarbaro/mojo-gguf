"""Lora2gguf ADAPTER_DIR --base HF_DIR --outfile OUT --outtype f32|f16|bf16|q8_0: a Hugging Face PEFT LoRA adapter
(adapter_model.safetensors + adapter_config.json) as a GGUF adapter, byte-identical to
`convert_lora_to_gguf.py ADAPTER_DIR --base HF_DIR --outfile OUT --outtype ...`.

Rules ported from the script, in the order it applies them:
  * tensors: taken in file offset order (safetensors `keys()`); names lose `base_model.model.`, `.lora_A/B.weight`
    becomes `.weight`; multimodal names dropped; *norm tensors pass through first (F32), every other tensor must
    be a lora_A/lora_B pair and is written after them, per base tensor in first-seen order as
    `<gguf name>.lora_a`, `<gguf name>.lora_b` (token_embd's lora_a is transposed)
  * types: 1-D stays F32; the out type otherwise; Q8_0 needs the last dim divisible by 32, else that tensor is F16
    (lora_b has last dim = rank, so it is usually F16 under q8_0)
  * KVs: architecture, general.type=adapter, adapter.type=lora, then general.* from the ADAPTER directory name
    (and its generation_config.json / config.json) with the script's negative-parameter-count naming rules,
    adapter.lora.alpha, aLoRA invocation tokens, quantization version
  * not ported, FAIL: model card README.md frontmatter, invocation_string without tokens (needs a tokenizer),
    --base-model-id / no --base (hub download), --outtype auto, non-Qwen2 base architectures
"""
from std.sys import argv, exit
from std.ffi import external_call, c_ssize_t
from std.os.path import exists, isdir
from ggufx.model import pad_to, CHUNK, open_fd, pread_into
from st.json import Json
from st.safetensors import Shard, StTensor, check_tensor, dtype_code, ST_F32
from hfconv.naming import components, size_label, id_to_title
from hfconv.quant import emit_f32, emit_bf16, emit_q8_0
from hfconv.f16 import emit_f16
from hfconv.qwen2_names import qwen2_name
from hf2gguf import Gw, dropped, sampling, find_hparam, first_element, T_U32

comptime ALIGN = 32
comptime SLICE = 16384  # elements per conversion step (a multiple of 32)
comptime GGML_F32 = 0
comptime GGML_F16 = 1
comptime GGML_Q8_0 = 8
comptime GGML_BF16 = 30

@fieldwise_init
struct Item(Copyable, Movable):
    var src: Int  # index into Adapter.tensors
    var name: String
    var ggml: Int
    var transpose: Bool
    var shape: List[Int]  # element shape as written (outermost dim first)

struct Adapter(Movable):
    """The adapter file mapped read-only, tensors in the order of their data offsets."""
    var shard: Shard
    var tensors: List[StTensor]

    def __init__(out self, path: String) raises:
        self.shard = Shard(path)
        self.tensors = List[StTensor]()
        var fd = open_fd(path, 0)
        var hdr = List[UInt8](length=8, fill=0)
        pread_into(fd, hdr, 0, 8, 0)
        var n = 0
        for k in range(8):
            n |= Int(hdr[k]) << (8 * k)
        if n <= 0 or 8 + n > self.shard.size:
            raise Error(path + ": bad safetensors header length " + String(n))
        var buf = List[UInt8](length=n, fill=0)
        pread_into(fd, buf, 0, n, 8)
        _ = external_call["close", Int32](fd)
        var j = Json(buf^)
        var cur = j.open(0)
        var name = String()
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
            check_tensor(path, name, dt, shape, lo, hi, 8 + n, self.shard.size)
            var t = StTensor(name.copy(), dt, shape^, 0, 8 + n + lo, hi - lo)
            var at = len(self.tensors)
            for k in range(len(self.tensors)):
                if t.begin < self.tensors[k].begin:
                    at = k
                    break
            self.tensors.insert(at, t^)

def out_bytes(ggml: Int, n: Int) -> Int:
    if ggml == GGML_F32:
        return 4 * n
    if ggml == GGML_Q8_0:
        return n // 32 * 34
    return 2 * n

def out_type(want: Int, new_name: String, shape: List[Int]) -> Int:
    """Type per prepare_tensors: 1-D and *_norm.weight stay F32; Q8_0 falls back to F16 on a short row."""
    var n_dims = len(shape)
    if n_dims <= 1 or new_name.endswith("_norm.weight"):
        return GGML_F32
    if want == GGML_Q8_0 and shape[n_dims - 1] % 32 != 0:
        return GGML_F16
    return want

def mapped(name: String, qwen2_model: Bool, blocks: Int) raises -> String:
    """Qwen2Model.modify_tensors: GGUF name, or the script's "Can not map tensor" refusal."""
    var n = "model." + name if qwen2_model else name.copy()
    var new_name = qwen2_name(n)
    if new_name.byte_length() > 0 and new_name.startswith("blk."):
        var bid = 0
        for c in new_name.removeprefix("blk.").as_bytes():
            if c < 48 or c > 57:
                break
            bid = bid * 10 + Int(c) - 48
        if bid >= blocks:
            new_name = String()
    if new_name.byte_length() == 0:
        raise Error("Can not map tensor '" + name + "'")
    return new_name^

def last_part(path: String) -> String:
    var base = path.rstrip("/")
    return String(unsafe_from_utf8=Span(base.as_bytes())[base.rfind("/") + 1 : base.byte_length()])

def convert(lora: String, base_dir: String, outfile: String, outtype: String) raises:
    var want: Int
    var out_name: String
    if outtype == "f32":
        want = GGML_F32
        out_name = "F32"
    elif outtype == "f16":
        want = GGML_F16
        out_name = "F16"
    elif outtype == "bf16":
        want = GGML_BF16
        out_name = "BF16"
    elif outtype == "q8_0":
        want = GGML_Q8_0
        out_name = "Q8_0"
    else:
        raise Error("--outtype " + outtype + ": f32, f16, bf16 or q8_0 (auto fails in the script for adapters)")
    if exists(lora + "/README.md") and Json(lora + "/README.md").kind(0) == 45:
        raise Error("README.md model card frontmatter in " + lora + " is not supported")

    var lp = Json(lora + "/adapter_config.json")
    var av = lp.get(0, "lora_alpha")
    if av < 0:
        raise Error("adapter_config.json has no lora_alpha")
    var alpha = lp.number(av)
    var invocation = List[Int]()
    var tv = lp.get(0, "alora_invocation_tokens")
    if tv >= 0 and lp.kind(tv) == 91:
        var tc = lp.open(tv)
        while True:
            var e = lp.element(tc)
            if e < 0:
                break
            invocation.append(lp.integer(e))
    var iv = lp.get(0, "invocation_string")
    if len(invocation) == 0 and iv >= 0 and lp.kind(iv) == 34 and lp.string(iv).byte_length() > 0:
        raise Error("invocation_string without alora_invocation_tokens needs the base tokenizer, which lora2gguf does not run")

    var hp = Json(base_dir + "/config.json")
    var archs = hp.get(0, "architectures")
    var hf_arch = String()
    if archs >= 0:
        hf_arch = hp.string(first_element(hp, archs))
    if hf_arch != "Qwen2ForCausalLM" and hf_arch != "Qwen2Model":
        raise Error("architecture " + hf_arch + " is not supported (Qwen2ForCausalLM, Qwen2Model)")
    if hp.get(0, "text_config") >= 0:
        raise Error("text_config is not supported")
    var blk = find_hparam(hp, ["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
    if blk < 0:
        raise Error("base config.json: no layer count")
    var blocks = hp.integer(blk)

    # plan: norm tensors as they are met, then the lora pairs in first-seen order of their base tensor
    var ad = Adapter(lora + "/adapter_model.safetensors")
    var plan = List[Item]()
    var keys = List[String]()
    var ia = List[Int]()
    var ib = List[Int]()
    var qwen2_model = hf_arch == "Qwen2Model"
    for i in range(len(ad.tensors)):
        var name = ad.tensors[i].name.copy()
        var bn = name.replace("base_model.model.", "").replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
        bn = bn.replace(".lora_embedding_A", ".weight").replace(".lora_embedding_B", ".weight")
        if dropped(bn):
            continue
        if bn.endswith("e_score_correction_bias"):
            bn = bn.replace("e_score_correction_bias", "e_score_correction.bias")
        bn = bn.replace("language_model.", "")
        var is_a = ".lora_A.weight" in name or ".lora_embedding_A" in name
        var is_b = ".lora_B.weight" in name or ".lora_embedding_B" in name
        if not is_a and not is_b:
            if ".base_layer.weight" in name:
                continue
            if "_layernorm" in name or ".norm" in name:
                if bn.endswith(".attention.masked_bias") or bn.endswith(".attention.bias") or bn.endswith(".rotary_emb.inv_freq"):
                    continue
                var nn = mapped(bn, qwen2_model, blocks)
                if "_norm" in nn and len(ad.tensors[i].shape) != 1:
                    raise Error(name + ": norm tensor is not 1-D")
                plan.append(Item(i, nn^, GGML_F32, False, ad.tensors[i].shape.copy()))
                continue
            raise Error("Unexpected name '" + name + "': Not a lora_A or lora_B tensor")
        var k = -1
        for q in range(len(keys)):
            if keys[q] == bn:
                k = q
                break
        if k < 0:
            k = len(keys)
            keys.append(bn^)
            ia.append(-1)
            ib.append(-1)
        if is_a:
            ia[k] = i
        else:
            ib[k] = i
    var total = 0
    var pairs = False
    for k in range(len(keys)):
        if ia[k] < 0 or ib[k] < 0:
            raise Error(keys[k] + ": lora_A or lora_B tensor is missing")
        if keys[k].endswith(".attention.masked_bias") or keys[k].endswith(".attention.bias") or keys[k].endswith(".rotary_emb.inv_freq"):
            continue
        var nn = mapped(keys[k], qwen2_model, blocks)
        if "_norm" in nn:
            raise Error(keys[k] + ": a norm tensor with lora_A/lora_B is not supported")
        var sa = ad.tensors[ia[k]].shape.copy()
        var sb = ad.tensors[ib[k]].shape.copy()
        if len(sa) != 2 or len(sb) != 2:
            raise Error(keys[k] + ": lora_A/lora_B must be 2-D")
        if sa[0] != sb[1]:
            raise Error(keys[k] + ": rank mismatch, lora_A " + String(sa[0]) + " rows vs lora_B " + String(sb[1]) + " columns")
        var flip = "token_embd.weight" in nn
        var oa = List[Int]()
        oa.append(sa[1] if flip else sa[0])
        oa.append(sa[0] if flip else sa[1])
        total += sb[0] * oa[1]
        plan.append(Item(ia[k], nn + ".lora_a", out_type(want, nn + ".lora_a", oa), flip, oa^))
        plan.append(Item(ib[k], nn + ".lora_b", out_type(want, nn + ".lora_b", sb), False, sb^))
        pairs = True
    for p in plan:
        if p.name.endswith(".lora_a") or p.name.endswith(".lora_b"):
            continue
        var n = 1
        for d in p.shape:
            n *= d
        total += n
    var total_params = -total if pairs else total

    # general.* names from the adapter directory (and a config.json beside it), each field only if still unset
    var name = String()
    var org = String()
    var basename = String()
    var finetune = String()
    var version = String()
    var label = String()
    var ids = List[String]()
    if exists(lora + "/config.json"):
        var nop = Json(lora + "/config.json").get_string(0, "_name_or_path")
        if nop.byte_length() > 0 and len(nop.split("/")) <= 2:
            ids.append(nop^)
    ids.append(last_part(lora))
    for mid in ids:
        var c = components(mid, total_params)
        if name.byte_length() == 0 and c.full.byte_length() > 0:
            name = id_to_title(c.full)
        if org.byte_length() == 0 and c.org.byte_length() > 0:
            org = id_to_title(c.org)
        if basename.byte_length() == 0:
            basename = c.basename.copy()
        if finetune.byte_length() == 0:
            finetune = c.finetune.copy()
        if version.byte_length() == 0:
            version = c.version.copy()
        if label.byte_length() == 0:
            label = c.size_label.copy()
    if label.byte_length() == 0 and total_params > 0:
        label = size_label(total_params)
    if name.byte_length() == 0:
        name = last_part(base_dir)

    var path: String
    if isdir(outfile):
        var stem = basename.copy() if basename.byte_length() > 0 else name.copy()
        var fname = stem.strip().replace(" ", "-").replace("/", "-")
        if label.byte_length() > 0:
            fname += "-" + label
        if finetune.byte_length() > 0:
            fname += "-" + finetune.strip().replace(" ", "-")
        if version.byte_length() > 0:
            fname += "-" + version.strip().replace(" ", "-")
        fname += "-" + out_name
        if total_params < 0:
            fname += "-LoRA"
        path = outfile.rstrip("/") + "/" + fname + ".gguf"
    else:  # templated file name: {ftype} {outtype} {} take the lower-case type, {FTYPE} {OUTTYPE} the upper-case
        var slash = outfile.rfind("/")
        var head = String(unsafe_from_utf8=Span(outfile.as_bytes())[0 : slash + 1])
        var tail = String(unsafe_from_utf8=Span(outfile.as_bytes())[slash + 1 : outfile.byte_length()])
        var low = out_name.lower()
        tail = tail.replace("{ftype}", low).replace("{outtype}", low).replace("{FTYPE}", out_name).replace("{OUTTYPE}", out_name).replace("{}", low)
        path = head + tail

    var w = Gw(path)
    w.out.put(0x46554747, 4)
    w.out.put(3, 4)
    w.out.put(len(plan), 8)
    w.out.put(0, 8)  # KV count, patched below
    w.str("general.architecture", "qwen2")
    w.str("general.type", "adapter")
    w.str("adapter.type", "lora")
    var gc = Json(List[UInt8](Span("{}".as_bytes())))
    if exists(lora + "/generation_config.json"):
        try:
            gc = Json(lora + "/generation_config.json")
            _ = gc.skip(0)
        except:
            gc = Json(List[UInt8](Span("{}".as_bytes())))
    sampling(gc, w)
    w.str("general.name", name)
    if version.byte_length() > 0:
        w.str("general.version", version)
    if org.byte_length() > 0:
        w.str("general.organization", org)
    if finetune.byte_length() > 0:
        w.str("general.finetune", finetune)
    if basename.byte_length() > 0:
        w.str("general.basename", basename)
    if label.byte_length() > 0:
        w.str("general.size_label", label)
    w.f32("adapter.lora.alpha", alpha)
    if len(invocation) > 0:
        w.array("adapter.alora.invocation_tokens", T_U32, len(invocation))
        for t in invocation:
            w.out.put(t, 4)
    w.u32("general.quantization_version", 2)

    # tensor infos, then the aligned data section
    var off = 0
    for p in plan:
        w.out.put_str(p.name)
        w.out.put(len(p.shape), 4)
        for k in range(len(p.shape)):
            w.out.put(p.shape[len(p.shape) - 1 - k], 8)
        w.out.put(p.ggml, 4)
        w.out.put(off, 8)
        var n = 1
        for d in p.shape:
            n *= d
        off += pad_to(out_bytes(p.ggml, n), ALIGN)
    w.out.pad(ALIGN)
    for p in plan:
        ref t = ad.tensors[p.src]
        var n = t.numel()
        var dtype = t.dtype
        var addr = ad.shard.base + t.begin
        var flipped = List[Float32](length=n if p.transpose else 0, fill=0)
        if p.transpose:  # (rows, cols) -> (cols, rows), widened to F32; one tensor's worth of staging
            var rows = t.shape[0]
            var cols = t.shape[1]
            var tmp = List[Float32](length=n, fill=0)
            emit_f32(dtype, addr, 0, n, Int(tmp.unsafe_ptr()))
            for r in range(rows):
                for c in range(cols):
                    flipped[c * rows + r] = tmp[r * cols + c]
            dtype = ST_F32
            addr = Int(flipped.unsafe_ptr())
        var done = 0
        while done < n:
            var cnt = min(n - done, SLICE)
            var nb = out_bytes(p.ggml, cnt)
            if w.out.used + nb > CHUNK:
                w.out.flush()
            var dst = Int(w.out.buf.unsafe_ptr()) + w.out.used
            if p.ggml == GGML_F32:
                emit_f32(dtype, addr, done, cnt, dst)
            elif p.ggml == GGML_F16:
                emit_f16(dtype, addr, done, cnt, dst)
            elif p.ggml == GGML_BF16:
                emit_bf16(dtype, addr, done, cnt, dst)
            else:
                emit_q8_0(dtype, addr, done, cnt, dst)
            w.out.used += nb
            done += cnt
        w.out.pad(ALIGN)
    w.out.flush()
    var count = List[UInt8](length=8, fill=0)
    for k in range(8):
        count[k] = UInt8((w.n_kv >> (8 * k)) & 0xFF)
    if external_call["pwrite", c_ssize_t](w.out.fd, count.unsafe_ptr(), 8, Int64(16)) != 8:
        raise Error("cannot patch the KV count")
    w.out.close()
    print("wrote", path, len(plan), "tensors", w.n_kv, "kv", w.out.off, "bytes")

def run() raises:
    var a = argv()
    var lora = String()
    var base_dir = String()
    var outfile = String()
    var outtype = String("f32")
    var i = 1
    while i < len(a):
        var s = String(a[i])
        if s == "--base" or s == "--outfile" or s == "--outtype":
            if i + 1 >= len(a):
                raise Error(s + " needs a value")
            if s == "--base":
                base_dir = String(a[i + 1])
            elif s == "--outfile":
                outfile = String(a[i + 1])
            else:
                outtype = String(a[i + 1])
            i += 2
        elif lora.byte_length() == 0 and not s.startswith("--"):
            lora = s^
            i += 1
        else:
            raise Error("unknown argument " + s)
    if lora.byte_length() == 0:
        raise Error("usage: lora2gguf ADAPTER_DIR --base HF_DIR [--outfile OUT] [--outtype f32|f16|bf16|q8_0]")
    if base_dir.byte_length() == 0:
        raise Error("--base HF_DIR is required (hub download is not supported)")
    if outfile.byte_length() == 0:
        outfile = lora.copy()
    convert(lora, base_dir, outfile, outtype)

def main():
    try:
        run()
    except e:
        print("FAIL lora2gguf:", e)
        exit(1)
