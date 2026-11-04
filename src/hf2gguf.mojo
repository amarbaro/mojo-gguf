"""Hf2gguf HF_DIR --outfile OUT --outtype bf16|q8_0: a Hugging Face Qwen2 checkpoint (safetensors) as GGUF,
byte-identical to `convert_hf_to_gguf.py HF_DIR --outfile OUT --outtype ...`.

Rules ported from the script, in the order it applies them:
  * tensors: shards sorted by file name, tensors sorted by name inside a shard; multimodal names dropped;
    HF names mapped through hfconv.qwen2_names; 1-D tensors and *_norm.weight stay F32, the rest take the
    out type (Q8_0 needs the last dim divisible by 32; the script's F16 fallback is a FAIL here)
  * KVs: general.* from the directory name + generation_config.json, qwen2.* from config.json, then the
    tokenizer: vocab padded to config vocab_size with [PAD<i>], token types from the added-token flags,
    merges, special ids (tokenizer_config.json, then config.json), chat template
  * tokenizer.ggml.pre: the script hashes a tokenization of a fixed text with HF tokenizers. Here the name is
    looked up from the pre_tokenizer section of tokenizer.json (QWEN2_RE): the same regex and ByteLevel flags
    give the same hash for the same vocab family. A pre-tokenizer not in the table is a FAIL, never a guess.
"""
from std.sys import argv, exit
from std.ffi import external_call, c_ssize_t
from std.os.path import exists, isdir
from std.memory import bitcast
from gguf.model import Out, pad_to, CHUNK
from st.json import Json
from st.safetensors import Checkpoint, ST_F32
from hfconv.qwen2_names import qwen2_name
from hfconv.naming import components, size_label, id_to_title
from hfconv.quant import emit_f32, emit_bf16, emit_q8_0

comptime ALIGN = 32
comptime PAGE = 4096
comptime RELEASE = 4 << 20  # bytes of consumed source per madvise
comptime MADV_DONTNEED = 4
comptime SLICE = 16384  # elements per conversion step (a multiple of 32)
comptime GGML_F32 = 0
comptime GGML_Q8_0 = 8
comptime GGML_BF16 = 30
comptime FT_Q8_0 = 7
comptime FT_BF16 = 32
comptime T_U32 = 4
comptime T_I32 = 5
comptime T_F32 = 6
comptime T_BOOL = 7
comptime T_STR = 8
comptime T_ARR = 9
comptime TOK_NORMAL = 1
comptime TOK_CONTROL = 3
comptime TOK_USER = 4
comptime TOK_UNUSED = 5

# (regex of the Split pre-tokenizer in tokenizer.json, tokenizer.ggml.pre). Add a row only after the script's
# hash for a model of that family says so.
comptime QWEN2_RE = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

struct Gw(Movable):
    """GGUF writer: KVs in call order, the KV count patched into the header at the end."""
    var out: Out
    var n_kv: Int

    def __init__(out self, path: String) raises:
        self.out = Out(path)
        self.n_kv = 0

    def key(mut self, k: String, t: Int) raises:
        self.out.put_str(k)
        self.out.put(t, 4)
        self.n_kv += 1

    def str(mut self, k: String, v: String) raises:
        self.key(k, T_STR)
        self.out.put_str(v)

    def u32(mut self, k: String, v: Int) raises:
        self.key(k, T_U32)
        self.out.put(v, 4)

    def i32(mut self, k: String, v: Int) raises:
        self.key(k, T_I32)
        self.out.put(v & 0xFFFFFFFF, 4)

    def f32(mut self, k: String, v: Float64) raises:
        self.key(k, T_F32)
        self.out.put(Int(bitcast[DType.uint32, 1](Float32(v))), 4)

    def boolean(mut self, k: String, v: Bool) raises:
        self.key(k, T_BOOL)
        self.out.put(1 if v else 0, 1)

    def array(mut self, k: String, elem: Int, n: Int) raises:
        self.key(k, T_ARR)
        self.out.put(elem, 4)
        self.out.put(n, 8)

@fieldwise_init
struct Item(Copyable, Movable):
    var src: Int  # index into Checkpoint.tensors
    var name: String
    var ggml: Int
    var nbytes: Int

def dropped(name: String) -> Bool:
    """TextModel.filter_tensors: multimodal tensors are not written."""
    for p in ["mlp", "vit.", "vpm.", "siglip2.", "conformer.", "merger.", "resampler.", "sound_encoder.",
              "sound_projection.", "speech_embeddings."]:
        if name.startswith(p):
            return True
    for p in ["visual.", "vision.", "audio.", "talker.", "vision_", "audio_", "token2wav.", "code2wav.",
              "projector.", "pre_mm_projector_norm", "image_newline", "view_seperator", "patch_embed",
              "patch_embedding", "patch_merger.", "patch_merge_mlp.", "model.connector."]:
        if p in name:
            return True
    return False

def out_bytes(ggml: Int, n: Int) -> Int:
    if ggml == GGML_F32:
        return 4 * n
    if ggml == GGML_BF16:
        return 2 * n
    return n // 32 * 34

def find_hparam(j: Json, keys: List[String]) raises -> Int:
    for k in keys:
        var v = j.get(0, k)
        if v >= 0 and j.kind(v) != 110:
            return v
    return -1

def hp_u32(j: Json, mut w: Gw, out_key: String, keys: List[String]) raises:
    var v = find_hparam(j, keys)
    if v >= 0:
        w.u32(out_key, j.integer(v))

def hp_f32(j: Json, mut w: Gw, out_key: String, keys: List[String]) raises:
    var v = find_hparam(j, keys)
    if v >= 0:
        w.f32(out_key, j.number(v))

def sampling(j: Json, mut w: Gw) raises:
    """Metadata.load: generation_config.json values the script records, in the order it writes them."""
    var seq = j.get(0, "sequence")
    if seq >= 0 and j.kind(seq) == 34:
        w.str("general.sampling.sequence", j.string(seq))
    var v = j.get(0, "top_k")
    if v >= 0 and j.kind(v) != 110:
        w.i32("general.sampling.top_k", j.integer(v))
    for pair in [("top_p", "top_p"), ("min_p", "min_p"), ("xtc_probability", "xtc_probability"),
                 ("xtc_threshold", "xtc_threshold"), ("temperature", "temp")]:
        v = j.get(0, pair[0])
        if v >= 0 and j.kind(v) != 110:
            w.f32("general.sampling." + pair[1], j.number(v))
    v = j.get(0, "penalty_last_n")
    if v >= 0 and j.kind(v) != 110:
        w.i32("general.sampling.penalty_last_n", j.integer(v))
    v = j.get(0, "penalty_repeat")
    if v < 0:
        v = j.get(0, "repetition_penalty")
    if v >= 0 and j.kind(v) != 110:
        w.f32("general.sampling.penalty_repeat", j.number(v))
    v = j.get(0, "mirostat")
    if v >= 0 and j.kind(v) != 110:
        w.i32("general.sampling.mirostat", j.integer(v))
    for pair in [("mirostat_tau", "mirostat_tau"), ("mirostat_eta", "mirostat_eta")]:
        v = j.get(0, pair[0])
        if v >= 0 and j.kind(v) != 110:
            w.f32("general.sampling." + pair[1], j.number(v))

def looks_special(t: String) -> Bool:
    if t == "<pad>" or t == "<mask>" or t == "<2mass>" or t == "[@BOS@]":
        return True
    return (t.startswith("<|") and t.endswith("|>")) or (t.startswith("<｜") and t.endswith("｜>")) \
        or (t.startswith("<unused") and t.endswith(">"))

def pre_name(tj: Json) raises -> String:
    var pt = tj.get(0, "pre_tokenizer")
    var ok = pt >= 0 and tj.get_string(pt, "type") == "Sequence"
    if ok:
        var list = tj.get(pt, "pretokenizers")
        var cur = tj.open(list)
        var split = tj.element(cur)
        var byte_level = tj.element(cur)
        ok = split >= 0 and byte_level >= 0 and tj.element(cur) < 0
        if ok:
            ok = tj.get_string(split, "type") == "Split" and tj.get_string(split, "behavior") == "Isolated" \
                and tj.get_string(tj.get(split, "pattern"), "Regex") == QWEN2_RE and tj.kind(tj.get(split, "invert")) == 102
        if ok:
            ok = tj.get_string(byte_level, "type") == "ByteLevel" and tj.kind(tj.get(byte_level, "add_prefix_space")) == 102 \
                and tj.kind(tj.get(byte_level, "use_regex")) == 102
    if not ok:
        raise Error("pre-tokenizer not in the hf2gguf table (QWEN2_RE in src/hf2gguf.mojo): hash it with the script, then add a row")
    return "qwen2"

def tokenizer_kvs(dir: String, hp: Json, mut w: Gw) raises:
    if not exists(dir + "/tokenizer.json"):
        raise Error("no tokenizer.json in " + dir + " (slow-tokenizer-only checkpoints are not supported)")
    var tj = Json(dir + "/tokenizer.json")
    var tc = Json(List[UInt8](Span("{}".as_bytes())))
    if exists(dir + "/tokenizer_config.json"):
        tc = Json(dir + "/tokenizer_config.json")
    var vs = hp.get(0, "vocab_size")
    if vs < 0:
        raise Error("config.json has no vocab_size")
    var n_vocab = hp.integer(vs)
    var model = tj.get(0, "model")
    if tj.get_string(model, "type") != "BPE":
        raise Error("tokenizer model " + tj.get_string(model, "type") + " is not BPE")
    var tokens = List[String](capacity=n_vocab)
    var types = List[Int](capacity=n_vocab)
    for i in range(n_vocab):
        tokens.append("[PAD" + String(i) + "]")
        types.append(TOK_UNUSED)
    var cur = tj.open(tj.get(model, "vocab"))
    var key = String()
    while True:
        var v = tj.member(cur, key)
        if v < 0:
            break
        var id = tj.integer(v)
        if id < 0 or id >= n_vocab:
            raise Error("token id " + String(id) + " is not in 0 .. vocab_size " + String(n_vocab))
        tokens[id] = key.copy()
        types[id] = TOK_NORMAL
    # added tokens: tokenizer.json first, tokenizer_config.json added_tokens_decoder overrides the flags
    var a_id = List[Int]()
    var a_content = List[String]()
    var a_special = List[Bool]()
    cur = tj.open(tj.get(0, "added_tokens"))
    while True:
        var e = tj.element(cur)
        if e < 0:
            break
        a_id.append(tj.integer(tj.get(e, "id")))
        a_content.append(tj.get_string(e, "content"))
        a_special.append(tj.boolean(tj.get(e, "special")))
    var n_json = len(a_id)
    var dec = tc.get(0, "added_tokens_decoder")
    if dec >= 0:
        cur = tc.open(dec)
        while True:
            var e = tc.member(cur, key)
            if e < 0:
                break
            var id = atol_id(key)
            var found = -1
            for k in range(len(a_id)):
                if a_id[k] == id:
                    found = k
            if found < 0:
                a_id.append(id)
                a_content.append(tc.get_string(e, "content"))
                a_special.append(tc.boolean(tc.get(e, "special")))
            else:
                a_special[found] = tc.boolean(tc.get(e, "special"))
    for k in range(len(a_id)):
        var id = a_id[k]
        if id < 0 or id >= n_vocab:
            raise Error("added token id " + String(id) + " is not in 0 .. vocab_size " + String(n_vocab))
        var t = a_content[k].copy()
        if a_special[k] or looks_special(t):
            types[id] = TOK_CONTROL
        else:
            types[id] = TOK_USER
            t = t.replace("▁", " ")
        tokens[id] = t^
    var pp = tj.get(0, "post_processor")
    if pp >= 0 and tj.kind(pp) == 123:
        var procs = tj.get(pp, "processors")
        var kinds = List[String]()
        if procs >= 0:
            cur = tj.open(procs)
            while True:
                var e = tj.element(cur)
                if e < 0:
                    break
                kinds.append(tj.get_string(e, "type"))
        else:
            kinds.append(tj.get_string(pp, "type"))
        for kd in kinds:
            if kd == "TemplateProcessing" or kd == "RobertaProcessing":
                raise Error("post_processor " + kd + " is not supported")
    var nm = tj.get(0, "normalizer")
    if nm >= 0 and tj.kind(nm) == 123 and tj.get_string(nm, "type") != "NFC":
        raise Error("normalizer " + tj.get_string(nm, "type") + " is not supported")

    w.str("tokenizer.ggml.model", "gpt2")
    w.str("tokenizer.ggml.pre", pre_name(tj))
    w.array("tokenizer.ggml.tokens", T_STR, n_vocab)
    for t in tokens:
        w.out.put_str(t)
    w.array("tokenizer.ggml.token_type", T_I32, n_vocab)
    for t in types:
        w.out.put(t, 4)

    var mg = tj.get(model, "merges")
    var n_merges = 0
    cur = tj.open(mg)
    while tj.element(cur) >= 0:
        n_merges += 1
    if n_merges > 0:
        w.array("tokenizer.ggml.merges", T_STR, n_merges)
        cur = tj.open(mg)
        while True:
            var e = tj.element(cur)
            if e < 0:
                break
            if tj.kind(e) == 34:
                w.out.put_str(tj.string(e))
            else:  # [a, b] pairs (transformers >= 4.45): spaces inside a part become U+0120
                var c2 = tj.open(e)
                var a = tj.string(tj.element(c2)).replace(" ", "Ġ")
                var b = tj.string(tj.element(c2)).replace(" ", "Ġ")
                w.out.put_str(a + " " + b)

    # special token ids: tokenizer_config.json (token content -> id among tokenizer.json added tokens), then
    # config.json <typ>_token_id; first writer wins; add_<typ>_token flags from tokenizer_config.json
    var typs = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"]
    var keys = ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.unknown_token_id",
                "tokenizer.ggml.seperator_token_id", "tokenizer.ggml.padding_token_id",
                "tokenizer.ggml.cls_token_id", "tokenizer.ggml.mask_token_id"]
    var ids = List[Int]()
    var contents = List[String]()
    for _ in range(len(typs)):
        ids.append(-1)
        contents.append(String())
    var order = List[Int]()
    for k in range(len(typs)):
        var e = tc.get(0, typs[k] + "_token")
        if e >= 0 and tc.kind(e) == 34:
            contents[k] = tc.string(e)
        elif e >= 0 and tc.kind(e) == 123:
            contents[k] = tc.get_string(e, "content")
    if contents[0].byte_length() == 0:
        contents[0] = contents[5].copy()
    if contents[1].byte_length() == 0:
        contents[1] = contents[3].copy()
    for k in range(len(typs)):
        if contents[k].byte_length() > 0:
            for a in range(n_json):
                if a_content[a] == contents[k]:
                    ids[k] = a_id[a]
                    order.append(k)
                    break
    for k in range(len(typs)):
        var v = hp.get(0, typs[k] + "_token_id")
        if v >= 0 and hp.kind(v) != 110 and ids[k] < 0:
            var id = hp.integer(v)
            if id < 0:
                raise Error("invalid value for special token type " + typs[k])
            ids[k] = id
            order.append(k)
    for k in order:
        w.u32(keys[k], ids[k])
    for t in ["bos", "eos", "sep"]:
        var e = tc.get(0, "add_" + t + "_token")
        if e >= 0 and (tc.kind(e) == 116 or tc.kind(e) == 102):
            w.boolean("tokenizer.ggml.add_" + t + "_token", tc.boolean(e))
    var ct = tc.get(0, "chat_template")
    if ct >= 0 and tc.kind(ct) == 34:
        w.str("tokenizer.chat_template", tc.string(ct))
    elif ct >= 0 and tc.kind(ct) != 110:
        raise Error("tokenizer_config.json chat_template is not a string (named templates are not supported)")
    elif exists(dir + "/chat_template.jinja"):
        var t = Json(dir + "/chat_template.jinja")
        w.str("tokenizer.chat_template", String(unsafe_from_utf8=Span(t.b)))

def atol_id(s: String) raises -> Int:
    var v = 0
    for c in s.as_bytes():
        if c < 48 or c > 57:
            raise Error("added_tokens_decoder key " + s + " is not a token id")
        v = v * 10 + Int(c) - 48
    return v

def convert(dir: String, outfile: String, outtype: String) raises:
    var ftype: Int
    if outtype == "bf16":
        ftype = FT_BF16
    elif outtype == "q8_0":
        ftype = FT_Q8_0
    else:
        raise Error("--outtype " + outtype + ": only bf16 and q8_0")
    if isdir(outfile):
        raise Error("--outfile must be a file path, not a directory")
    var hp = Json(dir + "/config.json")
    var archs = hp.get(0, "architectures")
    var hf_arch = String()
    if archs >= 0:
        hf_arch = hp.string(first_element(hp, archs))
    if hf_arch != "Qwen2ForCausalLM" and hf_arch != "Qwen2Model":
        raise Error("architecture " + hf_arch + " is not supported (Qwen2ForCausalLM, Qwen2Model)")
    if hp.get(0, "text_config") >= 0 or exists(dir + "/modules.json"):
        raise Error("text_config / sentence-transformers pooling modules are not supported")

    # plan: every tensor the script writes, in its order
    var ck = Checkpoint(dir)
    var plan = List[Item]()
    var total = 0
    for i in range(len(ck.tensors)):
        var name = ck.tensors[i].name.copy()
        if dropped(name):
            continue
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")
        name = name.replace("language_model.", "")
        if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".rotary_emb.inv_freq"):
            continue
        if hf_arch == "Qwen2Model":
            name = "model." + name
        var new_name = qwen2_name(name)
        if new_name.byte_length() == 0:
            raise Error("Can not map tensor '" + name + "'")
        var shape = ck.tensors[i].shape.copy()
        var n_dims = len(shape)
        var n = ck.tensors[i].numel()
        var ggml = GGML_F32
        if n_dims > 1 and not new_name.endswith("_norm.weight") and new_name.endswith(".weight"):
            ggml = GGML_BF16 if ftype == FT_BF16 else GGML_Q8_0
            if ggml == GGML_Q8_0 and shape[n_dims - 1] % 32 != 0:
                raise Error(new_name + ": last dim " + String(shape[n_dims - 1]) + " is not a multiple of 32 (the script falls back to F16)")
        plan.append(Item(i, new_name^, ggml, out_bytes(ggml, n)))
        total += n

    var w = Gw(outfile)
    w.out.put(0x46554747, 4)
    w.out.put(3, 4)
    w.out.put(len(plan), 8)
    w.out.put(0, 8)  # KV count, patched below
    w.str("general.architecture", "qwen2")
    w.str("general.type", "model")
    var gc = Json(List[UInt8](Span("{}".as_bytes())))
    if exists(dir + "/generation_config.json"):
        try:
            gc = Json(dir + "/generation_config.json")
            _ = gc.skip(0)
        except:
            gc = Json(List[UInt8](Span("{}".as_bytes())))
    sampling(gc, w)
    # name heuristics: _name_or_path first, then the directory name; each field only if still unset
    var name = String()
    var org = String()
    var basename = String()
    var finetune = String()
    var version = String()
    var label = String()
    var ids = List[String]()
    var nop = hp.get_string(0, "_name_or_path")
    if nop.byte_length() > 0 and len(nop.split("/")) <= 2:
        ids.append(nop^)
    var base = dir.rstrip("/")
    var slash = base.rfind("/")
    ids.append(String(unsafe_from_utf8=Span(base.as_bytes())[slash + 1 : base.byte_length()]))
    for mid in ids:
        var c = components(mid, total)
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
    if label.byte_length() == 0 and total > 0:
        label = size_label(total)
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

    var ar = String("qwen2.")
    var blocks = find_hparam(hp, ["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
    if blocks < 0:
        raise Error("config.json: no layer count")
    w.u32(ar + "block_count", hp.integer(blocks))
    hp_u32(hp, w, ar + "context_length", ["max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length"])
    hp_u32(hp, w, ar + "embedding_length", ["hidden_size", "n_embd", "dim"])
    hp_u32(hp, w, ar + "feed_forward_length", ["prefix_dense_intermediate_size", "dense_intermediate_size", "intermediate_size", "n_inner", "hidden_dim"])
    hp_u32(hp, w, ar + "attention.head_count", ["num_attention_heads", "n_head", "n_heads"])
    hp_u32(hp, w, ar + "attention.head_count_kv", ["num_key_value_heads", "n_kv_heads"])
    for k in ["rope_scaling", "rope_parameters", "local_rope_theta", "rope_local_theta", "rope_theta_local",
              "swa_rope_theta", "rope_local_base_freq", "num_local_experts", "num_experts", "n_routed_experts",
              "num_experts_per_tok", "n_group", "topk_group", "mrope_section", "head_dim"]:
        var v = hp.get(0, k)
        if v >= 0 and hp.kind(v) != 110:
            raise Error("config.json key " + k + " needs converter rules hf2gguf does not port yet")
    var causal = hp.get(0, "is_causal")
    if causal >= 0 and hp.kind(causal) == 102:
        raise Error("is_causal=false is not supported")
    hp_f32(hp, w, ar + "rope.freq_base", ["global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base"])
    hp_f32(hp, w, ar + "attention.layer_norm_rms_epsilon", ["rms_norm_eps", "norm_eps"])
    hp_f32(hp, w, ar + "attention.layer_norm_epsilon", ["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"])
    w.u32("general.file_type", ftype)
    w.u32("general.quantization_version", 2)
    tokenizer_kvs(dir, hp, w)

    # tensor infos, then the aligned data section
    var off = 0
    for p in plan:
        ref t = ck.tensors[p.src]
        w.out.put_str(p.name)
        w.out.put(len(t.shape), 4)
        for k in range(len(t.shape)):
            w.out.put(t.shape[len(t.shape) - 1 - k], 8)
        w.out.put(p.ggml, 4)
        w.out.put(off, 8)
        off += pad_to(p.nbytes, ALIGN)
    w.out.pad(ALIGN)
    for p in plan:
        ref t = ck.tensors[p.src]
        var base_addr = ck.shards[t.shard].base + t.begin
        var n = t.numel()
        var esz = 4 if t.dtype == ST_F32 else 2
        var released = base_addr & ~(PAGE - 1)
        var done = 0
        while done < n:
            var cnt = min(n - done, SLICE)
            var nb = out_bytes(p.ggml, cnt)
            if w.out.used + nb > CHUNK:
                w.out.flush()
            var dst = Int(w.out.buf.unsafe_ptr()) + w.out.used
            if p.ggml == GGML_F32:
                emit_f32(t.dtype, base_addr, done, cnt, dst)
            elif p.ggml == GGML_BF16:
                emit_bf16(t.dtype, base_addr, done, cnt, dst)
            else:
                emit_q8_0(t.dtype, base_addr, done, cnt, dst)
            w.out.used += nb
            done += cnt
            # source pages are read once: hand them back as we go, so RSS stays near the staging buffers
            var upto = (base_addr + done * esz) & ~(PAGE - 1)
            if upto - released >= RELEASE:
                _ = external_call["madvise", Int32](released, upto - released, Int32(MADV_DONTNEED))
                released = upto
        var tail = (base_addr + n * esz + PAGE - 1) & ~(PAGE - 1)
        if tail > released:
            _ = external_call["madvise", Int32](released, tail - released, Int32(MADV_DONTNEED))
        w.out.pad(ALIGN)
    w.out.flush()
    var count = List[UInt8](length=8, fill=0)
    for k in range(8):
        count[k] = UInt8((w.n_kv >> (8 * k)) & 0xFF)
    if external_call["pwrite", c_ssize_t](w.out.fd, count.unsafe_ptr(), 8, Int64(16)) != 8:
        raise Error("cannot patch the KV count")
    w.out.close()
    print("wrote", outfile, len(plan), "tensors", w.n_kv, "kv", w.out.off, "bytes")

def first_element(j: Json, arr: Int) raises -> Int:
    var cur = j.open(arr)
    return j.element(cur)

def run() raises:
    var a = argv()
    var dir = String()
    var outfile = String()
    var outtype = String()
    var i = 1
    while i < len(a):
        if (a[i] == "--outfile" or a[i] == "--outtype") and i + 1 < len(a):
            if a[i] == "--outfile":
                outfile = String(a[i + 1])
            else:
                outtype = String(a[i + 1])
            i += 2
        elif dir.byte_length() == 0 and not String(a[i]).startswith("--"):
            dir = String(a[i])
            i += 1
        else:
            raise Error("unknown argument " + String(a[i]))
    if dir.byte_length() == 0 or outfile.byte_length() == 0 or outtype.byte_length() == 0:
        raise Error("usage: hf2gguf HF_DIR --outfile OUT --outtype bf16|q8_0")
    convert(dir, outfile, outtype)

def main():
    try:
        run()
    except e:
        print("FAIL hf2gguf:", e)
        exit(1)
