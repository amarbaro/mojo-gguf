"""Encode the converter's check text with a tokenizer.json, the way `tokenizer.encode(chktxt)` does in HF tokenizers:
normalizer -> added tokens -> pre-tokenizer -> BPE -> post-processor.

The check text is fixed, so two shortcuts are exact rather than approximate:
  * normalizers fold per character from tables generated out of the text itself plus ASCII (chk_table.mojo), no
    Unicode data: a character outside both folds to itself, which only matters for a non-ASCII added token;
  * only vocab entries and merges built from substrings of the text's pre-tokenized words can ever take part in
    BPE, so tokenizer.json is scanned once and everything else is skipped, never loaded.
A pipeline stage this file does not model is a FAIL naming it, never a guess.
"""
from std.collections import Dict, Set
from std.os.path import exists
from uregex import Pattern
from st.json import Json
from hfconv.chk_table import CHKTXT, FOLD_NFC, FOLD_NFD, FOLD_NFKC, FOLD_NFKD, FOLD_LOWER, fold_char

comptime QUOTE = 34
comptime OBJECT = 123
comptime QWEN2_RE = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
comptime GPT2_RE = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"


@fieldwise_init
struct Seg(Copyable, Movable):
    """A byte span of a piece; hit = the pattern matched it (as opposed to the text between matches)."""
    var start: Int
    var end: Int
    var hit: Bool


@fieldwise_init
struct Piece(Copyable, Movable):
    """id >= 0: an added token; id < 0: text still to tokenize."""
    var id: Int
    var text: String


struct Added(Copyable, Movable):
    var id: Int
    var content: String
    var single_word: Bool
    var lstrip: Bool
    var rstrip: Bool
    var normalized: Bool

    def __init__(out self, id: Int, var content: String, single_word: Bool, lstrip: Bool, rstrip: Bool, normalized: Bool):
        self.id = id
        self.content = content^
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = normalized


def substr(s: String, a: Int, b: Int) -> String:
    return String(unsafe_from_utf8=Span(s.as_bytes())[a:b])


def flag(j: Json, obj: Int, key: String, default: Bool = False) raises -> Bool:
    var v = j.get(obj, key)
    return j.boolean(v) if v >= 0 and j.kind(v) != 110 else default


def node_type(j: Json, node: Int) raises -> String:
    return j.get_string(node, "type") if node >= 0 and j.kind(node) == OBJECT else String()


def is_ws(cp: Int) -> Bool:
    return (cp >= 9 and cp <= 13) or cp == 32 or cp == 0x85 or cp == 0xA0 or cp == 0x1680 or (cp >= 0x2000 and cp <= 0x200A) or cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F or cp == 0x3000


def cp_len(lead: UInt8) -> Int:
    return 1 if lead < 0x80 else (2 if lead < 0xE0 else (3 if lead < 0xF0 else 4))


def cp_at(b: Span[Byte, _], i: Int) -> Int:
    var n = cp_len(b[i])
    if n == 1:
        return Int(b[i])
    var v = Int(b[i]) & (0x3F >> (n - 1))
    for k in range(1, n):
        v = (v << 6) | (Int(b[i + k]) & 0x3F)
    return v


def ws_start(b: Span[Byte, _], end: Int) -> Int:
    """Start of the whitespace run that ends at `end`."""
    var i = end
    while i > 0:
        var s = i - 1
        while s > 0 and (b[s] & 0xC0) == 0x80:
            s -= 1
        if not is_ws(cp_at(b, s)):
            break
        i = s
    return i


def ws_end(b: Span[Byte, _], start: Int) -> Int:
    """End of the whitespace run that starts at `start`."""
    var i = start
    while i < len(b) and is_ws(cp_at(b, i)):
        i += cp_len(b[i])
    return i


def is_word_cp(cp: Int) -> Bool:
    return (cp >= 48 and cp <= 57) or (cp >= 65 and cp <= 90) or cp == 95 or (cp >= 97 and cp <= 122) or (cp >= 0x80 and not is_ws(cp))


def fold(text: String, form: Int) -> String:
    var out = String()
    for c in text.codepoints():
        var r = fold_char(form, Int(c.to_u32()))
        out += r if r.byte_length() > 0 else String(c)
    return out^


def normalize(j: Json, node: Int, text: String) raises -> String:
    var t = node_type(j, node)
    if t == "":
        return text
    if t == "Sequence":
        var out = text
        var cur = j.open(j.get(node, "normalizers"))
        while True:
            var e = j.element(cur)
            if e < 0:
                break
            out = normalize(j, e, out)
        return out^
    if t == "NFC":
        return fold(text, FOLD_NFC)
    if t == "NFD":
        return fold(text, FOLD_NFD)
    if t == "NFKC":
        return fold(text, FOLD_NFKC)
    if t == "NFKD":
        return fold(text, FOLD_NFKD)
    if t == "Lowercase":
        return fold(text, FOLD_LOWER)
    raise Error("normalizer " + t + " is not modelled")


def _prev_start(b: Span[Byte, _], i: Int) -> Int:
    var s = i - 1
    while s > 0 and (b[s] & 0xC0) == 0x80:
        s -= 1
    return s


def split_added(text: String, toks: List[Added], normalized: Bool) -> List[Piece]:
    """HF AddedVocabulary::find_matches: leftmost-longest literal matches of the tokens whose `normalized`
    flag is `normalized`, single_word / lstrip / rstrip applied, the text between them kept as pieces."""
    var out = List[Piece]()
    var b = text.as_bytes()
    var n = len(b)
    var cand = List[Int]()
    for k in range(len(toks)):
        if toks[k].normalized == normalized and toks[k].content.byte_length() > 0:
            cand.append(k)
    var start_offset = 0
    var pos = 0
    while pos < n and len(cand) > 0:
        var best = -1
        var best_len = 0
        for k in cand:
            var c = toks[k].content.as_bytes()
            var m = len(c)
            if m > best_len and pos + m <= n:
                var same = True
                for q in range(m):
                    if b[pos + q] != c[q]:
                        same = False
                        break
                if same:
                    best = k
                    best_len = m
        if best < 0:
            pos += 1
            continue
        var start = pos
        var stop = pos + best_len
        pos = stop
        ref t = toks[best]
        if t.single_word:
            var before = start == 0 or not is_word_cp(cp_at(b, _prev_start(b, start)))
            var after = stop == n or not is_word_cp(cp_at(b, stop))
            if not (before and after):
                continue
        if t.lstrip:
            start = max(ws_start(b, start), start_offset)
        var end = ws_end(b, stop) if t.rstrip else stop
        if start_offset < start:
            out.append(Piece(-1, substr(text, start_offset, start)))
        out.append(Piece(t.id, String()))
        start_offset = end
    if start_offset < n or len(out) == 0:
        out.append(Piece(-1, substr(text, start_offset, n)))
    return out^


def segments(pat: Pattern, text: String, invert: Bool) -> List[Seg]:
    var segs = List[Seg]()
    var prev = 0
    for m in pat.findall_spans(text):
        if m.start > prev:
            segs.append(Seg(prev, m.start, invert))
        segs.append(Seg(m.start, m.end, not invert))
        prev = m.end
    if prev < text.byte_length():
        segs.append(Seg(prev, text.byte_length(), invert))
    return segs^


def regex_escape(s: String) -> String:
    var out = String()
    for c in s.codepoints():
        var cp = Int(c.to_u32())
        var plain = cp >= 0x80 or (cp >= 48 and cp <= 57) or (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122)
        out += String(c) if plain else "\\" + String(c)
    return out^


def merge_segments(segs: List[Seg], behavior: String) raises -> List[Seg]:
    """tokenizers SplitDelimiterBehavior over the match / gap segments of one piece."""
    var out = List[Seg]()
    var prev_hit = False
    if behavior == "Isolated":
        return segs.copy()
    if behavior == "Removed":
        for s in segs:
            if not s.hit:
                out.append(Seg(s.start, s.end, False))
    elif behavior == "MergedWithPrevious" or behavior == "Contiguous":
        for s in segs:
            var join = (s.hit and not prev_hit) if behavior == "MergedWithPrevious" else (s.hit == prev_hit)
            if join and len(out) > 0:
                out[len(out) - 1].end = s.end
            else:
                out.append(Seg(s.start, s.end, False))
            prev_hit = s.hit
    elif behavior == "MergedWithNext":
        var rev = List[Seg]()
        for i in range(len(segs) - 1, -1, -1):
            var s = segs[i].copy()
            if s.hit and not prev_hit and len(rev) > 0:
                rev[len(rev) - 1].start = s.start
            else:
                rev.append(Seg(s.start, s.end, False))
            prev_hit = s.hit
        for i in range(len(rev) - 1, -1, -1):
            out.append(rev[i].copy())
    else:
        raise Error("Split behavior " + behavior + " is not modelled")
    return out^


def byte_chars() -> List[String]:
    """GPT-2 bytes_to_unicode: printable bytes stand for themselves, the rest take 256.. in order."""
    var out = List[String]()
    var extra = 256
    for b in range(256):
        var plain = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)
        if plain:
            out.append(String(Codepoint.from_u32(UInt32(b)).value()))
        else:
            out.append(String(Codepoint.from_u32(UInt32(extra)).value()))
            extra += 1
    return out^


def pretokenize(j: Json, node: Int, var pieces: List[String]) raises -> List[String]:
    var t = node_type(j, node)
    if t == "":
        return pieces^
    if t == "Sequence":
        var cur = j.open(j.get(node, "pretokenizers"))
        while True:
            var e = j.element(cur)
            if e < 0:
                break
            pieces = pretokenize(j, e, pieces^)
        return pieces^
    var out = List[String]()
    if t == "Split" or t == "Digits":
        var rx: String
        var behavior = String("Isolated")
        var invert = False
        if t == "Digits":
            rx = "\\p{N}" if flag(j, node, "individual_digits") else "\\p{N}+"
        else:
            var pv = j.get(node, "pattern")
            rx = j.get_string(pv, "Regex")
            if rx == "":
                rx = regex_escape(j.get_string(pv, "String"))
            if rx == "":
                raise Error("Split pattern is neither Regex nor String")
            behavior = j.get_string(node, "behavior")
            invert = flag(j, node, "invert")
        var pat = Pattern(rx)
        for p in pieces:
            for s in merge_segments(segments(pat, p, invert), behavior):
                if s.end > s.start:
                    out.append(substr(p, s.start, s.end))
        return out^
    if t == "ByteLevel":
        var prefix = flag(j, node, "add_prefix_space")
        var use_regex = flag(j, node, "use_regex", True)
        var gpt2 = Pattern(GPT2_RE)
        var chars = byte_chars()
        for p in pieces:
            var s = p
            if prefix and not s.startswith(" "):
                s = " " + s
            var parts = gpt2.split_keep(s) if use_regex else [s]
            for part in parts:
                var w = String()
                for byte in part.as_bytes():
                    w += chars[Int(byte)]
                if w.byte_length() > 0:
                    out.append(w^)
        return out^
    raise Error("pre_tokenizer " + t + " is not modelled")


def chars_of(w: String) -> List[String]:
    var out = List[String]()
    for c in w.codepoints():
        out.append(String(c))
    return out^


struct Bpe(Movable):
    var vocab: Dict[String, Int]
    var ranks: Dict[Int, Int]
    var ignore_merges: Bool
    var unk: Int
    var fuse_unk: Bool

    def __init__(out self, j: Json, model: Int, words: List[String]) raises:
        """Load only the vocab entries that are substrings of `words`, and the merges among them."""
        self.vocab = Dict[String, Int]()
        self.ranks = Dict[Int, Int]()
        self.ignore_merges = flag(j, model, "ignore_merges")
        self.fuse_unk = flag(j, model, "fuse_unk")
        self.unk = -1
        if flag(j, model, "byte_fallback"):
            raise Error("BPE byte_fallback is not modelled")
        for key in ["continuing_subword_prefix", "end_of_word_suffix"]:
            if j.get_string(model, key) != "":
                raise Error("BPE " + key + " is not modelled")
        var dropout = j.get(model, "dropout")
        if dropout >= 0 and j.kind(dropout) != 110:
            raise Error("BPE dropout is not modelled")
        var want = Set[String]()
        for w in words:
            var at = List[Int]()  # byte offset of every codepoint boundary
            var off = 0
            for c in w.codepoints():
                at.append(off)
                off += c.utf8_byte_length()
            at.append(off)
            for i in range(len(at) - 1):
                for k in range(i + 1, len(at)):
                    want.add(substr(w, at[i], at[k]))  # fresh strings: a Set key must not be appended to later
        var unk_tok = j.get_string(model, "unk_token")
        if unk_tok != "":
            want.add(unk_tok)
        var cur = j.open(j.get(model, "vocab"))
        var key = String()
        while True:
            var v = j.member(cur, key)
            if v < 0:
                break
            if key in want:
                self.vocab[key] = j.integer(v)
        if unk_tok != "":
            self.unk = self.vocab.get(unk_tok, -1)
        var mcur = j.open(j.get(model, "merges"))
        var rank = 0
        while True:
            var e = j.element(mcur)
            if e < 0:
                break
            var a: String
            var b: String
            if j.kind(e) == QUOTE:
                var s = j.string(e)
                var sp = s.find(" ")
                a = substr(s, 0, sp)
                b = substr(s, sp + 1, s.byte_length())
            else:
                var pc = j.open(e)
                a = j.string(j.element(pc))
                b = j.string(j.element(pc))
            var ia = self.vocab.get(a, -1)
            var ib = self.vocab.get(b, -1)
            if ia >= 0 and ib >= 0:
                var ic = self.vocab.get(a + b, -1)
                if ic >= 0:
                    self.ranks[(ia << 32) | ib] = (rank << 32) | ic
            rank += 1

    def encode_word(self, w: String, mut out: List[Int]):
        if self.ignore_merges:
            var whole = self.vocab.get(w, -1)
            if whole >= 0:
                out.append(whole)
                return
        var syms = List[Int]()
        var last_unk = False
        for ch in chars_of(w):
            var id = self.vocab.get(ch, -1)
            if id >= 0:
                syms.append(id)
                last_unk = False
            elif self.unk >= 0:
                if not (self.fuse_unk and last_unk):
                    syms.append(self.unk)
                last_unk = True
        while len(syms) > 1:
            var best = -1
            var best_rank = 1 << 62
            var best_id = 0
            for i in range(len(syms) - 1):
                var v = self.ranks.get((syms[i] << 32) | syms[i + 1], -1)
                if v >= 0 and (v >> 32) < best_rank:
                    best_rank = v >> 32
                    best = i
                    best_id = v & 0xFFFFFFFF
            if best < 0:
                break
            syms[best] = best_id
            _ = syms.pop(best + 1)
        for id in syms:
            out.append(id)


def read_added(j: Json) raises -> List[Added]:
    var out = List[Added]()
    var arr = j.get(0, "added_tokens")
    if arr < 0 or j.kind(arr) != 91:
        return out^
    var cur = j.open(arr)
    while True:
        var e = j.element(cur)
        if e < 0:
            break
        var special = flag(j, e, "special")
        out.append(Added(
            j.integer(j.get(e, "id")), j.get_string(e, "content"), flag(j, e, "single_word"),
            flag(j, e, "lstrip"), flag(j, e, "rstrip"), flag(j, e, "normalized", not special),
        ))
    return out^


def special_ids(j: Json, node: Int, name: String) raises -> List[Int]:
    var out = List[Int]()
    var arr = j.get(j.get(node, "special_tokens"), name)
    arr = j.get(arr, "ids")
    if arr < 0:
        raise Error("post_processor special token " + name + " has no ids")
    var cur = j.open(arr)
    while True:
        var e = j.element(cur)
        if e < 0:
            break
        out.append(j.integer(e))
    return out^


def post_process(j: Json, node: Int, var ids: List[Int]) raises -> List[Int]:
    var t = node_type(j, node)
    if t == "" or t == "ByteLevel":
        return ids^
    if t == "Sequence":
        var cur = j.open(j.get(node, "processors"))
        while True:
            var e = j.element(cur)
            if e < 0:
                break
            ids = post_process(j, e, ids^)
        return ids^
    var out = List[Int]()
    if t == "RobertaProcessing" or t == "BertProcessing":
        for k in ["cls", "sep"]:
            var pair = j.open(j.get(node, k))
            _ = j.element(pair)
            var id = j.integer(j.element(pair))
            if k == "cls":
                out.append(id)
            else:
                out.extend(ids.copy())
                out.append(id)
        return out^
    if t == "TemplateProcessing":
        var cur = j.open(j.get(node, "single"))
        while True:
            var e = j.element(cur)
            if e < 0:
                break
            var sp = j.get(e, "SpecialToken")
            if sp >= 0:
                out.extend(special_ids(j, node, j.get_string(sp, "id")))
            else:
                out.extend(ids.copy())
        return out^
    raise Error("post_processor " + t + " is not modelled")


def json_escape(s: String) -> String:
    var out = String()
    for c in s.codepoints():
        out += "\\" + String(c) if (c == Codepoint.ord("\\") or c == Codepoint.ord('"')) else String(c)
    return out^


def tokenizer_class(dir: String) raises -> String:
    var path = dir + "/tokenizer_config.json"
    return Json(path).get_string(0, "tokenizer_class") if exists(path) else String()


def encode_chktxt(dir: String) raises -> List[Int]:
    """transformers' Qwen2Tokenizer ignores the pipeline written in tokenizer.json and builds its own (NFC, the
    QWEN2_RE split, ByteLevel, no post-processor); AutoTokenizer.encode is the converter's ground truth, so a
    dir whose tokenizer_class names it takes that pipeline here too. The vocab and merges are the file's."""
    var j = Json(dir + "/tokenizer.json")
    var cls = tokenizer_class(dir)
    if cls == "Qwen2Tokenizer" or cls == "Qwen2TokenizerFast":
        var builtin = String('{"normalizer":{"type":"NFC"},"post_processor":null,"pre_tokenizer":{"type":"Sequence",')
        builtin += '"pretokenizers":[{"type":"Split","pattern":{"Regex":"' + json_escape(QWEN2_RE)
        builtin += '"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]}}'
        return encode_with(j, Json(List[UInt8](builtin.as_bytes())))
    return encode_with(j, j)


def encode_with(j: Json, nodes: Json) raises -> List[Int]:
    var model = j.get(0, "model")
    if node_type(j, model) != "BPE":
        raise Error("tokenizer.json model is '" + node_type(j, model) + "', only BPE is modelled")
    var toks = read_added(j)
    var norm = nodes.get(0, "normalizer")
    for i in range(len(toks)):
        if toks[i].normalized:  # a normalized token is matched in its normalized form
            toks[i].content = normalize(nodes, norm, toks[i].content)
    var pre = nodes.get(0, "pre_tokenizer")
    var ids_in = List[Int]()  # added-token id, or -1 for a word
    var words = List[String]()
    for p1 in split_added(CHKTXT, toks, False):
        if p1.id >= 0:
            ids_in.append(p1.id)
            words.append(String())
            continue
        for p2 in split_added(normalize(nodes, norm, p1.text), toks, True):
            if p2.id >= 0:
                ids_in.append(p2.id)
                words.append(String())
                continue
            for w in pretokenize(nodes, pre, [p2.text]):
                ids_in.append(-1)
                words.append(w)
    var bpe = Bpe(j, model, words)
    var ids = List[Int]()
    for i in range(len(words)):
        if ids_in[i] >= 0:
            ids.append(ids_in[i])
        else:
            bpe.encode_word(words[i], ids)
    return post_process(nodes, nodes.get(0, "post_processor"), ids^)
