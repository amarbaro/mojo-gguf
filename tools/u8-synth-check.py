"""Differential check of hf-update's tokenizer pipeline beyond the six cached models: synthetic tokenizer.json
files built with HF `tokenizers` (Split behaviors, Digits, added-token flags, normalizers, post-processors,
ignore_merges, unk fusing), each encoded by `tokenizers` and by hf-update, compared on ntok and the sha256.

    u8-synth-check.py HF_UPDATE_BIN LLAMACPP_DIR WORKDIR      (needs `tokenizers`: a python with it, run without -I)
Prints `ok NAME` / `FAIL NAME ...` per variant and `PASS n/n` or `FAIL k/n`; exit 1 on any FAIL."""
import json, os, re, subprocess, sys
from hashlib import sha256
from tokenizers import Tokenizer, AddedToken, Regex, normalizers, pre_tokenizers, processors, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

bin_, llama, work = sys.argv[1:4]
src = open(os.path.join(llama, "conversion/base.py")).read()
chktxt = eval(re.search(r"\n\s+chktxt = (.+)\n", src[src.index("def get_vocab_base_pre"):]).group(1))
corpus = [chktxt, "the quick brown fox 123 jumps over the lazy dog's tea, 'll 33 apple 3.3 ...... ---- "] * 8
QWEN2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
BL = lambda **k: pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, **k)


def build(pre, norm=None, post=None, added=(), bytelevel=True, unk=None, fuse=False, ignore_merges=False, vocab_size=400,
          alphabet=None):
    tok = Tokenizer(BPE(unk_token=unk, fuse_unk=fuse, ignore_merges=ignore_merges))
    tok.pre_tokenizer = pre
    if norm is not None:
        tok.normalizer = norm
    if bytelevel:
        tok.decoder = decoders.ByteLevel()
    specials = ["<unk>"] if unk else []
    tr = BpeTrainer(vocab_size=vocab_size, special_tokens=specials, show_progress=False,
                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if bytelevel else (alphabet or []))
    tok.train_from_iterator(corpus, tr)
    if added:
        tok.add_tokens(list(added))
    if post is not None:
        tok.post_processor = post
    return tok


def seq(*p):
    return pre_tokenizers.Sequence(list(p))


def split(rx, behavior="isolated", invert=False):
    return pre_tokenizers.Split(Regex(rx), behavior=behavior, invert=invert)


def bos_eos(tok_id_bos=1, tok_id_eos=2):
    return processors.TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> $B:1 </s>:1",
                                         special_tokens=[("<s>", 1), ("</s>", 2)])


V = {}
V["bytelevel-regex"] = lambda: build(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True))
V["bytelevel-prefix"] = lambda: build(pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True))
V["qwen2-nfc"] = lambda: build(seq(split(QWEN2), BL()), normalizers.NFC())
V["digits-individual"] = lambda: build(seq(pre_tokenizers.Digits(individual_digits=True), BL()))
V["digits-runs"] = lambda: build(seq(pre_tokenizers.Digits(individual_digits=False), BL()))
for beh in ["removed", "isolated", "merged_with_previous", "merged_with_next", "contiguous"]:
    for inv in (False, True):
        V[f"split-{beh}-{'inv' if inv else 'fwd'}"] = (lambda b=beh, i=inv: build(seq(split(r"\p{N}+|\s+", b, i), BL())))
V["split-string"] = lambda: build(seq(pre_tokenizers.Split("..", "isolated"), BL()))
V["nfkc"] = lambda: build(seq(split(QWEN2), BL()), normalizers.NFKC())
V["lowercase"] = lambda: build(seq(split(QWEN2), BL()), normalizers.Lowercase())
V["nfkc-lower-seq"] = lambda: build(seq(split(QWEN2), BL()), normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()]))
V["added-plain"] = lambda: build(seq(split(QWEN2), BL()), added=[AddedToken("33", normalized=False), AddedToken("apple", normalized=False),
                                                                  AddedToken("'ll", normalized=True)])
V["added-lstrip-rstrip"] = lambda: build(seq(split(QWEN2), BL()), added=[
    AddedToken("3333", normalized=False, lstrip=True), AddedToken("apple", normalized=False, rstrip=True),
    AddedToken("....", normalized=False, lstrip=True, rstrip=True)])
V["added-single-word"] = lambda: build(seq(split(QWEN2), BL()), added=[
    AddedToken("33", normalized=False, single_word=True), AddedToken("the", normalized=False, single_word=True),
    AddedToken("sure", normalized=False, single_word=True)])
V["added-normalized-lower"] = lambda: build(seq(split(QWEN2), BL()), normalizers.Lowercase(), added=[
    AddedToken("rE", normalized=True), AddedToken("tea", normalized=True, lstrip=True)])
V["added-longest"] = lambda: build(seq(split(QWEN2), BL()), added=[
    AddedToken("333", normalized=False), AddedToken("3333", normalized=False), AddedToken("33333", normalized=False),
    AddedToken("  ", normalized=False), AddedToken("\n\n", normalized=False)])
V["post-template"] = lambda: build(seq(split(QWEN2), BL()), post=bos_eos(), added=[AddedToken("<s>", special=True), AddedToken("</s>", special=True)])
V["post-roberta"] = lambda: build(seq(split(QWEN2), BL()), added=[AddedToken("<s>", special=True), AddedToken("</s>", special=True)],
                                  post=processors.RobertaProcessing(("</s>", 2), ("<s>", 1)))
V["post-bert"] = lambda: build(seq(split(QWEN2), BL()), added=[AddedToken("<s>", special=True), AddedToken("</s>", special=True)],
                               post=processors.BertProcessing(("</s>", 2), ("<s>", 1)))
V["post-sequence"] = lambda: build(seq(split(QWEN2), BL()), added=[AddedToken("<s>", special=True), AddedToken("</s>", special=True)],
                                   post=processors.Sequence([processors.ByteLevel(), bos_eos()]))
V["ignore-merges"] = lambda: build(seq(split(QWEN2), BL()), ignore_merges=True, vocab_size=600)
ALPHA = [chr(c) for c in range(32, 127)]
V["unk-nofuse"] = lambda: build(split(r"\s+|\S+"), bytelevel=False, unk="<unk>", fuse=False, alphabet=ALPHA)
V["unk-fuse"] = lambda: build(split(r"\s+|\S+"), bytelevel=False, unk="<unk>", fuse=True, alphabet=ALPHA)

os.makedirs(work, exist_ok=True)
fails = []
for name, mk in V.items():
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    try:
        tok = mk()
        tok.save(os.path.join(d, "tokenizer.json"))
        ids = tok.encode(chktxt).ids
        want = f"ntok {len(ids)} chkhsh {sha256(str(ids).encode()).hexdigest()}"
        r = subprocess.run([bin_, d], capture_output=True, text=True, timeout=60)
        got = r.stdout.strip().rsplit(" pre ", 1)[0] if r.returncode == 0 else r.stdout.strip() or r.stderr.strip()[:200]
    except Exception as e:
        got, want = f"{type(e).__name__}: {e}", ""
    if got == want:
        print("ok  ", name)
    else:
        print("FAIL", name, "\n   got ", got[:150], "\n   want", want[:150])
        fails.append(name)
n = len(V)
if fails:
    print(f"FAIL {len(fails)}/{n} {' '.join(fails)}")
    sys.exit(1)
print(f"PASS {n}/{n}")
