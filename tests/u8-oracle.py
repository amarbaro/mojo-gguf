"""U8 oracle: the pinned converter's pre-tokenizer fingerprint for local tokenizer dirs. For each DIR prints
`DIR ntok N chkhsh H pre NAME` (NAME from the pinned conversion/base.py table, `unknown` if absent) and writes
DIR's token ids of the check text to OUT/<basename>.ids (one per line). Needs transformers.
usage: u8-oracle.py LLAMACPP OUT DIR..."""
import os, re, sys
from hashlib import sha256
from transformers import AutoTokenizer
src = open(os.path.join(sys.argv[1], "conversion/base.py")).read()
body = src[src.index("def get_vocab_base_pre"):]
chktxt = eval(re.search(r"\n\s+chktxt = (.+)\n", body).group(1))
table = dict(re.findall(r'if chkhsh == "([0-9a-f]{64})":(?:\s*#[^\n]*)*\s*res = "([^"]+)"', body))
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for d in sys.argv[3:]:
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=False)
    ids = tok.encode(chktxt)
    h = sha256(str(ids).encode()).hexdigest()
    with open(os.path.join(out, os.path.basename(d.rstrip("/")) + ".ids"), "w") as f:
        f.write("\n".join(map(str, ids)) + "\n")
    print(f"{d} ntok {len(ids)} chkhsh {h} pre {table.get(h, 'unknown')}")
