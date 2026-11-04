"""HF tensor name -> GGUF tensor name, from the generated per-arch rows in tensor_map.mojo.

A row is a template: a block row matches `pre + N + post` for a canonical decimal N below n_blocks and expands
the value with N when it carries {bid}. tools/gen-tables.py rejects any arch where one key maps to two values,
so the first matching row is the only matching row.

Names are base names as in gguf-py's TensorNameMap.mapping: no ".weight" or ".bias" suffix. The caller strips the
suffix first and appends it to the answer, as gguf-py's get_type_and_name does with try_suffixes.
"""
from .tensor_map import Row, arch_rows


def block_id(name: String, pre: String, post: String, n_blocks: Int) -> Int:
    """N when name is pre + N + post with N a canonical decimal below n_blocks; -1 otherwise."""
    var n = name.byte_length()
    var lo = pre.byte_length()
    var hi = n - post.byte_length()
    if hi <= lo or hi - lo > 9 or not name.startswith(pre) or not name.endswith(post):
        return -1
    var b = name.as_bytes()
    if hi - lo > 1 and b[lo] == 48:
        return -1
    var v = 0
    for i in range(lo, hi):
        var c = Int(b[i])
        if c < 48 or c > 57:
            return -1
        v = v * 10 + c - 48
    return v if v < n_blocks else -1


def map_tensor_name(arch: String, n_blocks: Int, hf_name: String) -> String:
    """GGUF name for an HF name in arch at n_blocks; empty when unmapped."""
    for r in arch_rows(arch):
        if r[4]:
            var bid = block_id(hf_name, r[0], r[1], n_blocks)
            if bid >= 0:
                return r[2] + (String(bid) if r[5] else "") + r[3]
        elif hf_name == r[0]:
            return r[2]
    return ""


def expand_rows(arch: String, n_blocks: Int) -> List[Tuple[String, String]]:
    """Every (HF name, GGUF name) of the arch at n_blocks, in row order, duplicates included."""
    var out = List[Tuple[String, String]]()
    for r in arch_rows(arch):
        if r[4]:
            for bid in range(n_blocks):
                out.append((r[0] + String(bid) + r[1], r[2] + (String(bid) if r[5] else "") + r[3]))
        else:
            out.append((r[0], r[2]))
    return out^
