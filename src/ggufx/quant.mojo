"""GGML tensor types: name, block elements and block bytes, as gguf-py's GGML_QUANT_SIZES (pinned 1744c6bde).

`tensor_bytes` is gguf-py's `n_elems * type_size // block_size`; an unknown id raises like the enum lookup.
"""

@fieldwise_init
struct Row(Copyable, Movable):
    var id: Int
    var name: String
    var block: Int  # elements per block
    var size: Int  # bytes per block


def rows() -> List[Row]:
    var t = List[Row]()
    t.append(Row(0, "F32", 1, 4))
    t.append(Row(1, "F16", 1, 2))
    t.append(Row(2, "Q4_0", 32, 18))
    t.append(Row(3, "Q4_1", 32, 20))
    t.append(Row(6, "Q5_0", 32, 22))
    t.append(Row(7, "Q5_1", 32, 24))
    t.append(Row(8, "Q8_0", 32, 34))
    t.append(Row(9, "Q8_1", 32, 40))
    t.append(Row(10, "Q2_K", 256, 84))
    t.append(Row(11, "Q3_K", 256, 110))
    t.append(Row(12, "Q4_K", 256, 144))
    t.append(Row(13, "Q5_K", 256, 176))
    t.append(Row(14, "Q6_K", 256, 210))
    t.append(Row(15, "Q8_K", 256, 292))
    t.append(Row(16, "IQ2_XXS", 256, 66))
    t.append(Row(17, "IQ2_XS", 256, 74))
    t.append(Row(18, "IQ3_XXS", 256, 98))
    t.append(Row(19, "IQ1_S", 256, 50))
    t.append(Row(20, "IQ4_NL", 32, 18))
    t.append(Row(21, "IQ3_S", 256, 110))
    t.append(Row(22, "IQ2_S", 256, 82))
    t.append(Row(23, "IQ4_XS", 256, 136))
    t.append(Row(24, "I8", 1, 1))
    t.append(Row(25, "I16", 1, 2))
    t.append(Row(26, "I32", 1, 4))
    t.append(Row(27, "I64", 1, 8))
    t.append(Row(28, "F64", 1, 8))
    t.append(Row(29, "IQ1_M", 256, 56))
    t.append(Row(30, "BF16", 1, 2))
    t.append(Row(34, "TQ1_0", 256, 54))
    t.append(Row(35, "TQ2_0", 256, 66))
    t.append(Row(39, "MXFP4", 32, 17))
    t.append(Row(40, "NVFP4", 64, 36))
    t.append(Row(41, "Q1_0", 128, 18))
    t.append(Row(42, "Q2_0", 64, 18))
    return t^


def row(t: Int) raises -> Row:
    for r in rows():
        if r.id == t:
            return r.copy()
    raise Error("unknown GGML tensor type " + String(t))


def tensor_bytes(t: Int, n_elems: Int) raises -> Int:
    var r = row(t)
    return n_elems * r.size // r.block
