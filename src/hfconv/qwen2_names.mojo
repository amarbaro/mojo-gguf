"""Hugging Face Qwen2 tensor name -> GGUF name; empty when the converter would refuse it
("Can not map tensor"). One small table, to be swapped for the generated tensor_map lookup."""

def _layer_stem(rest: String) -> String:
    if rest == "input_layernorm":
        return "attn_norm"
    if rest == "post_attention_layernorm":
        return "ffn_norm"
    if rest == "self_attn.q_proj":
        return "attn_q"
    if rest == "self_attn.k_proj":
        return "attn_k"
    if rest == "self_attn.v_proj":
        return "attn_v"
    if rest == "self_attn.o_proj":
        return "attn_output"
    if rest == "mlp.gate_proj":
        return "ffn_gate"
    if rest == "mlp.up_proj":
        return "ffn_up"
    if rest == "mlp.down_proj":
        return "ffn_down"
    return String()

def qwen2_name(name: String) -> String:
    var stem: String
    var suffix: String
    if name.endswith(".weight"):
        suffix = ".weight"
        stem = String(name.removesuffix(".weight"))
    elif name.endswith(".bias"):
        suffix = ".bias"
        stem = String(name.removesuffix(".bias"))
    else:
        return String()
    if stem == "model.embed_tokens":
        return "token_embd" + suffix
    if stem == "model.norm":
        return "output_norm" + suffix
    if stem == "lm_head":
        return "output" + suffix
    if not stem.startswith("model.layers."):
        return String()
    var tail = String(stem.removeprefix("model.layers."))
    var dot = tail.find(".")
    if dot <= 0:
        return String()
    var bid = String(unsafe_from_utf8=Span(tail.as_bytes())[0:dot])
    for c in bid.as_bytes():
        if c < 48 or c > 57:
            return String()
    var new_stem = _layer_stem(String(unsafe_from_utf8=Span(tail.as_bytes())[dot + 1 : tail.byte_length()]))
    if new_stem.byte_length() == 0:
        return String()
    return "blk." + bid + "." + new_stem + suffix
