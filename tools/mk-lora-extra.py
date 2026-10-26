"""Synthetic PEFT adapter for the paths tests/mk-lora.py does not reach (U7 extra check): embedding adapter
(lora_embedding_A/B, transposed lora_a), norm pass-through, base_layer entry, mixed F32/F16/BF16, rank 32 (Q8_0 lora_a),
aLoRA tokens, generation_config.json, header order != data order.   usage: mk-lora-extra.py OUT_DIR"""
import json, struct, os, sys
import numpy as np
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)
def bf16(a): return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
def enc(dt, a):
    if dt == "F32": return a.astype(np.float32).tobytes()
    if dt == "F16": return a.astype(np.float16).tobytes()
    return bf16(a)
P = "base_model.model.model."
ts = [  # (name, dtype, array) in DATA order
 (P+"layers.3.self_attn.q_proj.lora_B.weight", "F32", rng.standard_normal((896, 32))*0.02),
 (P+"layers.3.self_attn.q_proj.lora_A.weight", "F16", rng.standard_normal((32, 896))*0.02),
 (P+"embed_tokens.lora_embedding_A", "BF16", rng.standard_normal((4, 151936))*0.02),
 (P+"embed_tokens.lora_embedding_B", "BF16", rng.standard_normal((896, 4))*0.02),
 (P+"norm.weight", "F16", rng.standard_normal((896,))),
 (P+"layers.0.input_layernorm.weight", "BF16", rng.standard_normal((896,))),
 (P+"layers.0.self_attn.o_proj.base_layer.weight", "F32", rng.standard_normal((2, 2))),
 (P+"layers.0.self_attn.o_proj.lora_A.weight", "BF16", rng.standard_normal((4, 896))*0.02),
 (P+"layers.0.self_attn.o_proj.lora_B.weight", "BF16", rng.standard_normal((896, 4))*0.02),
]
hdr, blobs, off = {}, [], 0
for n, dt, a in ts:
    b = enc(dt, a); hdr[n] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off+len(b)]}; blobs.append(b); off += len(b)
# header order reversed vs data order
h = json.dumps({k: hdr[k] for k in reversed(list(hdr))}, separators=(",",":")).encode(); h += b" " * ((8-len(h)%8)%8)
with open(out+"/adapter_model.safetensors","wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h)
    for b in blobs: f.write(b)
json.dump({"r": 4, "lora_alpha": 12.5, "peft_type":"LORA", "alora_invocation_tokens": [5, 6, 77]}, open(out+"/adapter_config.json","w"))
json.dump({"temperature": 0.7, "top_k": 20}, open(out+"/generation_config.json","w"))
