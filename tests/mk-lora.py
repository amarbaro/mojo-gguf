"""Synthetic PEFT LoRA adapters for qwen2.5-0.5b-instruct (U7 fixtures), seeded, numpy only.
Adapter a: f32, r 8, alpha 16, q/k/v/o projections. Adapter b: bf16, r 4, alpha 8, gate/up/down projections.
usage: mk-lora.py HF_BASE_DIR OUT_DIR   (writes OUT_DIR/a and OUT_DIR/b)"""
import json, os, struct, sys
import numpy as np

base, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(base, "config.json")))
H, F, NL = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
KV = H // cfg["num_attention_heads"] * cfg["num_key_value_heads"]
shapes = {"q_proj": (H, H), "k_proj": (KV, H), "v_proj": (KV, H), "o_proj": (H, H),
          "gate_proj": (F, H), "up_proj": (F, H), "down_proj": (H, F)}
where = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
         "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}

def write_st(path, tensors):
    hdr, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = arr.tobytes() if dt == "F32" else (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in blobs:
            f.write(b)

for tag, dt, r, alpha, mods, seed in (("a", "F32", 8, 16, ["q_proj", "k_proj", "v_proj", "o_proj"], 1),
                                      ("b", "BF16", 4, 8, ["gate_proj", "up_proj", "down_proj"], 2)):
    rng = np.random.default_rng(seed)
    d = os.path.join(out, tag); os.makedirs(d, exist_ok=True)
    ts = {}
    for l in range(NL):
        for m in mods:
            o, i = shapes[m]
            p = f"base_model.model.model.layers.{l}.{where[m]}.{m}"
            ts[p + ".lora_A.weight"] = (dt, (rng.standard_normal((r, i)) * 0.02).astype(np.float32))
            ts[p + ".lora_B.weight"] = (dt, (rng.standard_normal((o, r)) * 0.02).astype(np.float32))
    write_st(os.path.join(d, "adapter_model.safetensors"), ts)
    json.dump({"base_model_name_or_path": base, "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r,
               "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none", "target_modules": mods},
              open(os.path.join(d, "adapter_config.json"), "w"), indent=2)
    print(tag, len(ts), "tensors", dt)
