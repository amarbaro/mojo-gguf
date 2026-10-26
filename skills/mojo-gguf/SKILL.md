---
name: mojo-gguf
description: "Read, rewrite, quantize, convert and dequantize GGUF files from Mojo or the shell with llama.cpp's exact output: HF safetensors to GGUF (Qwen2), LoRA to GGUF, legacy GGML to GGUF, llama-quantize --pure types, gguf-py's dump, hash, set-metadata and convert-endian. Use when a GGUF tool must produce the same bytes as llama.cpp's Python scripts and quantizer without Python, or when Mojo code must read a GGUF header or dequantize tensor rows."
---

# mojo-gguf

GGUF tooling in Mojo 1.1. Each command is checked byte for byte (or bit for bit) against the pinned llama.cpp
(`1744c6bde`). Which gates run where is in `docs/gates.md`.

## Install

```
git clone <repo> && cd mojo-gguf && uv sync
mojo build -I src src/hf2gguf.mojo -o hf2gguf        # any command: src/<name>.mojo
```

As a library: `mojo build -I <repo>/src app.mojo`, then `from gguf import Gguf`.

## Use

```
gguf-rewrite IN OUT [--name STR] [--remove KEY]...
gguf-tool dump|hash|set-metadata|convert-endian ARGS      # argv of the gguf-py scripts
hf2gguf HF_DIR --outfile OUT --outtype bf16|q8_0           # Qwen2 only
quantize --pure IN OUT Q8_0|Q4_0|Q4_K|Q6_K|Q2_K|Q2_0|TQ1_0|TQ2_0 [NTHREADS]
gguf-dequant MODEL TENSOR ROW0 N OUT.f32
lora2gguf ADAPTER_DIR --base HF_DIR [--outfile OUT] [--outtype f32|f16|bf16|q8_0]
ggml2gguf --input IN --output OUT
```

```mojo
from gguf import Gguf

def main() raises:
    var g = Gguf("model.gguf")
    print(g.summary())
    print(g.find("general.architecture") >= 0)
```

## Rules

- `Gguf(path)` reads the header only. It raises on a missing file, a bad magic, a truncated file or an unknown value
  type; catch it, do not assume the file is sound.
- `hf2gguf` and `lora2gguf` support Qwen2 only. Every other architecture, dtype or tokenizer layout is a `FAIL`, never
  a guess: do not retry with other flags, convert with llama.cpp's script instead.
- `quantize` implements `--pure` only: every weight tensor gets the one type (with llama-quantize's block-size
  fallbacks), norms and biases are copied. It is not the mixed recipe of `llama-quantize Q4_K_M`.
- `gguf-tool` and the converters match the Python scripts on stdout, exit code and written bytes, including their
  quirks; `ggml2gguf` refuses a GGJT Q8_0 file because the script does.
- Never edit `src/gguf/consts.mojo`, `tensor_map.mojo` or `grids.mojo`: regenerate with `tools/gen-tables.py` and
  `tools/gen-grids.sh`.
- A change is done when `scripts/ci.sh` prints `PASS ci`; the converter, quantizer and dequant gates need
  `scripts/oracle-gates.sh` (a llama.cpp build at the pin, Python with torch, Qwen2.5-0.5B-Instruct).
