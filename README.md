# mojo-gguf

<!-- badges -->
[![ci](https://github.com/amarbaro/mojo-gguf/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/amarbaro/mojo-gguf/actions/workflows/ci.yml) [![latest release](https://img.shields.io/github/v/release/amarbaro/mojo-gguf)](https://github.com/amarbaro/mojo-gguf/releases/latest) [![license](https://img.shields.io/github/license/amarbaro/mojo-gguf)](https://github.com/amarbaro/mojo-gguf/blob/main/LICENSE)
<!-- /badges -->


GGUF tools in Mojo that write the bytes llama.cpp's own tools write: a GGUF reader and writer, the four gguf-py
scripts (`dump`, `hash`, `set-metadata`, `convert-endian`), the Qwen2 safetensors-to-GGUF converter, the LoRA and
legacy-GGML converters, eight `llama-quantize --pure` types and a dequantizer for 19 types. Every command is
compared with llama.cpp pinned at `1744c6bde`: converter and quantizer outputs are byte-identical on
Qwen2.5-0.5B-Instruct, dequantized rows are bit-equal to `ggml`'s `to_float`. Linux x86-64, CPU only.

```mojo
from ggufx import Gguf

def main() raises:
    var g = Gguf("model.gguf")                       # header and tensor table only, no tensor data is read
    print(g.summary())                               # kv 22 tensors 11 alignment 32 data_offset 1472
    print("arch present:", g.find("general.architecture") >= 0)
```

```
scripts/deps.sh                                                              # mojo-uregex into lib/, pinned
mojo build -I src -I lib src/hf2gguf.mojo -o hf2gguf
./hf2gguf Qwen2.5-0.5B-Instruct --outfile qwen.bf16.gguf --outtype bf16      # 0.26 s, same bytes as the script
mojo build -I src -I lib src/quantize.mojo -o quantize
./quantize --pure qwen.bf16.gguf qwen.q4_k.gguf Q4_K                         # same bytes as llama-quantize --pure
```

## Install

Requires Mojo 1.2 (`uv sync` installs it) and Linux x86-64, no linker flags. The tokenizer converter imports
[mojo-uregex](https://github.com/amarbaro/mojo-uregex): `scripts/deps.sh` fetches it into `lib/`, pinned to a commit.

| Way | Command |
|---|---|
| A command | `mojo build -I path/to/mojo-gguf/src -I path/to/mojo-gguf/lib path/to/mojo-gguf/src/<command>.mojo -o <command>` |
| The library | `mojo build -I path/to/mojo-gguf/src app.mojo`, then `from ggufx import Gguf` |
| Vendor | copy `src/ggufx/` into your project |

## What you get

| Command | Replaces | Scope |
|---|---|---|
| `gguf-rewrite` | `gguf_new_metadata.py` (`--name`, `--remove`) | GGUF v3 little-endian reader and writer; no edit gives the input's bytes back |
| `gguf-tool dump\|hash\|set-metadata\|convert-endian` | the four gguf-py scripts | same argv, stdout, exit code and written bytes |
| `hf2gguf` | `convert_hf_to_gguf.py` | Qwen2 (`Qwen2ForCausalLM`, `Qwen2Model`), `--outtype bf16\|q8_0` |
| `quantize --pure` | `llama-quantize --pure` | `Q8_0 Q4_0 Q4_K Q6_K Q2_K Q2_0 TQ1_0 TQ2_0` |
| `gguf-dequant` | `to_float` in `ggml-quants.c` | 19 types: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ2_XXS IQ1_S IQ1_M IQ4_NL IQ4_XS TQ1_0 Q2_0 |
| `lora2gguf` | `convert_lora_to_gguf.py` | PEFT adapters on Qwen2 bases, `--outtype f32\|f16\|bf16\|q8_0` |
| `ggml2gguf` | `convert_llama_ggml_to_gguf.py` | GGML, GGMF v1, GGJT v1 to v3 |
| `hf-update` | the check in `convert_hf_to_gguf_update.py` | pre-tokenizer fingerprint of a local `tokenizer.json`, no Python |

The generated tables (`src/ggufx/consts.mojo`, `tensor_map.mojo`, `grids.mojo`) come from gguf-py and `ggml-common.h`
through `tools/gen-tables.py` and `tools/gen-grids.sh`, and a gate fails when they drift. Commands and library calls
are in [docs/api.md](docs/api.md); which gate checks what, where it runs and its last receipt is in
[docs/gates.md](docs/gates.md).

## Performance

One run each of the gates' own timing lines, Ryzen 7 7800X3D (16 threads), Linux, a machine that was running other
jobs. The oracle column is llama.cpp's tool on the same input. Receipts: [docs/receipts/](docs/receipts/).

| Task (Qwen2.5-0.5B-Instruct) | mojo-gguf | llama.cpp tool | Receipt |
|---|---|---|---|
| `hf2gguf` bf16, 994 MB out | 0.26 s, 33 MiB peak RSS | `convert_hf_to_gguf.py` 12.55 s, 2252 MiB | `u3.txt` |
| `hf2gguf` q8_0 | 0.34 s, 33 MiB | 6.77 s, 2051 MiB | `u3.txt` |
| `quantize --pure` Q8_0 | 0.82 s, 49 MiB | `llama-quantize` 1.06 s, 1002 MiB | `u4.txt` |
| `quantize --pure` Q4_K | 0.71 s, 43 MiB | 1.29 s, 954 MiB | `u4.txt` |
| `quantize --pure` Q2_K | 0.61 s, 41 MiB | 1.17 s, 938 MiB | `u4.txt` |
| `lora2gguf`, 2 adapters x 3 out types | 0.01 to 0.04 s, 16 to 17 MiB | `convert_lora_to_gguf.py` 3.2 to 3.6 s, 1184 to 1185 MiB | `u7.txt` |
| `gguf-rewrite`, 994 MB bf16 file | 0.16 s, 13 MiB | | `u1.txt` |

Tensors are streamed through fixed buffers and consumed source pages are handed back, which is where the memory
difference comes from. The other quantizer types are in `u4.txt`: 0.5 to 0.9 s against 0.8 to 1.3 s for
`llama-quantize`.

## Limits

- Scope of "byte-identical": the Qwen2.5-0.5B-Instruct outputs of the converter and quantizer gates, the synthetic
  adapters and legacy files in the gates, and gguf-py-written fixtures. It is not a proof for every model or every
  input.
- `hf2gguf` and `lora2gguf` convert Qwen2 only. Other architectures, rope scaling, MoE keys, `head_dim`, source
  dtypes other than BF16, F16 and F32, sentence-transformers pooling, named chat templates and tokenizers without
  `tokenizer.json` are a `FAIL`, never a guess.
- `quantize` implements `--pure` only (every weight gets one type, with llama-quantize's block-size fallbacks). No
  mixed recipe such as `Q4_K_M`, no importance matrices, no types beyond the eight.
- `gguf-dequant` covers the 19 types above; any other (Q5_1, TQ2_0, IQ2_XS, IQ3_S, MXFP4 and more) is a `FAIL`.
- `ggml2gguf` refuses a GGJT Q8_0 input because the script does, and does not support `--model-metadata-dir` or
  `--vocab-dir`.
- The gates cover GGUF v3 little-endian inputs and `convert-endian ... big`.
- CI runs on synthetic fixtures against the pinned llama.cpp's Python. The converter, quantizer, dequant and LoRA
  gates need a llama.cpp build, Python with torch and the Qwen2.5-0.5B model, so they run locally
  (`scripts/oracle-gates.sh`), not on a hosted runner. The GPU is not used.
- `gguf-tool` does not include the GUI editor of gguf-py.
