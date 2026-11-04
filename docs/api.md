# API

Every command below is checked against the pinned llama.cpp (`1744c6bde`) by a gate in `tests/` (list in
[gates.md](gates.md)); the library calls are the ones the commands use. Build any command with
`scripts/deps.sh`, then `mojo build -I src -I lib src/<file>.mojo -o <name>`. All commands print `FAIL <command>: <reason>` and exit 1 on bad input.

## Commands

| Command | Source | Replaces | Behaviour |
|---|---|---|---|
| `gguf-rewrite IN OUT [--name STR] [--remove KEY]...` | `src/gguf_rewrite.mojo` | `gguf_new_metadata.py` (name, remove) | Parses a GGUF v3 little-endian file and writes it again. With no edit the output is byte-identical to the input. First stdout line: `kv K tensors T alignment A data_offset D`. |
| `gguf-tool dump\|hash\|set-metadata\|convert-endian ARGS` | `src/gguf_tool.mojo` | `gguf_dump.py`, `gguf_hash.py`, `gguf_set_metadata.py`, `gguf_convert_endian.py` | Same argv, stdout, exit code and written bytes as the four scripts. |
| `hf2gguf HF_DIR --outfile OUT --outtype bf16\|q8_0` | `src/hf2gguf.mojo` | `convert_hf_to_gguf.py` | Qwen2 only (`Qwen2ForCausalLM`, `Qwen2Model`). Safetensors in, GGUF out, byte-identical to the script. |
| `quantize --pure IN OUT TYPE [NTHREADS]` | `src/quantize.mojo` | `llama-quantize --pure` | `TYPE` is one of `Q8_0 Q4_0 Q4_K Q6_K Q2_K Q2_0 TQ1_0 TQ2_0` (`Q4_K_M` is accepted as `Q4_K`). Same block-size fallbacks as llama-quantize. |
| `gguf-dequant MODEL TENSOR ROW0 N OUT.f32` | `src/gguf_dequant.mojo` | `to_float` of `ggml-quants.c` | Writes `N` rows of one tensor as little-endian f32. 19 types, bit-equal to llama.cpp. |
| `lora2gguf ADAPTER_DIR --base HF_DIR [--outfile OUT] [--outtype f32\|f16\|bf16\|q8_0]` | `src/lora2gguf.mojo` | `convert_lora_to_gguf.py` | PEFT adapters for Qwen2 bases, byte-identical to the script. |
| `ggml2gguf --input IN --output OUT [--name S] [--desc S] [--gqa N] [--eps E] [--context-length N]` | `src/ggml2gguf.mojo` | `convert_llama_ggml_to_gguf.py` | Legacy GGML, GGMF v1 and GGJT v1 to v3 to GGUF, byte-identical. A GGJT Q8_0 input is refused (exit 1) exactly as the script refuses it. |
| `hf-update DIR` | `src/hf_update.mojo` | the check in `convert_hf_to_gguf_update.py` | Prints `ntok N chkhsh H pre NAME` for a local `tokenizer.json`: the pre-tokenizer fingerprint the converter uses, computed without Python. |
| `type-census FILE.gguf...` | `tools/type-census.mojo` | | Tensor count and bytes per ggml type, per file and in total, from the header and tensor table only. |

Not supported, each is a `FAIL` and never a guess: other architectures than Qwen2 in `hf2gguf` and `lora2gguf`, rope
scaling, MoE keys, source dtypes other than BF16, F16 and F32, tokenizers without `tokenizer.json`, `ggml2gguf
--model-metadata-dir` and `--vocab-dir` (exit 2), `quantize` without `--pure`, importance matrices.

## Library

`from ggufx import Gguf` (add `-I src`).

| Call | Behaviour |
|---|---|
| `Gguf(path)` | Opens and parses a GGUF file: header, KV records, tensor infos. Tensor data is never loaded. Raises on a missing file, a bad magic, a truncated file or an unknown value type. |
| `g.kvs`, `g.tensors` | `List[KV]` (`key`, `vtype`, `start`, `end`, `types`, `count`) and `List[TensorInfo]` (`name`, `dims`, `ttype`, `offset`, `at`). Offsets are file offsets for KVs and data-section offsets for tensors. |
| `g.alignment`, `g.data_offset`, `g.fsize`, `g.big` | Alignment, start of the data section, file size, and whether the file was written for a big-endian host. |
| `g.find(key)` | Index into `g.kvs`, or -1. |
| `g.summary()` | `kv K tensors T alignment A data_offset D`, the same numbers gguf-py's `GGUFReader` reports. |
| `g.rewrite(out_path, name, remove)` | Writes the model. With no edit (`name` empty, `remove` empty) the layout is the source's. |

`from ggufx.dequant import deq_rows, quant_block`

| Call | Behaviour |
|---|---|
| `quant_block(t)` | `(elements, bytes)` of one block of ggml type `t`; `(0, 0)` when there is no dequantizer. |
| `deq_rows(t, src, ne0, nrows, dst)` | Dequantizes `nrows` contiguous rows of `ne0` elements into `nrows * ne0` f32, rows split over threads. Each f32 is the C reference's operation in the C reference's order. Raises on an unsupported type or a row that is not a whole number of blocks. |

`from ggufx.tensor_lookup import map_tensor_name`

| Call | Behaviour |
|---|---|
| `map_tensor_name(arch, n_blocks, hf_name)` | GGUF name for an HF tensor name (no `.weight` suffix) in `arch`, or an empty string. Tables come from gguf-py's `TensorNameMap` through `tools/gen-tables.py`. |

`from ggufx.par import parallelize`

| Call | Behaviour |
|---|---|
| `parallelize(task, ctx, items, nthreads=0)` | Runs `task(ctx, i)` for `i` in `0..items` on pthreads (all cores when `nthreads` is 0) and returns when all are done. `ctx` is the address of a struct the task reads. |

## Generated files

`src/ggufx/consts.mojo` and `src/ggufx/tensor_map.mojo` are written by `tools/gen-tables.py GGUFPY OUT_DIR` from the
pinned gguf-py, and `src/ggufx/grids.mojo` by `tools/gen-grids.sh` from the pinned `ggml-common.h`. Do not edit them:
`tests/u2-gate.sh` fails when the committed files differ from a fresh generation.
