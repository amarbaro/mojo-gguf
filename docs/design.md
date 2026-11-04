# Design

## Layout
- `src/gguf/`: the GGUF reader and writer (`model.mojo`), value decoding (`values.mojo`), the ggml type tables and dequantizers (`quant.mojo`,
  `dequant.mojo`, `quants.mojo`), the quantizer grids, `sha.mojo` for `hash`, and `fmt.mojo` for the text of `dump`.
- `src/st/`: safetensors (every shard mmapped read-only) and the JSON reader for its headers and for
  `tokenizer.json`. `src/hfconv/`: the Hugging Face to GGUF rules (tensor naming, f16/bf16, tokenizer). The pre-tokenizer
  patterns run on `mojo-uregex`, a `[tool.mojo] deps` entry that `scripts/deps.sh` fetches into `lib/`, pinned to a commit.
- Each command is one file in `src/` with a `main()` (`hf2gguf`, `quantize`, `lora2gguf`, `ggml2gguf`, `gguf_tool`, `gguf_rewrite`, `gguf_dequant`,
  `hf_update`). They are built with `mojo build -I src -I lib`; the three packages are also precompiled for the conda recipe.

## Target: the same bytes
The reference is llama.cpp pinned at `1744c6bde` (its `convert_*.py`, `llama-quantize --pure`, `gguf-py` and `ggml-quants.c`). A command is done when
its output is byte-identical with the reference on Qwen2.5-0.5B-Instruct and on generated fixtures, not when it looks right. `docs/gates.md` lists the
gate per command and what each needs (the hermetic ones run in `scripts/ci.sh`; the model and llama.cpp-build gates run locally).

## Memory
- The GGUF reader keeps a 256 KiB sliding window and records each key's file span; it never loads tensor data. Tensor rows are `pread` on demand.
- Safetensors shards are mmapped read-only.
- The quantizer splits its rows over NTHREADS threads (`par.mojo`) and the dequantizer works in chunks of rows; both write with `pwrite`.

## Bad input
Files are untrusted: lengths and counts at or above 2^63, offsets outside the file, tensors with no dimensions, negative token ids and invalid
UTF-8 or UTF-16 surrogates raise a `FAIL` line and exit 1 (or decode lossily, for strings that are only printed) rather than index out of range.
`tests/negative-gate.sh` writes each of these by hand and also checks that OUT equal to IN is refused.

## Dependencies
Only `mojo` (the Python packages in `pyproject.toml` are oracles: gguf-py needs numpy, pyyaml and tqdm, the HF scripts need tokenizers). No linker
flags (README).

## Rejected
No GPU and no llama.cpp binding: the tools stand in for llama.cpp's scripts and `llama-quantize`, they do not call them. Only Qwen2 is converted;
the quantizer covers the eight types listed in the README.
