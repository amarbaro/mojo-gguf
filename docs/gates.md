# Gates

A command is done when a gate that exercises it the way a user runs it passes against the pinned llama.cpp
(`1744c6bde`, fetched by `scripts/fetch-llamacpp.sh`). Gates print `PASS n/n` and exit 0, or `FAIL ...` and exit 1.
The last run of each is in [receipts/](receipts/).

## In CI (`scripts/ci.sh`, no model, no llama.cpp build)

| Gate | Checks | Result |
|---|---|---|
| `tests/core-gate.sh` | On two gguf-py-written fixtures (`tools/mkfixture.py`: all scalar KV types, arrays, ten tensor types, alignment 32 and 64): `gguf-rewrite` returns the input's bytes, prints gguf-py's summary numbers, and its `--name` and `--remove` edits equal `gguf_new_metadata.py`; `gguf-tool` dump (text, json, markdown, no tensors), hash, set-metadata and convert-endian equal the four scripts in stdout, exit code and written bytes; `gguf-dequant` is bit-equal to gguf-py's `dequantize` on F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q8_0 TQ1_0. Every case states the exit code both tools must return (`EXP`, default 0): the fixtures' I8 tensor is refused by `gguf_convert_endian.py` (exit 1, a message), a byteswappable fixture is the convert-endian success; a file truncated at 400 bytes is an exit 1 with a stderr message from all four `gguf-tool` commands, `gguf-rewrite` and `gguf-dequant`; a negative control checks that two tools failing alike do not pass for a success | `PASS 26/26` |
| `tests/gguf-cmp-selftest.sh` | the comparison tool passes a file against itself and fails on one flipped data byte and one flipped KV byte | `PASS 4/4` |
| `tests/u2-gate.sh` | the generator is deterministic, the committed tables equal a fresh generation, and a Mojo dump of the tables equals gguf-py's | `PASS 6/6` |
| `tests/u2-lookup-check.sh` | `map_tensor_name` equals gguf-py's `TensorNameMap` on every key at 2 and 5 blocks | `PASS 18527/18527` |
| `tests/u9-gate.sh`, `tests/u9-parity.sh` | `ggml2gguf` equals `convert_llama_ggml_to_gguf.py` on 3 legacy files, refuses the GGJT Q8_0 file as the script does, and agrees with it on 24 more container, version, dtype, vocabulary and argv cases | `PASS 6/6`, `PASS 24/24` |
| `check/u3-naming-check.sh` | the model-name heuristics equal gguf-py's `Metadata` on 25 ids x 3 parameter counts | `PASS 75/75` |
| `tools/u8-synth-check.py` | `hf-update` equals HF `tokenizers` (ntok and sha256 of the ids) on 31 synthetic tokenizers | `PASS 31/31` |
| `examples/quickstart.mojo` | builds and runs on a fixture | exit 0 |

## Local (`scripts/oracle-gates.sh`)

These compare with llama.cpp's own outputs for Qwen2.5-0.5B-Instruct, so they need a llama.cpp build at the pin
(`llama-quantize`, `libggml-base.so`), Python with torch and the model. `tools/mkoracle.sh` writes the reference files.

| Gate | Checks | Result |
|---|---|---|
| `tests/u1-gate.sh` | `gguf-rewrite` returns the bytes of 10 reference files, its summary line equals gguf-py's on one, and one edit (`--name`, `--remove`) equals `gguf_new_metadata.py` | `PASS 12/12` |
| `tests/u3-gate.sh` | `hf2gguf` bf16 and q8_0 are byte-identical to `convert_hf_to_gguf.py` | `PASS 4/4` |
| `tests/u4-gate.sh` | `quantize --pure` is byte-identical to `llama-quantize --pure` for the 8 types | `PASS 10/10` |
| `tests/u5-gate.sh` | `gguf-tool` equals the scripts on 10 cases (dump x4, hash x2, set-metadata x2, convert-endian x2) with the exit code stated per case, plus a truncated file per command and the negative control | `PASS 17/17` |
| `tests/u6-gate.sh` | `gguf-dequant` rows are bit-equal to `to_float` on the first and last tensor of every type in 7 reference files (rows 0, last and four seeded random ones, plus a flipped-byte falsifier per type) | `PASS 3/3` (9 types) |
| `tools/u6-extra-types.py` | random raw blocks of all 19 types, offset reads and a 20 MB streaming read against `to_float` | `PASS extra types` |
| `tests/u7-gate.sh`, `tools/lora-extra-check.sh` | `lora2gguf` is byte-identical to the script on 2 synthetic adapters x 3 out types, and on 6 more cases (embedding transpose, norms, F16, mixed dtypes, naming, outfile templates) | `PASS 8/8`, `PASS 6/6` |
| `tests/u8-gate.sh` | `hf-update` prints the converter's `chkhsh` and pre-tokenizer name for the Qwen2.5 tokenizer | `PASS 3/3` |

`tests/u1-gate.sh` and `tests/u5-gate.sh` add two cases when `FLASH_GGUF` points to a GGUF with large tokenizer
arrays. The receipts here were taken without it.

## Mutants

The core gate was run against five one-line mutants of the library; each fails it: a Q4_0 dequant offset, a Q8_0
dequant scale, the writer's padding rule (rounding up a size that is already aligned), a `dump` header line, and a
SHA-1 round constant. A sixth, a padding rule that is wrong only for sizes of 1 mod alignment, survives: the inputs
gguf-py writes never contain such a size. Details in [receipts/mutants.txt](receipts/mutants.txt).
