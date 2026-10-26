#!/usr/bin/env bash
# Fetch the pinned llama.cpp sources (oracle scripts and gguf-py) into .work/llama.cpp. No build: the CPU gates in
# scripts/ci.sh need only Python. The gates that use llama-quantize or libggml-base need your own build at the same pin.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=1744c6bde8d687ce9774b3b54e688eee0bfdf5b7
D=.work/llama.cpp
if [ -d "$D/.git" ] && [ "$(git -C "$D" rev-parse HEAD 2>/dev/null)" = "$PIN" ]; then echo "llama.cpp at pin $PIN"; exit 0; fi
rm -rf "$D"; mkdir -p "$D"
git -C "$D" init -q
git -C "$D" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$PIN" || { echo "FAIL fetch: llama.cpp $PIN"; exit 1; }
git -C "$D" -c advice.detachedHead=false checkout -q FETCH_HEAD
echo "llama.cpp at pin $PIN"
