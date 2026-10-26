#!/usr/bin/env bash
# U8 gate: `hf-update DIR` prints `ntok N chkhsh H pre NAME` for a local HF tokenizer dir: the pinned converter's
# pre-tokenizer fingerprint (sha256 of str(tokenizer.encode(chktxt))) and its table name (`unknown` when absent),
# computed by our own tokenizer, no Python, no network. Oracle: tests/u8-oracle.py with transformers, cached in
# .work/oracle/u8/summary.txt (+ <dir>.ids, the expected token ids, for diagnosis).
#   tests/u8-gate.sh    exit 0 only on PASS 9/9
source "$(dirname "$0")/gate-lib.sh"
W=$PWD/.work/u8; rm -rf "$W"; mkdir -p "$W"; UO=$MAIN/.work/oracle/u8; SUMMARY=${U8_SUMMARY:-$UO/summary.txt}
check llama-cpp-pinned pinned
build() { "$MOJO" build src/hf_update.mojo -I src -o "$W/hf-update" > "$W/build.log" 2>&1; }
check build build
one() { local line=$1 d=${1%% ntok *}; local want=${line#"$d "}
  local got; got=$("$W/hf-update" "$d" 2> "$W/$(basename "$d").err") || return 1
  [ "$got" = "$want" ] || { echo "     $(basename "$d"): got '$got' want '$want'"; return 1; }; }
while IFS= read -r line; do d0=${line%% ntok *}; check "chkhsh-$(basename "$(dirname "$d0")")-$(basename "$d0")" one "$line"; done < "$SUMMARY"
finish
