#!/usr/bin/env bash
# Fetch the Mojo libraries this repo imports ([tool.mojo] deps in pyproject.toml) into lib/, each pinned to a commit.
# A lib/<pkg> that exists already (a symlink to a local checkout) is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
while read -r pkg repo sha; do
  [[ -e lib/$pkg ]] && continue
  d=.work/deps/$repo
  rm -rf "$d" && git init -q "$d"
  { git -C "$d" fetch -q --depth 1 "https://github.com/amarbaro/$repo" "$sha" && git -C "$d" checkout -q FETCH_HEAD; } \
    || { echo "FAIL setup: fetch $repo $sha"; exit 1; }
  mkdir -p lib && ln -s "../$d/src/$pkg" "lib/$pkg"
done <<'PINS'
uregex mojo-uregex 73218ba2e7b9730ddb4bdcec29e97da977cadded
PINS
