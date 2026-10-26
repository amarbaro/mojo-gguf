# VENDORED COPY of mojo-uregex (src/uregex/__init__.mojo). This repo keeps a real file rather than a symlink or an
# -I path outside the tree, because a clone must build without anything outside it. Sync by hand if the
# upstream changes.
#
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from .pattern import Pattern, FullPattern, PatternT, Match, compile, findall, search, split_keep
