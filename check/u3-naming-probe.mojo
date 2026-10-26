"""Prints the hfconv.naming result for each `ID TOTAL` pair on argv, one line each (checked by u3-naming-check.sh)."""
from std.sys import argv
from hfconv.naming import components, size_label, id_to_title

def main() raises:
    var a = argv()
    var i = 1
    while i + 1 < len(a):
        var total = Int(String(a[i + 1]))
        var c = components(String(a[i]), total)
        print(String(a[i]), total, "|", id_to_title(c.full), "|", c.org, "|", c.basename, "|", c.finetune, "|", c.version, "|", c.size_label, "|", size_label(total))
        i += 2
