"""Prints the U2 tables as one line of compact JSON with sorted keys, byte-equal to tests/u2-oracle.py.

Every value comes from the generated tables (gguf/consts.mojo, gguf/tensor_map.mojo) through gguf/tensor_lookup.mojo.
No trailing newline, as json.dump writes none.
"""
from gguf.consts import value_types, quant_types, quant_sizes, arch_names, tensor_names, arch_tensors, gguf_keys
from gguf.tensor_map import mapped_arches
from gguf.tensor_lookup import expand_rows


def quote(s: String) -> String:
    return '"' + s + '"'


def member(key: String, value: String) -> String:
    return quote(key) + ":" + value


def obj(members: List[String]) -> String:
    var out = String("{")
    for i in range(len(members)):
        if i > 0:
            out += ","
        out += members[i]
    return out + "}"


def arr(items: List[String]) -> String:
    var out = String("[")
    for i in range(len(items)):
        if i > 0:
            out += ","
        out += items[i]
    return out + "]"


def sort_by_key(mut pairs: List[Tuple[String, String]]):
    for i in range(1, len(pairs)):
        var j = i
        while j > 0 and pairs[j][0] < pairs[j - 1][0]:
            pairs.swap_elements(j, j - 1)
            j -= 1


def name_map_obj() raises -> String:
    var arches = List[String]()
    for arch in mapped_arches():
        var pairs = expand_rows(arch, 2)
        sort_by_key(pairs)
        var members = List[String]()
        for i in range(len(pairs)):
            if i == 0 or pairs[i][0] != pairs[i - 1][0]:
                members.append(member(pairs[i][0], quote(pairs[i][1])))
            elif pairs[i][1] != pairs[i - 1][1]:
                raise Error("name map key " + pairs[i][0] + " maps to two values")
        arches.append(member(arch, obj(members)))
    return obj(arches)


def main() raises:
    var at = List[String]()
    var names = List[String]()
    var arch = String("")
    for r in arch_tensors():
        if r[0] != arch and len(names) > 0:
            at.append(member(arch, arr(names)))
            names = List[String]()
        arch = r[0]
        names.append(quote(r[1]))
    at.append(member(arch, arr(names)))

    var kv = List[String]()
    for r in gguf_keys():
        kv.append(member(r[0], quote(r[1])))

    var qs = List[String]()
    for r in quant_sizes():
        qs.append(member(r[0], "[" + String(r[1]) + "," + String(r[2]) + "]"))

    var qt = List[String]()
    for r in quant_types():
        qt.append(member(r[0], String(r[1])))

    var tn = List[String]()
    for r in tensor_names():
        tn.append(member(r[0], quote(r[1])))

    var vt = List[String]()
    for r in value_types():
        vt.append(member(r[0], String(r[1])))

    var an = List[String]()
    for r in arch_names():
        an.append(member(r[0], quote(r[1])))

    var top = List[String]()
    top.append(member("arch_names", obj(an)))
    top.append(member("arch_tensors", obj(at)))
    top.append(member("keys", obj(kv)))
    top.append(member("name_map", name_map_obj()))
    top.append(member("quant_sizes", obj(qs)))
    top.append(member("quant_types", obj(qt)))
    top.append(member("tensor_names", obj(tn)))
    top.append(member("value_types", obj(vt)))
    print(obj(top), end="")
