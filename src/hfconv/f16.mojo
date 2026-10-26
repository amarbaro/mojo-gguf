"""F16 payload converter, bit-exact with gguf-py (numpy `astype(float16)`, round to nearest even) from BF16 / F16 / F32
rows; same calling convention as hfconv.quant: source base address, element range, destination address."""
from std.memory import bitcast
from st.safetensors import ST_F16
from .quant import load_f32

def emit_f16(dtype: Int, base: Int, start: Int, count: Int, dst: Int):
    var o = Pointer[Scalar[DType.float16], MutAnyOrigin](unsafe_from_address=dst)
    if dtype == ST_F16:
        var s = Pointer[Scalar[DType.float16], ImmutAnyOrigin](unsafe_from_address=base)
        for i in range(count):
            o[unsafe_offset=i] = s[unsafe_offset=start + i]
        return
    for i in range(count):
        o[unsafe_offset=i] = load_f32(dtype, base, start + i).cast[DType.float16]()
