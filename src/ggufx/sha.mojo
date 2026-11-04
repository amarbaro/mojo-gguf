"""Streaming SHA-1 and SHA-256 (FIPS 180-4) for gguf_hash: feed chunks with update, read the digest once.

Hashed tensors are far larger than RAM budget, so update takes a chunk and keeps one 64-byte block.
"""
from std.bit import rotate_bits_left, rotate_bits_right


def _be32(b: List[UInt8], i: Int) -> UInt32:
    return (UInt32(b[i]) << 24) | (UInt32(b[i + 1]) << 16) | (UInt32(b[i + 2]) << 8) | UInt32(b[i + 3])


struct Sha256(Movable):
    var h: List[UInt32]
    var k: List[UInt32]
    var block: List[UInt8]
    var w: List[UInt32]
    var used: Int
    var total: Int

    def __init__(out self):
        self.h = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
        self.k = [0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA, 0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070, 0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2]
        self.block = List[UInt8](length=64, fill=0)
        self.w = List[UInt32](length=64, fill=0)
        self.used = 0
        self.total = 0

    def _compress(mut self, src: List[UInt8], at: Int):
        for t in range(16):
            self.w[t] = _be32(src, at + 4 * t)
        for t in range(16, 64):
            var s0 = rotate_bits_right[7](self.w[t - 15]) ^ rotate_bits_right[18](self.w[t - 15]) ^ (self.w[t - 15] >> 3)
            var s1 = rotate_bits_right[17](self.w[t - 2]) ^ rotate_bits_right[19](self.w[t - 2]) ^ (self.w[t - 2] >> 10)
            self.w[t] = self.w[t - 16] + s0 + self.w[t - 7] + s1
        var a = self.h[0]
        var b = self.h[1]
        var c = self.h[2]
        var d = self.h[3]
        var e = self.h[4]
        var f = self.h[5]
        var g = self.h[6]
        var hh = self.h[7]
        for t in range(64):
            var big1 = rotate_bits_right[6](e) ^ rotate_bits_right[11](e) ^ rotate_bits_right[25](e)
            var ch = (e & f) ^ (~e & g)
            var t1 = hh + big1 + ch + self.k[t] + self.w[t]
            var big0 = rotate_bits_right[2](a) ^ rotate_bits_right[13](a) ^ rotate_bits_right[22](a)
            var maj = (a & b) ^ (a & c) ^ (b & c)
            var t2 = big0 + maj
            hh = g
            g = f
            f = e
            e = d + t1
            d = c
            c = b
            b = a
            a = t1 + t2
        self.h[0] += a
        self.h[1] += b
        self.h[2] += c
        self.h[3] += d
        self.h[4] += e
        self.h[5] += f
        self.h[6] += g
        self.h[7] += hh

    def update(mut self, data: List[UInt8], n: Int):
        var i = 0
        if self.used > 0:
            while i < n and self.used < 64:
                self.block[self.used] = data[i]
                self.used += 1
                i += 1
            if self.used == 64:
                self._compress(self.block.copy(), 0)
                self.used = 0
        while n - i >= 64:
            self._compress(data, i)
            i += 64
        while i < n:
            self.block[self.used] = data[i]
            self.used += 1
            i += 1
        self.total += n

    def digest(mut self) -> List[UInt8]:
        """Pad the open block (0x80, zeros, 64-bit big-endian bit length), compress, return 32 bytes."""
        var bits = UInt64(self.total) * 8
        self.block[self.used] = 0x80
        for i in range(self.used + 1, 64):
            self.block[i] = 0
        if self.used >= 56:
            self._compress(self.block.copy(), 0)
            for i in range(64):
                self.block[i] = 0
        for k in range(8):
            self.block[56 + k] = UInt8((bits >> UInt64(8 * (7 - k))) & 0xFF)
        self._compress(self.block.copy(), 0)
        var out = List[UInt8]()
        for i in range(8):
            for k in range(4):
                out.append(UInt8((self.h[i] >> UInt32(8 * (3 - k))) & 0xFF))
        return out^


struct Sha1(Movable):
    var h: List[UInt32]
    var block: List[UInt8]
    var w: List[UInt32]
    var used: Int
    var total: Int

    def __init__(out self):
        self.h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
        self.block = List[UInt8](length=64, fill=0)
        self.w = List[UInt32](length=80, fill=0)
        self.used = 0
        self.total = 0

    def _compress(mut self, src: List[UInt8], at: Int):
        for t in range(16):
            self.w[t] = _be32(src, at + 4 * t)
        for t in range(16, 80):
            self.w[t] = rotate_bits_left[1](self.w[t - 3] ^ self.w[t - 8] ^ self.w[t - 14] ^ self.w[t - 16])
        var a = self.h[0]
        var b = self.h[1]
        var c = self.h[2]
        var d = self.h[3]
        var e = self.h[4]
        for t in range(80):
            var f: UInt32
            var k: UInt32
            if t < 20:
                f = (b & c) | (~b & d)
                k = 0x5A827999
            elif t < 40:
                f = b ^ c ^ d
                k = 0x6ED9EBA1
            elif t < 60:
                f = (b & c) | (b & d) | (c & d)
                k = 0x8F1BBCDC
            else:
                f = b ^ c ^ d
                k = 0xCA62C1D6
            var tmp = rotate_bits_left[5](a) + f + e + k + self.w[t]
            e = d
            d = c
            c = rotate_bits_left[30](b)
            b = a
            a = tmp
        self.h[0] += a
        self.h[1] += b
        self.h[2] += c
        self.h[3] += d
        self.h[4] += e

    def update(mut self, data: List[UInt8], n: Int):
        var i = 0
        if self.used > 0:
            while i < n and self.used < 64:
                self.block[self.used] = data[i]
                self.used += 1
                i += 1
            if self.used == 64:
                self._compress(self.block.copy(), 0)
                self.used = 0
        while n - i >= 64:
            self._compress(data, i)
            i += 64
        while i < n:
            self.block[self.used] = data[i]
            self.used += 1
            i += 1
        self.total += n

    def digest(mut self) -> List[UInt8]:
        """Pad, compress, return 20 bytes."""
        var bits = UInt64(self.total) * 8
        self.block[self.used] = 0x80
        for i in range(self.used + 1, 64):
            self.block[i] = 0
        if self.used >= 56:
            self._compress(self.block.copy(), 0)
            for i in range(64):
                self.block[i] = 0
        for k in range(8):
            self.block[56 + k] = UInt8((bits >> UInt64(8 * (7 - k))) & 0xFF)
        self._compress(self.block.copy(), 0)
        var out = List[UInt8]()
        for i in range(5):
            for k in range(4):
                out.append(UInt8((self.h[i] >> UInt32(8 * (3 - k))) & 0xFF))
        return out^


def hexdigest(d: List[UInt8]) -> String:
    var digits = "0123456789abcdef"
    var s = String()
    for b in d:
        s += String(digits[byte=Int(b) >> 4]) + String(digits[byte=Int(b) & 0xF])
    return s
