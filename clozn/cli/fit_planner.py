"""Minimal local GGUF-header parser used to derive stable model identity.

The metadata needed for identity lives at the front of a GGUF, so this module reads only enough
of a local file to parse that header. It never loads model weights or touches an accelerator.
"""
from __future__ import annotations

import os
import struct
from collections import Counter

GGUF_MAGIC = b"GGUF"

# GGUF metadata value-type enum (the tag that precedes every KV's payload).
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = range(8)
_T_STRING, _T_ARRAY = 8, 9
_T_UINT64, _T_INT64, _T_FLOAT64 = 10, 11, 12

# ggml_type enum (ggml.h) -- what a *tensor's* dtype byte means. Only the ones that actually turn
# up in released GGUFs are named; anything else prints as "type<N>" rather than guessing.
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}
# Norm/bias-style tensors llama.cpp keeps at full precision even in a quantized model -- excluded
# when picking the "dominant" quant so a Q4_K_M model doesn't get misread as "mostly F32".
_HIGH_PRECISION_TYPES = {0, 1, 30}  # F32, F16, BF16

# general.file_type metadata (llama_ftype enum) -- the AUTHORITATIVE quant label when present: it's
# written by the quantizer itself and is the only way to tell a _S/_M/_L variant apart (they can
# share the exact same dominant tensor type and differ only in a handful of upgraded tensors).
LLAMA_FTYPE_LABELS = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 1024: "GUESSED",
}


class NeedMoreBytes(ValueError):
    """Raised when the supplied bytes end before the GGUF header does."""

    def __init__(self, need_at_least: int, have: int):
        self.need_at_least = need_at_least
        self.have = have
        super().__init__(f"need more bytes: got {have}, need at least {need_at_least}")


class _Reader:
    """A bounds-checked little-endian cursor over an in-memory buffer. Any read past the end of
    the buffer raises NeedMoreBytes instead of a bare struct.error, naming exactly how many bytes
    would have been needed -- that's what lets the caller grow its HTTP Range/local read and retry."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def _take(self, fmt: str, size: int):
        end = self.pos + size
        if end > len(self.data):
            raise NeedMoreBytes(end, len(self.data))
        v = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos = end
        return v

    def bytes_(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise NeedMoreBytes(end, len(self.data))
        b = self.data[self.pos:end]
        self.pos = end
        return b

    def u8(self) -> int: return self._take("<B", 1)
    def i8(self) -> int: return self._take("<b", 1)
    def u16(self) -> int: return self._take("<H", 2)
    def i16(self) -> int: return self._take("<h", 2)
    def u32(self) -> int: return self._take("<I", 4)
    def i32(self) -> int: return self._take("<i", 4)
    def f32(self) -> float: return self._take("<f", 4)
    def bool_(self) -> bool: return self._take("<B", 1) != 0
    def u64(self) -> int: return self._take("<Q", 8)
    def i64(self) -> int: return self._take("<q", 8)
    def f64(self) -> float: return self._take("<d", 8)

    def str_(self) -> str:
        n = self.u64()
        return self.bytes_(n).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype == _T_UINT8: return self.u8()
        if vtype == _T_INT8: return self.i8()
        if vtype == _T_UINT16: return self.u16()
        if vtype == _T_INT16: return self.i16()
        if vtype == _T_UINT32: return self.u32()
        if vtype == _T_INT32: return self.i32()
        if vtype == _T_FLOAT32: return self.f32()
        if vtype == _T_BOOL: return self.bool_()
        if vtype == _T_STRING: return self.str_()
        if vtype == _T_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            return [self.value(elem_type) for _ in range(count)]
        if vtype == _T_UINT64: return self.u64()
        if vtype == _T_INT64: return self.i64()
        if vtype == _T_FLOAT64: return self.f64()
        raise ValueError(f"unknown GGUF metadata value type: {vtype}")


def _quant_label(metadata: dict, tensors: list[dict]) -> tuple[str, str]:
    """-> (label, source). Prefers general.file_type (exact, written by the quantizer); falls back
    to the dominant tensor ggml_type (can't tell _S/_M/_L apart -- flagged as approximate)."""
    ft = metadata.get("general.file_type")
    if isinstance(ft, int) and ft in LLAMA_FTYPE_LABELS:
        return LLAMA_FTYPE_LABELS[ft], "general.file_type"
    counts = Counter(t["ggml_type"] for t in tensors)
    if not counts:
        return "unknown", "no tensors in header"
    quantized = {t: c for t, c in counts.items() if t not in _HIGH_PRECISION_TYPES}
    pool = quantized or counts
    dominant = max(pool.items(), key=lambda kv: kv[1])[0]
    label = GGML_TYPE_NAMES.get(dominant, f"type{dominant}")
    return label, "dominant tensor type (approximate -- general.file_type metadata absent)"


def parse_gguf_header(data: bytes) -> dict:
    """Parse a GGUF binary header out of `data` (the first N bytes of a .gguf file/URL). Raises
    NeedMoreBytes if `data` is truncated before the header ends -- the caller should fetch more and
    retry. Returns a dict with useful identity fields (arch, context_length, n_layers,
    embedding_length, head_count, head_count_kv, quant, ...) plus the raw `metadata` dict and the
    parsed `tensors` list for anyone who wants more."""
    if len(data) < 4:
        raise NeedMoreBytes(4, len(data))
    if data[0:4] != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file: magic bytes {data[0:4]!r} != {GGUF_MAGIC!r}")

    r = _Reader(data)
    r.bytes_(4)                     # magic, already checked
    version = r.u32()
    tensor_count = r.u64()
    kv_count = r.u64()

    metadata: dict = {}
    for _ in range(kv_count):
        key = r.str_()
        vtype = r.u32()
        metadata[key] = r.value(vtype)

    tensors: list[dict] = []
    for _ in range(tensor_count):
        name = r.str_()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        ggml_type = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "shape": dims, "ggml_type": ggml_type, "offset": offset})

    arch = metadata.get("general.architecture", "") or ""

    def m(suffix, default=None):
        return metadata.get(f"{arch}.{suffix}", default)

    quant, quant_source = _quant_label(metadata, tensors)

    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": kv_count,
        "arch": arch,
        "name": metadata.get("general.name"),
        "context_length": m("context_length"),
        "n_layers": m("block_count"),
        "embedding_length": m("embedding_length"),
        "head_count": m("attention.head_count"),
        "head_count_kv": m("attention.head_count_kv") or m("attention.head_count"),
        "quant": quant,
        "quant_source": quant_source,
        "file_type": metadata.get("general.file_type"),
        "header_bytes": r.pos,          # how much of `data` the header actually used
        "metadata": metadata,
        "tensors": tensors,
    }


def gguf_header_from_path(path: str, initial_bytes: int = 1 << 20, max_bytes: int = 64 << 20) -> dict:
    """Read the first chunk of a local .gguf and parse it, growing the read if the header didn't
    fit (large tokenizer vocabularies routinely push real headers to several MB)."""
    size = os.path.getsize(path)
    n = min(initial_bytes, size) if size else initial_bytes
    data = b""
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            data = f.read(n)
            try:
                header = parse_gguf_header(data)
                break
            except NeedMoreBytes as e:
                if n >= size or n >= max_bytes:
                    raise
                n = min(max(e.need_at_least + 4096, n * 2), size, max_bytes)
    header["file_size_bytes"] = size
    header["path"] = os.path.abspath(path)
    header["bytes_read"] = len(data)
    return header
