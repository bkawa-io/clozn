"""Small, stdlib-only identity helpers for local GGUF artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Mapping

from clozn.cli.fit_planner import gguf_header_from_path


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    """Return the lowercase SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_contract(metadata: Mapping[str, object]) -> dict:
    return {
        key: metadata[key]
        for key in sorted(metadata)
        if key.startswith("tokenizer.")
    }


def gguf_identity(path: str | os.PathLike[str], *, include_file_hash: bool = True) -> dict:
    """Read the model, tokenizer, template, quantization, and file identity of a GGUF."""
    resolved = os.path.abspath(os.fspath(path))
    header = gguf_header_from_path(resolved)
    metadata = header["metadata"]
    tokens = metadata.get("tokenizer.ggml.tokens")
    vocab_size = (
        len(tokens)
        if isinstance(tokens, list)
        else metadata.get("tokenizer.ggml.vocab_size")
    )
    identity = {
        "name": header.get("name"),
        "architecture": header.get("arch"),
        "hidden_size": header.get("embedding_length"),
        "layer_count": header.get("n_layers"),
        "vocab_size": vocab_size,
        "head_count": header.get("head_count"),
        "head_count_kv": header.get("head_count_kv"),
        "tokenizer_sha256": _json_digest(_tokenizer_contract(metadata)),
        "chat_template_sha256": _json_digest(metadata.get("tokenizer.chat_template")),
        "quantization": header.get("quant"),
        "file_size": int(header["file_size_bytes"]),
        "filename": os.path.basename(resolved),
    }
    if include_file_hash:
        # Use the persistent path/size/mtime cache used by engine startup. The
        # lazy import avoids the identity module's sha256_file import cycle.
        from clozn.runs.identity import model_sha256

        identity["sha256"] = model_sha256(resolved) or sha256_file(resolved)
    return identity
