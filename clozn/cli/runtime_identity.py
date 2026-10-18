"""Canonical identity for the one private GGUF worker."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json


_HEX = frozenset("0123456789abcdef")


class WorkerRegistryConfigError(ValueError):
    """The configured worker identity is incomplete or contradictory."""


def _require_string(value, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerRegistryConfigError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value, field_name: str) -> str:
    value = _require_string(value, field_name)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise WorkerRegistryConfigError(
            f"{field_name} must be a lowercase 64-character SHA-256"
        )
    return value


@dataclass(frozen=True)
class RuntimeKey:
    """Behavior-bearing identity of the private worker, canonically hashed."""

    gguf_artifact_sha256: str
    context_size: int
    backend: str
    template_fingerprint: str
    engine_build: str
    key_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.gguf_artifact_sha256, "gguf_artifact_sha256")
        if (isinstance(self.context_size, bool)
                or not isinstance(self.context_size, int)
                or self.context_size < 1):
            raise WorkerRegistryConfigError("context_size must be a positive integer")
        _require_string(self.backend, "backend")
        fingerprint = _require_string(self.template_fingerprint, "template_fingerprint")
        if (not 16 <= len(fingerprint) <= 64
                or any(character not in _HEX for character in fingerprint)):
            raise WorkerRegistryConfigError(
                "template_fingerprint must be 16-64 lowercase hexadecimal characters"
            )
        _require_string(self.engine_build, "engine_build")
        encoded = json.dumps(
            self.facets(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        object.__setattr__(self, "key_sha256", hashlib.sha256(encoded).hexdigest())

    def facets(self) -> dict:
        return {
            "gguf_artifact_sha256": self.gguf_artifact_sha256,
            "context_size": self.context_size,
            "backend": self.backend,
            "template_fingerprint": self.template_fingerprint,
            "engine_build": self.engine_build,
        }

    def as_dict(self) -> dict:
        return {"key_sha256": self.key_sha256, **self.facets()}


__all__ = [
    "RuntimeKey", "WorkerRegistryConfigError",
]
