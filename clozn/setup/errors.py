"""Exceptions for clozn/setup/*, kept independent of clozn.cli.main.CloznError.

clozn/setup is pure client-side logic that clozn/cli/commands/setup_engine.py wraps for the CLI; keeping
its own exception hierarchy means clozn/setup never imports the CLI layer (avoiding a needless coupling
in the direction that matters -- setup logic should be testable and reusable with no argparse in sight).
The CLI command module catches SetupError and re-raises clozn.cli.main.CloznError(str(error)) so the
user-facing behavior (one clean line, no traceback) is unchanged.
"""
from __future__ import annotations


class SetupError(Exception):
    """A clean, user-facing failure in manifest fetch/selection/install. No dependency information --
    this exception itself carries the whole message; callers should not re-wrap its text."""


class ManifestError(SetupError):
    """The manifest document is missing, malformed, fails schema validation, or names a protocol major
    this clozn cannot speak."""


class SelectionError(SetupError):
    """No artifact in an otherwise-valid manifest matches the requested platform/backend/version."""


class TransportError(SetupError):
    """A URL could not be fetched: disallowed scheme, network failure, timeout, or non-2xx response."""


class VerificationError(SetupError):
    """A downloaded artifact's sha256 (or declared size) did not match the manifest."""


class ArchiveError(SetupError):
    """An archive could not be safely extracted: path traversal, an escaping symlink, an unsupported
    format, or a member count/size that looks like a decompression bomb."""


class LockError(SetupError):
    """Another `clozn setup` invocation is already in flight (see clozn/setup/lock.py)."""


class RegistryError(SetupError):
    """~/.clozn/engines/registry.json is corrupt in a way that cannot be self-healed, or a requested
    install key does not exist."""
