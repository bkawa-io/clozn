"""The worker <-> supervisor wire-contract version.

`clozn serve` (the product gateway + supervisor) speaks this contract to the private C++ worker
(`clozn-server`). The version is a plain ``"MAJOR.MINOR"`` string:

  * a **MAJOR** bump is breaking -- the supervisor refuses a worker whose major it does not support,
    rather than proxy a stream it can no longer parse;
  * a **MINOR** bump is additive/back-compatible -- new capability flags, new optional fields. A newer
    worker minor talking to an older supervisor is fine (the supervisor ignores what it doesn't know);
    an older worker minor is fine too (missing optional fields read as absent).

The exact same version string is pinned three ways and a golden-fixture test fails the moment they drift:

  * here (Python: the supervisor + gateway),
  * ``engine/core/serve/server_shared.hpp`` -> ``PROTOCOL_VERSION`` (C++: the worker),
  * ``protocol/fixtures/handshake.json`` (the shared contract Studio can also read).
"""

PROTOCOL_VERSION = "1.0"

# Majors this supervisor can drive. A worker announcing a major outside this set is refused at boot.
SUPPORTED_MAJORS = frozenset({1})


def parse_major(version) -> "int | None":
    """The integer MAJOR of a ``"MAJOR.MINOR"`` string, or None if it isn't a well-formed version.
    Anything non-string, empty, or non-numeric in the major slot is None (an unusable announcement)."""
    if not isinstance(version, str):
        return None
    head = version.split(".", 1)[0].strip()
    if not head.isdigit():
        return None
    return int(head)


def check_worker_protocol(version) -> "tuple[bool, str]":
    """Decide whether the supervisor may drive a worker announcing ``version`` on /health.

    Returns ``(ok, reason)``. A refusal reason is human-actionable -- the caller surfaces it verbatim,
    and the usual cause is a stale worker binary that predates the handshake, which a rebuild fixes.
    """
    if version is None:
        return False, (
            f"worker announced no protocol_version (supervisor speaks {PROTOCOL_VERSION}); "
            "the engine binary predates the handshake -- rebuild clozn-server"
        )
    major = parse_major(version)
    if major is None:
        return False, f"worker announced an unparseable protocol_version {version!r} (supervisor speaks {PROTOCOL_VERSION})"
    if major not in SUPPORTED_MAJORS:
        supported = ", ".join(str(m) for m in sorted(SUPPORTED_MAJORS))
        return False, (
            f"worker protocol major {major} (from {version!r}) is incompatible; "
            f"this supervisor speaks {PROTOCOL_VERSION} and supports major(s) {supported}"
        )
    return True, ""
