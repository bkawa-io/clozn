"""Engine identity facet (roadmap feature 01): which of the 4 discovery-precedence tiers produced the
running engine, its backend, its managed-install artifact sha256/version, and the protocol version its
own /health already reported.

Everything here is read out of `context`, never recomputed -- discovery_source/backend/artifact_sha256/
engine_version arrive via runtime_identity()'s `extra_context` kwarg (populated by
clozn.server.substrates._engine_discovery_context() from CLOZN_ENGINE_* env vars
clozn.cli.runtime_process.spawn_runtime() sets on the gateway subprocess), and protocol_version comes
straight off the SAME engine_health dict every other facet in clozn.runs.identity already reads. Any
field this process could not establish is simply absent from the returned dict -- never guessed, never
null-padded (roadmap rule: "Omit, never null-pad").
"""
from __future__ import annotations

NAME = "engine_artifact"

_CONTEXT_FIELDS = ("discovery_source", "backend", "artifact_sha256", "engine_version")


def identity(context):
    context = context if isinstance(context, dict) else {}
    out = {}
    for field in _CONTEXT_FIELDS:
        value = context.get(field)
        if value:
            out[field] = value
    health = context.get("engine_health")
    if isinstance(health, dict) and health.get("protocol_version") is not None:
        out["protocol_version"] = health["protocol_version"]
    return out
