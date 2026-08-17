"""Canonical reconstructed normal-generation seams.

These helpers retain the old raw-prompt generation behavior without importing
the legacy child-creating fork module.  They always derive steering from the
recorded run and never read or mutate live steering controls.
"""
from __future__ import annotations

from collections.abc import Mapping


def _inject_block(messages, block):
    if not block:
        return list(messages or [])
    result = [dict(message) for message in (messages or [])]
    for message in result:
        if message.get("role") == "system":
            message["content"] = (str(message.get("content") or "") + "\n\n" + block).strip()
            return result
    return [{"role": "system", "content": block}] + result


def prompt_base(run: Mapping, substrate):
    """Return the exact recorded prompt or its canonical template reconstruction."""
    final_prompt = run.get("final_prompt")
    if isinstance(final_prompt, str) and final_prompt:
        return final_prompt, "final_prompt"
    from clozn.receipts import rederive
    conditions = rederive.with_arm_conditions(dict(run))
    template = getattr(getattr(substrate, "engine", None), "apply_template", None)
    if not callable(template):
        return None, None
    try:
        return str(template(_inject_block(conditions["messages"], conditions["block"]))), "apply_template"
    except Exception:
        return None, None


def detect_retokenization(substrate, run: Mapping, expected_pieces: list[str]) -> bool | None:
    """Return True for a verified token-boundary shift, None when unverifiable."""
    score = getattr(substrate, "score_tokens", None)
    if not callable(score):
        return None
    expected = [str(piece) for piece in expected_pieces]
    try:
        from clozn.receipts import rederive
        conditions = rederive.with_arm_conditions(dict(run))
        tokens = score(conditions["messages"], None, continuation="".join(expected),
                       block=conditions["block"])
    except Exception:
        return None
    if not isinstance(tokens, list) or not tokens:
        return None
    actual = [str(token.get("piece", "")) for token in tokens if isinstance(token, Mapping)]
    return actual != expected


def recorded_steer_kwargs(substrate, run: Mapping) -> dict:
    """Build raw-engine steering kwargs from the recorded, not live, dial state."""
    try:
        strengths = dict(((run.get("behavior") or {}).get("active_dials")) or {})
    except Exception:
        strengths = {}
    steer = getattr(substrate, "steer", None)
    if steer is None or not strengths or not any(strengths.values()):
        return {}
    try:
        vector = steer.steer_vector(strengths)
    except Exception:
        return {}
    if not vector:
        return {}
    return {"steer_vec": vector, "steer": {"coef": 1.0, "layer": getattr(steer, "layer", 0)}}


def complete_greedy(engine, prompt: str, max_new: int, extra_kwargs: Mapping):
    reply = engine.complete(prompt, max_tokens=int(max_new), temperature=0.0,
                            rep_penalty=1.0, seed=0, **dict(extra_kwargs))
    choices = reply.get("choices") if isinstance(reply, Mapping) else None
    if not (isinstance(choices, list) and choices and isinstance(choices[0], Mapping)):
        return None, None
    return str(choices[0].get("text", "")), choices[0].get("finish_reason")


def complete_traced(engine, prompt: str, max_new: int, extra_kwargs: Mapping):
    try:
        from clozn.server.substrates import _engine_complete_traced
        reply, steps, finish, _divergence = _engine_complete_traced(
            engine, prompt, int(max_new), dict(extra_kwargs), sample=None)
    except Exception:
        return None
    if not isinstance(reply, str):
        return None
    return reply, steps, finish


__all__ = [
    "complete_greedy", "complete_traced", "detect_retokenization",
    "prompt_base", "recorded_steer_kwargs",
]
