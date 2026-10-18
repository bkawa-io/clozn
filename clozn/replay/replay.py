"""Persist one verified Minimal Context intervention as an immutable child run."""
from __future__ import annotations

import time

import clozn.runs.store as runlog


_CHANGE_KEYS = frozenset({"experiment", "greedy"})


def replay(
    run: dict,
    changes: dict,
    sub,
    *,
    max_new: int | None = None,
    messages_override: list[dict] | None = None,
    sampling_override: bool | dict | None = None,
) -> dict | None:
    """Generate and record one child from an already-resolved source deletion.

    The caller must provide the exact intervened message list and recorded decoding contract. Branch,
    section, memory, steering, and free re-roll controls are deliberately unsupported in v1.
    """
    try:
        if not isinstance(run, dict) or not isinstance(run.get("id"), str) or not run["id"]:
            return None
        if not isinstance(changes, dict) or set(changes).difference(_CHANGE_KEYS):
            return None
        experiment = changes.get("experiment")
        if not isinstance(experiment, dict) or experiment.get("intervention", {}).get("kind") != "delete_source":
            return None
        if not isinstance(messages_override, list) or not messages_override:
            return None
        if sampling_override is None or not isinstance(sampling_override, (bool, dict)):
            return None

        from clozn.runs.think_tags import sanitize_messages

        messages = sanitize_messages(messages_override)
        if not messages or any(not isinstance(message, dict) for message in messages):
            return None
        chat = getattr(sub, "chat", None)
        if not callable(chat):
            return None

        budget = int(max_new) if isinstance(max_new, int) and not isinstance(max_new, bool) and max_new > 0 else 256
        trace_steps: list = []
        prompt_evidence: dict = {}
        call_kwargs = {
            "max_new": budget,
            "sample": sampling_override,
            "trace_out": trace_steps,
            "mem_out": prompt_evidence,
        }
        started = time.time()
        while True:
            try:
                reply = chat(messages, **call_kwargs)
                break
            except TypeError as exc:
                message = str(exc)
                unsupported = next(
                    (key for key in ("trace_out", "mem_out") if key in call_kwargs and key in message),
                    None,
                )
                if unsupported is None:
                    raise
                del call_kwargs[unsupported]

        finish_reason = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
        try:
            metadata = (sub.run_meta() or None) if hasattr(sub, "run_meta") else None
        except Exception:
            metadata = None
        metadata = {**(metadata or {}), "max_tokens": budget}
        try:
            identity = (sub.identity_meta() or None) if hasattr(sub, "identity_meta") else None
        except Exception:
            identity = None

        run_id = runlog.record(
            source="replay",
            client=run.get("client") or "studio",
            model=run.get("model"),
            substrate=run.get("substrate"),
            messages=messages,
            response=reply if isinstance(reply, str) else str(reply),
            trace=trace_steps,
            finish_reason=finish_reason,
            meta=metadata,
            parent_run_id=run["id"],
            changes_applied=changes,
            started=started,
            assembled_messages=prompt_evidence.get("assembled_messages"),
            final_prompt=prompt_evidence.get("final_prompt"),
            identity=identity,
            session_key=run.get("session_key"),
            client_key=run.get("client_key"),
            client_key_source=run.get("client_key_source"),
            project_key=run.get("project_key"),
        )
        return runlog.get_run(run_id) if run_id is not None else None
    except Exception:
        return None


__all__ = ["replay"]
