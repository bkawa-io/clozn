"""One native generation must produce exactly ONE run record.

Two writers can persist a `/api/clozn/generate` turn: the gateway's
`generation_gateway._native_log_run`, and `clozn run`'s own `_log_run_cli`. Both used to fire
unconditionally, so every CLI turn wrote two runs -- a real duplicate, shipped and unnoticed because
each writer's own tests passed in isolation.

The CLI's record is the richer of the two and is the one kept: it carries the user's raw question (so
`messages` reads as a question rather than chat-template syntax) and `final_prompt`, the exact wire
input Gate 0 requires -- neither of which the gateway can see, since that route only ever receives an
already-rendered prompt and its `_log_run` call passes no `final_prompt` at all.

So the CLI declares itself via `X-Clozn-Client-Journals` and the gateway stands down. Every OTHER
caller -- Studio, curl, third-party integrations, the live smoke battery -- sends no such header and
still gets a server-side run, which is the receipts gap that journal was added to close.
"""
from __future__ import annotations

from clozn.server import generation_gateway


class _Handler:
    """Minimal stand-in for the request handler: just the header bag and the journal seam."""

    def __init__(self, headers=None):
        self.headers = dict(headers or {})
        self.calls = []

    def _log_run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "run_fake_id"


_FRAMES = [
    {"type": "gen_started", "prompt_tokens": 3},
    {"type": "token", "piece": "hi"},
    {"type": "gen_finished", "reason": "eos"},
]


def test_gateway_stands_down_when_the_client_declares_it_journals():
    handler = _Handler({"X-Clozn-Client-Journals": "1"})

    run_id = generation_gateway._native_log_run(handler, {"prompt": "rendered"}, _FRAMES, 1.0)

    assert run_id is None
    assert handler.calls == [], (
        "the gateway journaled a turn whose client already journals it -- that is the duplicate-run "
        "bug this header exists to prevent"
    )


def test_gateway_still_journals_for_every_other_caller():
    handler = _Handler()

    run_id = generation_gateway._native_log_run(handler, {"prompt": "rendered"}, _FRAMES, 1.0)

    assert run_id == "run_fake_id"
    assert len(handler.calls) == 1, (
        "a caller that does NOT declare client-side journaling must still get a persisted run -- "
        "this is the receipts gap the native journal was added to close"
    )


def test_an_empty_or_absent_header_value_does_not_suppress_the_journal():
    """Only an affirmative declaration stands the gateway down.

    A proxy that strips a value, or a client that sets the key to "", must not silently lose its run.
    Failing open here is the safe direction: a duplicate is visible and fixable, a missing receipt is
    neither.
    """
    for value in ("", "   "):
        handler = _Handler({"X-Clozn-Client-Journals": value})
        generation_gateway._native_log_run(handler, {"prompt": "rendered"}, _FRAMES, 1.0)
        assert len(handler.calls) == 1, f"value {value!r} suppressed the journal; it must not"


def test_the_cli_sends_the_header_on_both_of_its_request_paths():
    """The header only works if `clozn run` actually sends it -- on the streaming path AND the
    non-streaming one. An earlier version of this fix would have covered only one."""
    import inspect
    from clozn.cli.commands import run as cli_run

    source = inspect.getsource(cli_run)
    assert source.count("X-Clozn-Client-Journals") == 2, (
        "expected the header on both /api/clozn/generate request sites in clozn run"
    )
