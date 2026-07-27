"""test_narrate_cli -- model-free tests for clozn_cli.py's `clozn explain --why` (EXPLAIN_THIS_ANSWER_SPEC.md
Milestone 4, the TUI half). M4's /runs/<id>/narrate (research/narrate.py, wired into clozn_server.py) already
assembles the accountable-self narration + confabulation-diff server-side; this flag is DISPLAY ONLY -- it
generates nothing itself, it just POSTs to the already-live endpoint and renders whatever comes back. This
file tests format_narrate(), the pure "narrate() object -> terminal text" function clozn_cli.py factors out
specifically so that's possible with a canned dict: no running Studio, no model, no GPU.

Layout (mirrors test_explain_cli.py exactly):
  * canned-dict tests drive format_narrate() directly against hand-built /narrate-shaped fixtures -- exactly
    narrate.narrate()'s documented 4-key contract -- and assert the honesty invariants (the narration prose
    renders, every flag renders as a visible warning, the note/matcher always renders, an empty/thin result
    reads as honest rather than as an error, and nothing beyond the four keys is ever rendered even if a
    stray 5th field is present -- the trap guard, carried all the way to the terminal).
  * a wire-format check drives the REAL clozn_server /narrate endpoint in-process (the same no-socket
    object.__new__(H) trick test_explain_cli.py and test_narrate_server.py already use), with the matcher
    forced to the lexical fallback (monkeypatch semantic_matcher.available -> False) so this stays model-free
    -- the real cross-encoder judge is test_semantic_matcher_gated.py's job, not a unit test's.
  * _fetch_narrate()'s honest failure path is checked against a guaranteed-closed local port -- no network
    flakiness, no live Studio required (mirrors _fetch_explain's own such test).
  * the `--why` flag itself: opt-in (defaults False), and cmd_explain only calls _fetch_narrate when it is
    set -- proven by monkeypatching _fetch_narrate to raise if it's ever called without --why.

What this file can NOT exercise: a live HTTP round trip against a *running* `clozn studio` process, or the
real semantic (NLI) matcher -- both are, respectively, a manual follow-up and test_semantic_matcher_gated.py's
job.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))          # tests/
REPO = os.path.dirname(HERE)                                 # repo root (clozn/cli.py lives here)
sys.path.insert(0, REPO)

import clozn.cli.main as clozn_cli                                            # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.cli.commands.explain as explain_cmd                                     # noqa: E402
from clozn.server import app as cs                                    # noqa: E402


import clozn.runs.store as runlog                                                # noqa: E402
import clozn.receipts.semantic_matcher as semantic_matcher                                      # noqa: E402


# ---------------------------------------------------------------------------------- canned-dict fixtures
# Exactly narrate.narrate()'s documented return shape: {"constrained_narration": {"narration","receipt_ids"},
# "flags": [...], "unsupported_claims": [...], "note": ...}. No other key is ever part of the real contract.

HAPPY_PATH = {
    "constrained_narration": {
        "narration": "I kept the reply brief because you asked for that. [mem_1]",
        "receipt_ids": ["mem_1"],
    },
    "flags": [
        'WARNING: credits "I mentioned dragons because I love mythology."; no receipt for that.',
    ],
    "unsupported_claims": [
        {"claim": "I mentioned dragons because I love mythology.", "supported": False,
         "flag": 'WARNING: credits "I mentioned dragons because I love mythology."; no receipt for that.',
         "matched_ids": [], "matched_terms": []},
    ],
    "note": (
        "constrained_narration is the answer surface; the model's unconstrained self-report is never "
        "included here (THE TRAP guard) -- only its diff against the receipts, as flags, is. Matcher used: "
        "lexical_default. If that is lexical_default: it is a WEAK keyword-overlap proxy ..."
    ),
}

EMPTY_PATH = {
    "constrained_narration": {"narration": "", "receipt_ids": []},
    "flags": [],
    "unsupported_claims": [],
    "note": "constrained_narration is the answer surface ... Matcher used: lexical_default. ...",
}


def test_happy_path_renders_narration_flag_and_note():
    out = clozn_cli.format_narrate(HAPPY_PATH)
    assert "I kept the reply brief because you asked for that." in out
    assert "dragons" in out and "mythology" in out          # the flagged claim, verbatim
    assert "WARNING" in out                                  # the flag is rendered as a visible warning
    assert "⚠" in out                                        # a distinct visual marker, not folded into prose
    assert "lexical_default" in out                          # the note names the matcher that ran


def test_flag_appears_only_as_a_warning_never_as_unflagged_prose():
    """THE TRAP GUARD, carried to the terminal: the confabulated claim text may appear (inside a flag --
    that's the point, showing exactly what was caught) but the render must never present it as part of the
    trusted narration itself. Concretely: the flagged claim text must not appear on the same line as the
    narration line, only under the "caught in the diff" section."""
    out = clozn_cli.format_narrate(HAPPY_PATH)
    lines = out.splitlines()
    narration_lines = [ln for ln in lines if "I kept the reply brief" in ln]
    assert narration_lines and "dragons" not in narration_lines[0]


def test_empty_narration_and_no_flags_is_an_honest_first_class_result_not_an_error():
    out = clozn_cli.format_narrate(EMPTY_PATH)
    assert "no receipt-backed narration" in out
    assert "no unsupported claims flagged" in out
    assert "error" not in out.lower()
    assert "traceback" not in out.lower()


def test_empty_dict_degrades_fully_and_honestly():
    out = clozn_cli.format_narrate({})
    assert "no receipt-backed narration" in out
    assert "no unsupported claims flagged" in out
    assert "error" not in out.lower()


def test_note_always_renders_so_the_honesty_level_is_never_hidden():
    out = clozn_cli.format_narrate(HAPPY_PATH)
    assert HAPPY_PATH["note"][:40] in out   # the note's own text, verbatim


def test_multiple_flags_each_render_as_their_own_warning_line():
    obj = dict(HAPPY_PATH)
    obj["flags"] = [
        'WARNING: credits "I am a 40-language consultant."; no receipt for that.',
        'WARNING: credits "I remember your dog Rex."; no receipt for that.',
    ]
    out = clozn_cli.format_narrate(obj)
    assert "40-language consultant" in out
    assert "your dog Rex" in out
    assert out.count("⚠") == 2


def test_never_renders_a_field_beyond_the_documented_four():
    """Defense in depth for the trap guard: even if some upstream bug attached a 5th field (e.g. the raw
    unconstrained self-report) onto the object handed to format_narrate, this renderer must never surface
    it -- it only ever reads constrained_narration/flags/note (unsupported_claims is read only insofar as
    flags already carries its rendering)."""
    leaky = dict(HAPPY_PATH)
    leaky["unconstrained_why"] = "THIS MUST NEVER APPEAR: secretly I am a pineapple pizza connoisseur"
    leaky["raw_why"] = "THIS MUST NEVER APPEAR EITHER"
    out = clozn_cli.format_narrate(leaky)
    assert "pineapple" not in out
    assert "NEVER APPEAR EITHER" not in out


@pytest.mark.parametrize("garbage", [
    None, "not a dict", 42, [], ["also", "not", "a", "dict"],
    {"constrained_narration": "nope", "flags": "nope", "note": 3},
    {"constrained_narration": {"narration": 12345}, "flags": [None, 3, {}], "unsupported_claims": "nope"},
])
def test_never_raises_on_malformed_input(garbage):
    out = clozn_cli.format_narrate(garbage)        # must not raise
    assert isinstance(out, str) and out


# --------------------------------------------------------------------------------- wire-format compatibility
# Drives the REAL clozn_server /narrate handler in-process (mirrors test_narrate_server.py's own no-socket
# object.__new__(H) trick and its `iso` fixture) so format_narrate() is proven against a genuine server
# response, not just a fixture hand-built above. The matcher is forced to the lexical fallback so no
# ~440MB cross-encoder checkpoint loads in a unit test.

class NarrateFakeSub:
    """Two-armed fake substrate, routing on the prompt text exactly like narrate.py's two calls do (mirrors
    test_narrate_server.py's own NarrateFakeSub; duplicated locally so this file stays self-contained)."""
    name = "qwen"

    def chat(self, messages, max_new=256, sample=True):
        joined = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        if "Why did you answer that way" in joined:
            return "I mentioned dragons because I love mythology."       # the confabulation sample
        return "I kept the reply brief because you asked for that."      # the constrained narration


def _post(path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the stores; SUB is the fake two-armed substrate; force the LEXICAL fallback so no checkpoint
    loads (the NLI path is test_semantic_matcher_gated.py's job). Mirrors test_narrate_server.py's `iso`."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(semantic_matcher, "available", lambda: False)
    monkeypatch.setattr(cs, "SUB", NarrateFakeSub())
    return tmp_path


def _seed_run():
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                         messages=[{"role": "user", "content": "Tell me about dragons, briefly."}],
                         response="Dragons are mythical reptilian creatures.",
                         behavior={"active_dials": {"warm": 0.2}})


def test_format_narrate_renders_a_genuine_server_response(iso):
    rid = _seed_run()
    out_json = _post(f"/runs/{rid}/narrate")
    assert set(out_json.keys()) == {"constrained_narration", "flags", "unsupported_claims", "note"}
    out = clozn_cli.format_narrate(out_json)
    assert isinstance(out, str) and out
    assert "I kept the reply brief" in out
    # THE TRAP GUARD, proven end to end: the confabulation ("dragons"/"mythology") is caught and flagged
    # (no receipt: only a "warm" dial is on record, no lexical overlap with the dragons/mythology claim) --
    # so it shows up ONLY inside a warning line, never as trusted, unflagged narration prose.
    assert any("dragons" in f.lower() for f in out_json["flags"])
    assert "dragons" in out and "WARNING" in out
    assert "lexical_default" in out_json["note"]
    assert "lexical_default" in out


def test_format_narrate_renders_a_genuine_404_shape(iso):
    """/narrate's 404 is {"error": "run not found"} -- format_narrate must not choke on a payload with none
    of its normal four keys, and must degrade to the same honest empty-result rendering."""
    out_json = _post("/runs/run_does_not_exist/narrate")
    assert out_json == {"error": "run not found"}
    out = clozn_cli.format_narrate(out_json)
    assert isinstance(out, str) and out
    assert "no receipt-backed narration" in out


def test_format_narrate_renders_a_genuine_503_shape(iso, monkeypatch):
    """/narrate's 503 (no substrate) is {"error": "narration needs the qwen substrate"} -- same honest
    degradation, never a crash."""
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out_json = _post(f"/runs/{rid}/narrate")
    assert out_json == {"error": "narration requires a ready product model worker"}
    out = clozn_cli.format_narrate(out_json)
    assert isinstance(out, str) and out
    assert "no receipt-backed narration" in out


# ------------------------------------------------------------------------------------------- _fetch_narrate

def test_fetch_narrate_is_an_honest_cloznerror_when_studio_is_down():
    """No live Studio required (and per the task at hand, none is running): a guaranteed-closed local port
    must fail as one clean, actionable CloznError -- never a raw urllib traceback."""
    port = clozn_cli._free_port()   # bound then released -- nothing is listening on it
    with pytest.raises(clozn_cli.CloznError):
        clozn_cli._fetch_narrate(port, "run_x")


# ------------------------------------------------------------------------------------------- --why (opt-in)

def test_why_flag_defaults_to_false():
    args = clozn_cli.build_parser().parse_args(["explain", "run_x"])
    assert args.why is False


def test_why_flag_opts_in():
    args = clozn_cli.build_parser().parse_args(["explain", "run_x", "--why"])
    assert args.why is True
    assert args.fn is clozn_cli.cmd_explain


def test_cmd_explain_without_why_never_calls_fetch_narrate(monkeypatch):
    """The generation is opt-in: without --why, _fetch_narrate must never even be called.

    cmd_explain/_fetch_explain/_fetch_narrate all live in clozn.cli.commands.explain; cmd_explain's call
    to _fetch_narrate(...) is a bare-name lookup resolved against THAT module's globals, so the patch has
    to land there (not on clozn.cli.main's re-exported copy of the name) to be observed."""
    monkeypatch.setattr(explain_cmd, "_fetch_explain", lambda port, rid: {})

    def _boom(port, rid):
        raise AssertionError("_fetch_narrate must not be called without --why")
    monkeypatch.setattr(explain_cmd, "_fetch_narrate", _boom)

    args = clozn_cli.build_parser().parse_args(["explain", "run_x"])
    clozn_cli.cmd_explain(args)   # must not raise


def test_cmd_explain_with_why_calls_narrate_and_prints_it(monkeypatch, capsys):
    monkeypatch.setattr(explain_cmd, "_fetch_explain", lambda port, rid: {})
    monkeypatch.setattr(explain_cmd, "_fetch_narrate", lambda port, rid: HAPPY_PATH)

    args = clozn_cli.build_parser().parse_args(["explain", "run_x", "--why"])
    clozn_cli.cmd_explain(args)

    out = capsys.readouterr().out
    assert "I kept the reply brief because you asked for that." in out
    assert "lexical_default" in out
