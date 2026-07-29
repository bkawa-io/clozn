"""test_section_drill -- model-free tests for POST /runs/<id>/section-drill (the recursive drill-down that
turns "which section" into "which sentence" for ONE high-influence section).

No model, no GPU, no engine launch, and no live gateway touched anywhere in this file -- every substrate
here is a small deterministic fake keyed on which marker strings survive in the (spliced) messages a real
EngineSubstrate.score_tokens would vary on, mirroring tests/test_section_influence.py's own fixture style
(same `_dispatch`/`_post` no-socket handler-driving trick, the same `iso` isolated-store fixture, the same
"fake keyed on a marker substring" scoring convention).
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.runs.store as runlog                 # noqa: E402
from clozn.runs import sections as clozn_sections  # noqa: E402
import clozn.settings as clozn_settings            # noqa: E402
from clozn.server import app as cs                 # noqa: E402


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every flat-file store this file's runs touch (mirrors test_section_influence.py's own
    `iso`). SUB starts at None -- the route's 503 path is the default; tests that want real scoring set a
    fake substrate explicitly."""
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


def _dispatch(method, path, body_obj=None):
    """Drives the REAL do_POST handler with no socket -- the object.__new__(H) trick used throughout this
    repo's *_server tests -- so this proves the actual route registration in clozn.server.app, not just
    the route module's functions in isolation."""
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


# ============================================================================================================
# ============================================= fixtures/fakes ================================================
# ============================================================================================================

DRILL_TEXT = ("The kayak trip starts at dawn on the wide river bend. "
             "The guide checks every life vest twice before departure. "
             "Lunch is packed cold to save time out on the water.")


def _seed_drill_run(section_name="rag_context", source="auto", extra_sections=None):
    messages = [
        {"role": "system", "content": "base system prompt"},
        {"role": "user", "content": DRILL_TEXT},
        {"role": "user", "content": "What time should I arrive?"},
    ]
    sections = [
        {"id": f"sec_{section_name}", "name": section_name, "source": source,
         "parts": [{"message_index": 1, "start": 0, "end": len(DRILL_TEXT)}],
         "char_count": len(DRILL_TEXT), "preview": DRILL_TEXT[:20]},
    ]
    if extra_sections:
        sections.extend(extra_sections)
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="engine",
                         messages=messages, response="dawn", behavior={"active_dials": {}},
                         trace={"token_ids": [1]}, sections=sections)


class DrillFakeSub:
    """.score_tokens()'s logprob depends on which of three sentence-level markers -- one per DRILL_TEXT
    sentence -- are still present in the messages actually scored. "life vest" is the deliberately BIG
    effect (removing it hurts the most), "dawn" a medium one, "packed cold" the smallest -- lets the
    happy-path test prove ranking/shares are computed from REAL per-sentence deltas, not just "some
    numbers came back"."""
    name = "engine"

    def __init__(self):
        self.calls: list = []

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls.append({"messages": messages})
        text = " ".join(str((m or {}).get("content", "")) for m in (messages or []) if isinstance(m, dict))
        has_dawn = "dawn" in text
        has_vest = "life vest" in text
        has_lunch = "packed cold" in text
        if has_dawn and has_vest and has_lunch:
            lp = -0.1                              # baseline: everything present
        elif has_vest and has_lunch and not has_dawn:
            lp = -1.0                               # "dawn" sentence removed -- medium effect
        elif has_dawn and has_lunch and not has_vest:
            lp = -3.0                               # "life vest" sentence removed -- big effect
        elif has_dawn and has_vest and not has_lunch:
            lp = -0.5                               # "packed cold" sentence removed -- small effect
        else:
            lp = -5.0
        return [{"id": 1, "piece": "kayak", "logprob": lp}]


class RecorderFakeSub:
    """Records every call's messages verbatim; a CONSTANT logprob regardless of content -- used only for
    tests that check WHAT was spliced, never the resulting shares/ranking (those are DrillFakeSub's job)."""
    name = "engine"

    def __init__(self):
        self.calls: list = []

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls.append(messages)
        return [{"id": 1, "piece": "x", "logprob": -0.2}]


# ============================================================================================================
# ==================================== error / degrade paths ==================================================
# ============================================================================================================

def test_route_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/section-drill", {"section": "rag_context"})
    assert out == {"error": "run not found"}


def test_route_missing_section_field_is_a_clean_400(iso):
    rid = _seed_drill_run()
    out = _post(f"/runs/{rid}/section-drill", {})
    assert out["error"]
    assert "section" in out["error"]


def test_route_no_manifest_is_a_clean_400(iso):
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    out = _post(f"/runs/{rid}/section-drill", {"section": "rag_context"})
    assert "no section named" in out["error"]


def test_route_unknown_section_is_a_clean_400(iso):
    rid = _seed_drill_run()
    out = _post(f"/runs/{rid}/section-drill", {"section": "not_a_real_section"})
    assert "no section named 'not_a_real_section'" in out["error"]


def test_route_memory_card_section_is_declined_with_honest_400(iso):
    rid = _seed_drill_run(section_name="card_1", source="memory_card")
    out = _post(f"/runs/{rid}/section-drill", {"section": "card_1"})
    assert "memory-card" in out["error"]


def test_route_no_parts_is_a_clean_400(iso):
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                        sections=[{"id": "sec_x", "name": "x", "source": "auto", "parts": [],
                                  "char_count": 0, "preview": ""}])
    out = _post(f"/runs/{rid}/section-drill", {"section": "x"})
    assert "no usable parts" in out["error"]


def test_route_needs_the_substrate_503(iso):
    rid = _seed_drill_run()                          # iso's default SUB is None
    out = _post(f"/runs/{rid}/section-drill", {"section": "rag_context"})
    assert out == {"error": "section-drill requires worker token scoring"}


def test_route_multipart_section_is_declined_with_an_honest_note_not_wrong_offsets(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", DrillFakeSub())
    messages = [
        {"role": "system", "content": "first part text here"},
        {"role": "assistant", "content": "second part text here"},
        {"role": "user", "content": "final question"},
    ]
    sections = [{"id": "sec_multi", "name": "multi", "source": "auto",
                "parts": [{"message_index": 0, "start": 0, "end": len("first part text here")},
                         {"message_index": 1, "start": 0, "end": len("second part text here")}],
                "char_count": 40, "preview": "first part text heresecond part text he"}]
    rid = runlog.record(source="cli", messages=messages, response="ok",
                        behavior={"active_dials": {}}, trace={"token_ids": [1]}, sections=sections)
    out = _post(f"/runs/{rid}/section-drill", {"section": "multi"})
    assert out["run_id"] == rid
    assert out["parent_section"] == "multi"
    assert out["sub_sections"] == []
    assert "more than one part" in out["note"]


def test_route_unsplittable_section_returns_a_single_honest_subsection(iso, monkeypatch):
    """A section too short for drill_split to find any sentence/newline boundary in -> 200 with exactly
    ONE sub-section (the whole thing, share == 1.0) and a note saying so explicitly, per the brief's
    "nothing finer to split" honesty requirement."""
    class ShortFakeSub:
        name = "engine"

        def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                         steer_strengths=None, steer_vec=None, topk=0):
            text = " ".join(str((m or {}).get("content", "")) for m in (messages or []) if isinstance(m, dict))
            lp = -0.1 if "nine" in text else -2.0
            return [{"id": 1, "piece": "p", "logprob": lp}]

    monkeypatch.setattr(cs, "SUB", ShortFakeSub())
    short_text = "Arrive by nine sharp."
    assert clozn_sections.drill_split(short_text) == [(0, len(short_text))]
    messages = [
        {"role": "system", "content": short_text},
        {"role": "user", "content": "when should I show up?"},
    ]
    sections = [{"id": "sec_short", "name": "short_sec", "source": "auto",
                "parts": [{"message_index": 0, "start": 0, "end": len(short_text)}],
                "char_count": len(short_text), "preview": short_text}]
    rid = runlog.record(source="cli", messages=messages, response="nine",
                        behavior={"active_dials": {}}, trace={"token_ids": [1]}, sections=sections)
    out = _post(f"/runs/{rid}/section-drill", {"section": "short_sec"})
    assert out["parent_section"] == "short_sec"
    assert len(out["sub_sections"]) == 1
    assert out["sub_sections"][0]["name"] == "short_sec.1"
    assert out["sub_sections"][0]["influence_share"] == 1.0
    assert "nothing finer to split" in out["note"]


# ============================================================================================================
# ============================================= happy path =====================================================
# ============================================================================================================

def test_route_happy_path_multi_sentence_ranks_by_measured_effect_and_shares_sum_to_one(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", DrillFakeSub())
    rid = _seed_drill_run()
    out = _post(f"/runs/{rid}/section-drill", {"section": "rag_context"})

    assert out["run_id"] == rid
    assert out["method"] == "teacher_forced"
    assert "NOT causal proof" in out["note"]
    assert out["parent_section"] == "rag_context"
    assert out["baseline_logprob"] == round(-0.1, 6)

    subs = out["sub_sections"]
    assert len(subs) == 3
    names = [s["name"] for s in subs]
    assert names == ["rag_context.2", "rag_context.1", "rag_context.3"]   # "life vest" (biggest) ranked first
    shares = [s["influence_share"] for s in subs]
    assert shares[0] > shares[1] > shares[2]
    assert abs(sum(shares) - 1.0) < 1e-6
    for s in subs:
        assert set(s.keys()) == {"name", "preview", "influence_share", "log_prob_delta",
                                 "per_token_delta", "summary"}
        assert s["log_prob_delta"] < 0                # removing any one sentence makes the fit worse here
    # previews trace back to the actual sentence text, in order
    assert subs[[n for n in names].index("rag_context.1")]["preview"].startswith("The kayak trip")
    assert subs[[n for n in names].index("rag_context.2")]["preview"].startswith("The guide checks")
    assert subs[[n for n in names].index("rag_context.3")]["preview"].startswith("Lunch is packed")


def test_route_can_be_dispatched_by_section_id_too(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", DrillFakeSub())
    rid = _seed_drill_run()
    out = _post(f"/runs/{rid}/section-drill", {"section": "sec_rag_context"})
    assert out["parent_section"] == "rag_context"
    assert len(out["sub_sections"]) == 3


# ============================================================================================================
# ============================================= offset remapping ===============================================
# ============================================================================================================

def test_offset_remapping_ablated_prompt_has_exactly_the_sub_span_removed(iso, monkeypatch):
    """THE offset-remap correctness test: the section's part starts at a NONZERO offset within its message
    (there is real text BEFORE it in the same message), so a bug that forgot to add `part["start"]` before
    slicing would splice out the wrong characters entirely -- this catches that class of bug, not just
    "some delta came back"."""
    monkeypatch.setattr(cs, "SUB", RecorderFakeSub())

    prefix = "Irrelevant lead-in text that is not part of the section at all. "
    sent_a = "Alpha sentence about kayaking trips at dawn on the river today."
    sent_b = "Beta sentence about checking every life vest before departure now."
    section_text = sent_a + " " + sent_b
    msg_content = prefix + section_text
    part_start = len(prefix)
    part_end = part_start + len(section_text)

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": msg_content},
        {"role": "user", "content": "final question"},
    ]
    sections = [{"id": "sec_off", "name": "off_test", "source": "auto",
                "parts": [{"message_index": 1, "start": part_start, "end": part_end}],
                "char_count": len(section_text), "preview": section_text[:20]}]
    rid = runlog.record(source="cli", messages=messages, response="ok",
                        behavior={"active_dials": {}}, trace={"token_ids": [1]}, sections=sections)

    # ground truth: the SAME drill_split function the route uses, called directly on the section's own text
    spans = clozn_sections.drill_split(section_text)
    assert len(spans) == 2

    sub = cs.SUB
    out = _post(f"/runs/{rid}/section-drill", {"section": "off_test"})
    assert out["parent_section"] == "off_test"
    assert len(out["sub_sections"]) == 2

    # every recorded call whose target message content DIFFERS from the untouched original is an ablated
    # ("without") arm -- there should be exactly one per sub-section, in span order, each with EXACTLY the
    # remapped span removed (start offset by part_start, nothing more, nothing less).
    ablated = [c for c in sub.calls if c[1]["content"] != msg_content]
    assert len(ablated) == 2
    for (a, b), call in zip(spans, ablated):
        expected = msg_content[:part_start + a] + msg_content[part_start + b:]
        assert call[1]["content"] == expected
        assert len(call[1]["content"]) == len(msg_content) - (b - a)   # exactly the sub-span's length gone
