"""Tests for the shareable receipt card renderer (card_html.render_card) — pure fixture bundles in,
assertions on the HTML string out. No server, no substrate, no fs."""
from __future__ import annotations

import re

import clozn.receipts.bundle as receipt_bundle
from clozn.server.card_html import render_card


def _run(**over) -> dict:
    run = {
        "id": "run_test0000001_abc123",
        "created_at": "2026-07-12T04:11:19",
        "created_ts": 1783854679.9,
        "source": "openai_api",
        "client": "studio",
        "model": "clozn-qwen",
        "substrate": "engine",
        "prompt_summary": "What country is shaped like a boot?",
        "response_summary": "Italy is shaped like a boot.",
        "messages": [{"role": "user", "content": "What country is shaped like a boot?"}],
        "response": "Italy is shaped like a boot.",
        "trace": {
            "tokens": ["Italy", " is", " shaped", " like", " a", " boot", "."],
            "confidence": [0.91, 0.98, 0.95, 0.98, 0.99, 0.44, 0.99],
            "alternatives": [],
        },
        "timing": {"started_at": 1.0, "ended_at": 1.4, "duration_ms": 400},
        "parent_run_id": None,
        "finish_reason": "stop",
        "error": None,
        "meta": {},
    }
    run.update(over)
    return run


_RECEIPTS = {
    "run_id": "run_test0000001_abc123",
    "receipts": [{
        "influence": {"text": "enjoys <geography> trivia"},
        "has_effect": True,
        "causal_verified": True,
        "baseline_reply": "Italy is shaped like a boot.",
        "ablated_reply": "The country shaped like a boot is Italy.",
        "delta": {"words": [6, 8], "wps": [6.0, 8.0], "changed": 38},
    }],
    "forced_receipts": [{
        "influence": {"text": "enjoys <geography> trivia"},
        "mode": "forced",
        "causal_verified": True,
        "has_effect": True,
        "sum_nats": 2.437,
        "mean_nats_per_token": 0.348,
        "top_dependent": [
            {"index": 0, "piece": " Italy", "delta": 1.2},
            {"index": 5, "piece": " boot", "delta": -0.31},
        ],
    }],
    "skipped": [],
    "redundant_pairs": [],
}


def _bundle(run=None, receipts=None) -> dict:
    return receipt_bundle.build(run if run is not None else _run(), explain=None, receipts=receipts)


def _influence_map(*, clear=True) -> dict:
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "prompt_spans": [
            {"id": "p.m000.c000", "role": "system", "source_kind": "assembled_message",
             "text": "Answer with the capital only."},
            {"id": "p.m001.c000", "role": "user", "source_kind": "assembled_message",
             "text": "What is the capital of France?"},
        ],
        "answer_spans": [
            {"id": "a.t0000", "text": "Paris"},
            {"id": "a.t0001", "text": "."},
            {"id": "a.t0002", "text": " Done"},
        ],
        "links": [
            {"context_span_id": "p.m000.c000", "answer_span_id": "a.t0000",
             "delta_nats": 0.72, "abs_delta_nats": 0.72, "effect": "supports",
             "clears_floor": clear},
            {"context_span_id": "p.m001.c000", "answer_span_id": "a.t0000",
             "delta_nats": -0.41, "abs_delta_nats": 0.41, "effect": "suppresses",
             "clears_floor": clear},
            {"context_span_id": "p.m001.c000", "answer_span_id": "a.t0001",
             "delta_nats": 0.16, "abs_delta_nats": 0.16, "effect": "supports",
             "clears_floor": clear},
        ],
        "thresholds": {"cell_abs_delta_nats": 0.05},
        "selection": {"omitted_source_ids": ["p.m002"]},
    }


def _bundle_with_influence(influence=None) -> dict:
    bundle = _bundle()
    bundle["influence_map"] = influence if influence is not None else _influence_map()
    return bundle


# --------------------------------------------------------------- receipts present: numbers, escaped
def test_receipts_render_with_numbers_and_escaping():
    out = render_card(_bundle(receipts=_RECEIPTS))
    # forced-receipt numbers appear
    assert "+2.437" in out               # sum_nats
    assert "+1.200" in out               # top-dependent delta
    assert "-0.310" in out               # negative delta
    assert " Italy" in out               # dependent-token piece
    # influence text appears ESCAPED, never raw
    assert "enjoys &lt;geography&gt; trivia" in out
    assert "enjoys <geography>" not in out
    # verdict chips + derivation tag
    assert "changed the answer" in out
    assert "leave-one-out + forced scoring" in out
    # the honest-absence sentence must NOT appear when receipts exist
    assert "no receipts computed for this run" not in out


# --------------------------------------------------------------------- no receipts: honest absence
def test_no_receipts_renders_honest_absence():
    out = render_card(_bundle())
    assert ("no receipts computed for this run — receipts are measured on demand, "
            "never assumed") in out


# ------------------------------------------------------------------------------- injection-proofing
def test_script_in_reply_is_escaped_inert():
    payload = "<script>alert(1)</script>"
    run = _run(response=payload,
               messages=[{"role": "user", "content": "say " + payload}],
               trace={"tokens": ["<script>", "alert(1)", "</script>"],
                      "confidence": [0.9, 0.9, 0.9], "alternatives": []})
    out = render_card(_bundle(run=run))
    assert "<script" not in out                      # the card ships zero <script> of its own, so: none at all
    assert "&lt;script&gt;" in out                   # the payload is present, inert


# ------------------------------------------------------------------------------- self-containment
def test_no_external_urls_in_src_or_href_attributes():
    out = render_card(_bundle(receipts=_RECEIPTS))
    attrs = re.findall(r"""(?:src|href)\s*=\s*["'][^"']*["']""", out, flags=re.IGNORECASE)
    for a in attrs:
        assert "http://" not in a and "https://" not in a, a
    assert "@import" not in out and "url(" not in out    # CSS can't fetch either


# ------------------------------------------------------- linked context <-> answer influence map
def test_influence_map_links_both_directions_with_css_only_hover_and_focus():
    out = render_card(_bundle_with_influence())
    assert "context ↔ answer influence" in out
    assert "Answer with the capital only." in out and "Paris" in out
    assert out.count('tabindex="0"') == 6
    # Context -> answer and answer -> context selectors are generated only from
    # bounded numeric display indices; no artifact text or IDs enter CSS.
    assert ".imap:has(#ic:checked):has(.im-p0:is(:hover,:focus)) .from-p0-s" in out
    assert ".imap:has(#ia0:checked) .from-a0-s" in out
    assert ".imap:has(#ia0:checked) .from-a0-x" in out
    # A non-clear pin suppresses transient hover/focus, so two relationships
    # cannot visually union.  The clear radio is the explicit hover-mode gate.
    assert ".imap:has(#ic:checked) .im-span:is(:hover,:focus)" in out
    assert ".imap:has(.im-p0:is(:hover,:focus))" not in out
    assert 'type="radio" name="i" id="ip0"' in out
    assert 'for="ic"' in out and "clear pinned highlight" in out
    assert "not percentages, attention weights, or a circuit trace" in out
    assert "measurement floor: 0.050 nats" in out
    assert "1 visible answer span(s) have no clear source" in out
    assert "1 recorded prompt source(s) were outside the bounded measurement" in out
    assert "<script" not in out


def test_influence_map_escapes_span_payloads_and_keeps_them_out_of_css():
    influence = _influence_map()
    influence["prompt_spans"][0]["text"] = '</style><script>alert("prompt")</script>'
    influence["answer_spans"][0]["text"] = '<img src="https://attacker.invalid/x">'
    out = render_card(_bundle_with_influence(influence))
    assert "<script" not in out and "<img" not in out and "</style><script" not in out
    assert "&lt;script&gt;" in out and "&lt;img src=&quot;https://attacker.invalid/x&quot;&gt;" in out
    assert "attacker.invalid" not in re.findall(r"<style>(.*?)</style>", out, re.DOTALL)[0]


def test_influence_map_below_floor_and_unavailable_states_are_explicit():
    out = render_card(_bundle_with_influence(_influence_map(clear=False)))
    assert "No clear source found: no visible context-answer link cleared 0.050 nats." in out
    assert ".from-p0-s{" not in out

    unavailable = {
        "schema": "clozn.context_answer_influence.v1", "status": "unavailable", "available": False,
        "error": {"code": "scoring_unavailable", "message": "teacher-forced scoring was not available"},
    }
    unavailable_out = render_card(_bundle_with_influence(unavailable))
    assert "map unavailable — scoring_unavailable" in unavailable_out
    assert "teacher-forced scoring was not available" in unavailable_out


def test_influence_map_surfaces_the_redundant_pair_check_honestly():
    influence = _influence_map()
    influence["redundancy_check"] = {
        "performed": True,
        "context_span_ids": ["p.m000.c000", "p.m001.c000"],
        "per_answer_token": [
            {"answer_span_id": "a.t0000", "individual_sum_nats": 1.0, "joint_delta_nats": 0.2,
             "interaction_nats": -0.8},
        ],
    }
    out = render_card(_bundle_with_influence(influence))
    assert "Redundant-pair check: context 1 and context 2" in out
    assert "strongest measured interaction 0.800 nats" in out
    assert "never a percentage of total explanation" in out


def test_influence_map_omits_the_redundancy_note_when_not_performed():
    influence = _influence_map()
    influence["redundancy_check"] = {"performed": False, "reason": "fewer than two context spans clear"}
    out = render_card(_bundle_with_influence(influence))
    assert "Redundant-pair check" not in out


def test_influence_map_prompt_spans_prioritize_clearing_ones_when_over_the_display_cap():
    """Coarse-to-fine refinement (Phase 3.7) can push the measured prompt-span count above the card's
    display cap. A naive positional slice would silently drop whichever spans land past the cap -- this
    proves the card instead keeps every clearing span, moving non-clearing filler out of the way first."""
    influence = _influence_map()
    filler = [
        {"id": f"p.filler{i:03d}", "role": "user", "source_kind": "assembled_message", "text": f"filler {i}"}
        for i in range(7)
    ]
    influence["prompt_spans"] = filler + influence["prompt_spans"]
    out = render_card(_bundle_with_influence(influence))
    # Both real, clearing spans still render even though 9 total spans exceed the cap of 8 -- a naive
    # `[:8]` positional slice would have cut the second one.
    assert "Answer with the capital only." in out
    assert "What is the capital of France?" in out
    assert "interactive view was truncated for receipt size" in out


def test_influence_map_answer_surface_is_bounded_and_says_when_truncated():
    influence = _influence_map(clear=False)
    influence["answer_spans"] = [
        {"id": f"a.t{i:04d}", "text": "x"} for i in range(300)
    ]
    influence["links"] = []
    out = render_card(_bundle_with_influence(influence))
    assert "im-a255" in out and "im-a256" not in out
    assert "interactive view was truncated for receipt size" in out


# ------------------------------------------------------------------------------- structure smoke
def test_structure_smoke():
    out = render_card(_bundle(receipts=_RECEIPTS))
    assert out.startswith("<!doctype html>")
    assert out.count("<html") == 1 and out.count("</html>") == 1
    # sections, in order
    for section in ("run receipt", "the exchange", "influences &amp; receipts",
                    "lens readouts", "lineage"):
        assert section in out, section
    # masthead + footer both carry the run id
    assert out.count("run_test0000001_abc123") >= 2
    # The receipt carries both Studio themes without a script or external dependency.
    assert 'class="receipt-shell"' in out
    assert 'class="theme-toggle-input"' in out
    assert "halo" in out and "cathedral" in out
    for signal in ("--signal-cyan", "--signal-mint", "--signal-violet",
                   "--signal-pink", "--signal-peach"):
        assert signal in out
    # Studio uses square instrument geometry throughout.
    assert "border-radius" not in out
    # per-token confidence shading: opacity varies with conf, low-conf gets the dotted class
    assert 'class="tk lo"' in out
    # Captured and derived remain explicit, but the legend is compact.
    assert "run record" in out
    assert "post-run computation" in out


# ------------------------------------------------------------------------------- lens readouts
def test_lens_readouts_render_with_provenance():
    run = _run()
    run["trace"]["workspace_readouts"] = [
        {"type": "workspace_readout", "run_id": run["id"], "token_index": 0, "token_text": "Italy",
         "layer": 25, "position": 0, "provider": "jacobian_lens_l25",
         "provider_type": "jacobian_lens", "readout_kind": "token",
         "top_readouts": [{"label": " peninsula", "score": 0.8}, {"label": " coast", "score": 0.5}]},
    ]
    out = render_card(_bundle(run=run))
    assert "layer 25" in out
    assert "peninsula" in out and "coast" in out
    assert "jacobian_lens_l25" in out
    assert "NOT the model&#x27;s literal thought" in out     # the honesty caption, escaped
    assert "no lens readout recorded" not in out


def test_lens_absent_renders_honest_one_liner():
    out = render_card(_bundle())
    assert "no lens readout recorded on this run" in out


# ------------------------------------------------------------------------------------ lineage
def test_lineage_tree_renders_parent_and_children():
    bundle = _bundle(run=_run(parent_run_id="run_parent00001_aaaaaa"))
    bundle["lineage"] = {
        "run_id": "run_test0000001_abc123",
        "tree": {"id": "run_parent00001_aaaaaa", "change_label": None, "children": [
            {"id": "run_test0000001_abc123", "change_label": "re-roll", "is_current": True,
             "children": [{"id": "run_child000001_bbbbbb", "change_label": "memory off",
                           "children": []}]},
        ]},
    }
    out = render_card(bundle)
    assert "run_parent00001_aaaaaa" in out
    assert "run_child000001_bbbbbb" in out
    assert "this run" in out


def test_lineage_absent_is_honest():
    out = render_card(_bundle())
    assert "no lineage — an original run" in out


# ------------------------------------------------------------------------- degrades, never raises
def test_degenerate_bundles_never_raise():
    for bundle in ({}, {"run": None}, {"run": {}, "trace": None},
                   receipt_bundle.build(None)):
        out = render_card(bundle)
        assert out.startswith("<!doctype html>")
        assert "run receipt" in out
