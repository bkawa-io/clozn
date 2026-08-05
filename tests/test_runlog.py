"""Schema-level, no-model tests for research/runlog.py (roadmap issue I3).

Covers record -> list_runs -> get_run, asserts the run schema fields exist, that the cheap UI flags are
computed (a run carrying a memory card gets 'memory'; a low-confidence trace gets 'low-confidence'; etc.),
and that pruning keeps <= KEEP runs. The store is isolated by pointing runlog.RUNS_DIR at a pytest tmp dir
at runtime -- RUNS_DIR is a module global, so we don't need to (and per I3 must not) edit runlog.py.
"""
from contextlib import closing
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/ on path
import clozn.runs.store as runlog
from clozn.runs import summaries  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """Redirect the run store to a temp dir for the duration of one test."""
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


def test_record_returns_id_and_persists(store):
    rid = store.record(source="cli", client="clozn-cli", model="qwen", substrate="QwenSubstrate",
                       messages=[{"role": "user", "content": "hi there"}], response="hello")
    assert rid is not None
    assert rid.startswith("run_")
    rec = store.get_run(rid)
    assert rec is not None
    assert rec["id"] == rid


def test_watch_cursor_uses_journal_insertion_order_not_generation_start(store):
    first = store.record(source="api", messages=[{"role": "user", "content": "fast"}],
                         response="first", started=2000.0, ended=2001.0)
    cursor = store.cursor_for_run(first)
    # This request started much earlier but finalized after the cursor was captured.
    late = store.record(source="api", messages=[{"role": "user", "content": "slow"}],
                        response="second", started=1000.0, ended=2002.0)
    page = store.runs_after(cursor)
    assert [run["id"] for run in page["runs"]] == [late]
    assert store.latest_run()["id"] == late


def test_association_filters_are_exact_and_derived_runs_default_off(store):
    from clozn.runs.association import client_key, session_key
    ck = client_key("client-a")
    sk = session_key("session-a")
    organic = store.record(source="openai_api", client_key=ck, session_key=sk,
                           messages=[{"role": "user", "content": "one"}], response="organic")
    store.record(source="replay", client_key=ck, session_key=sk,
                 messages=[{"role": "user", "content": "two"}], response="derived")
    assert store.latest_run(client_id="client-a", session_id="session-a")["id"] == organic
    assert store.latest_run(client_id="different") is None


def test_record_persists_optional_opaque_project_association(store):
    from clozn.runs.association import project_key
    opaque = project_key("workspace-one")
    rid = store.record(source="openai_api", project_key=opaque,
                       messages=[{"role": "user", "content": "one"}], response="answer")

    assert store.get_run(rid)["project_key"] == opaque


def test_record_schema_fields(store):
    rid = store.record(source="studio_chat",
                       messages=[{"role": "user", "content": "what is 2+2?"}], response="4")
    rec = store.get_run(rid)
    for k in ("id", "created_at", "created_ts", "source", "client", "model", "substrate",
              "prompt_summary", "response_summary", "messages", "response", "memory", "behavior",
              "assembled_messages", "final_prompt", "trace", "timing", "parent_run_id",
              "changes_applied", "error", "project_key", "output_contract", "flags"):
        assert k in rec, f"missing schema field {k}"
    assert rec["source"] == "studio_chat"
    assert rec["prompt_summary"] == "what is 2+2?"        # last user message summarized
    assert rec["response_summary"] == "4"
    assert set(("started_at", "ended_at", "duration_ms")).issubset(rec["timing"])


def test_output_contract_round_trips_and_sets_compact_tool_call_flag(store):
    contract = {
        "schema": "clozn.structured_io.v1",
        "mode": "tools",
        "raw_output": '{"type":"tool_call","name":"weather","arguments":{"city":"Oslo"}}',
        "qualification": {"model_sha256": "a" * 64, "parser_id": "parser-v1"},
        "outcome": {"status": "parsed", "kind": "tool_call", "tool_name": "weather"},
        "recovery": {"policy": "none", "attempts": []},
    }
    rid = store.record(
        source="openai_api", messages=[{"role": "user", "content": "weather?"}], response="",
        output_contract=contract,
    )

    saved = store.get_run(rid)
    assert saved["output_contract"] == contract
    assert "tool-call" in saved["flags"]
    summary = next(row for row in store.list_runs() if row["id"] == rid)
    assert "tool-call" in summary["flags"]


def test_output_contract_parse_error_flag_is_compact_and_malformed_shapes_are_safe(store):
    failed = store.record(
        source="openai_api", messages=[{"role": "user", "content": "json"}], response="not-json",
        error="malformed_model_output: expected one JSON object",
        output_contract={
            "schema": "clozn.structured_io.v1",
            "mode": "json_schema",
            "outcome": {"status": "error", "kind": "parse_error",
                        "code": "malformed_model_output"},
        },
    )
    flags = store.get_run(failed)["flags"]
    assert "output-parse-error" in flags
    assert "error" in flags

    # Direct/legacy callers are not trusted to honor the type annotation. Bad evidence must be dropped,
    # never turn a logging-only feature into loss of the entire run.
    malformed = store.record(
        source="legacy", messages=[{"role": "user", "content": "hi"}], response="hello",
        output_contract=["not", "an", "object"],  # type: ignore[arg-type]
    )
    assert malformed is not None
    saved = store.get_run(malformed)
    assert saved["output_contract"] == {}
    assert "tool-call" not in saved["flags"]
    assert "output-parse-error" not in saved["flags"]


def test_record_persists_assembled_messages_when_provided(store):
    assembled = [{"role": "system", "content": "MEMORY BLOCK"},
                 {"role": "user", "content": "what is 2+2?"}]
    rid = store.record(source="studio_chat",
                       messages=[{"role": "user", "content": "what is 2+2?"}],
                       response="4", assembled_messages=assembled)
    assert store.get_run(rid)["assembled_messages"] == assembled


def test_record_persists_final_prompt_when_provided(store):
    """backlog #5: the EXACT rendered chat-template string the model saw is stored on the run record."""
    rendered = ("<|im_start|>system\nMEMORY BLOCK<|im_end|>\n"
                "<|im_start|>user\nwhat is 2+2?<|im_end|>\n<|im_start|>assistant\n")
    rid = store.record(source="engine_chat",
                       messages=[{"role": "user", "content": "what is 2+2?"}],
                       response="4", final_prompt=rendered)
    assert store.get_run(rid)["final_prompt"] == rendered


def test_record_final_prompt_defaults_to_none(store):
    """No rendered string in hand (e.g. a torch substrate) -> the field is present but None; consumers
    then fall back to assembled_messages. Present-but-None, never a KeyError."""
    rid = store.record(source="studio_chat",
                       messages=[{"role": "user", "content": "hi"}], response="hey")
    rec = store.get_run(rid)
    assert "final_prompt" in rec
    assert rec["final_prompt"] is None


def test_explicit_import_loads_legacy_run_without_final_prompt(store, tmp_path):
    """Legacy JSON is imported once; normal reads never fall back to it implicitly."""
    import json
    legacy_dir = tmp_path / "legacy-runs"
    legacy_dir.mkdir()
    legacy = {"id": "run_legacy_000000", "source": "engine_chat",
              "messages": [{"role": "user", "content": "hi"}], "response": "hey",
              "assembled_messages": [{"role": "user", "content": "hi"}]}   # NOTE: no "final_prompt" key
    with open(legacy_dir / "run_legacy_000000.json", "w", encoding="utf-8") as f:
        json.dump(legacy, f)
    assert store.get_run("run_legacy_000000") is None
    assert store.import_json_dir(str(legacy_dir))["imported"] == 1
    rec = store.get_run("run_legacy_000000")
    assert rec is not None                                  # loads without crashing
    assert rec.get("final_prompt") is None                 # absent -> None via .get(), the documented fallback
    assert rec["assembled_messages"] == [{"role": "user", "content": "hi"}]


def test_log_run_forwards_final_prompt_to_the_record(store, monkeypatch):
    """The handler glue (backlog #5): _log_run reads mem_out['final_prompt'] and persists it as
    run.final_prompt. Drives the REAL do_POST handler object with no socket + SUB=None (no engine, no
    model) -- purely the forwarding logic, mirroring test_rederive_server.py's object.__new__(H) trick."""
    import time
    from clozn.server import app as cs
    monkeypatch.setattr(cs, "SUB", None)                   # no substrate -> dials {}, run_meta skipped
    h = object.__new__(cs.make_handler())
    h.headers = {"User-Agent": "pytest"}
    rendered = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    rid = h._log_run("engine_chat", [{"role": "user", "content": "hi"}], "hey", "clozn-engine", time.time(),
                     mem_out={"mode": "prompt", "applied": [], "gate": 0.0,
                              "assembled_messages": [{"role": "user", "content": "hi"}],
                              "final_prompt": rendered})
    assert rid is not None
    assert store.get_run(rid)["final_prompt"] == rendered




def test_log_run_honors_surface_reported_dials_instead_of_claiming_live_state(store, monkeypatch):
    """Raw completion reports what actually reached the worker, even if a live dial is configured."""
    import time
    from clozn.server import app as cs

    class Steer:
        def active(self):
            return {"warm": 0.8}

    class Sub:
        steer = Steer()

    monkeypatch.setattr(cs, "SUB", Sub())
    h = object.__new__(cs.make_handler())
    h.headers = {"User-Agent": "pytest"}
    rid = h._log_run(
        "openai_completion", [{"role": "user", "content": "raw"}], "reply", "model", time.time(),
        mem_out={"mode": "prompt", "applied": [], "active_dials": {}, "final_prompt": "raw"},
    )
    assert store.get_run(rid)["behavior"]["active_dials"] == {}


def test_no_loop_guard_flag_when_the_guard_never_fired(store):
    rid = store.record(source="openai_api", messages=[{"role": "user", "content": "q"}], response="a",
                       memory={"cards_applied": ["x"]})
    flags = store.get_run(rid)["flags"]
    assert "memory-retried" not in flags
    assert "memory-loop-guard" not in flags


def test_prompt_summary_uses_last_user_message(store):
    rid = store.record(source="cli", messages=[
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ], response="ok")
    assert store.get_run(rid)["prompt_summary"] == "second"


def test_flag_memory(store):
    rid = store.record(source="studio_chat", messages=[{"role": "user", "content": "q"}],
                       response="a", memory={"cards_applied": ["mem_1"]})
    assert "memory" in store.get_run(rid)["flags"]


def test_flag_pending_memory(store):
    rid = store.record(source="studio_chat", messages=[{"role": "user", "content": "q"}],
                       response="a", memory={"proposed_cards": ["mem_2"]})
    assert "pending-memory" in store.get_run(rid)["flags"]


def test_flag_steered(store):
    rid = store.record(source="studio_chat", messages=[{"role": "user", "content": "q"}],
                       response="a", behavior={"active_dials": {"concise": 0.4}})
    assert "steered" in store.get_run(rid)["flags"]


def test_flag_low_confidence(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       trace={"tokens": ["x", "y"], "confidence": [0.9, 0.12]})
    assert "low-confidence" in store.get_run(rid)["flags"]


def test_flag_replayed_and_error(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       parent_run_id="run_parent", error="boom")
    flags = store.get_run(rid)["flags"]
    assert "replayed" in flags
    assert "error" in flags


def test_no_spurious_flags(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="short answer",
                       trace={"tokens": ["a"], "confidence": [0.95]})
    assert store.get_run(rid)["flags"] == []


# ---- trace-blob integrity: a corrupt/missing blob is SURFACED, never a silent empty {} (BACKLOG §2) ----

def test_corrupt_trace_blob_is_surfaced_not_silently_empty(store):
    """A tampered trace blob must read back as corrupt, never as an empty {} -- a run may not present 'no
    trace' when the truth is 'the causal evidence was altered'. The digest is verified on every read."""
    import glob
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       trace={"tokens": ["x", "y"], "confidence": [0.9, 0.12]})
    good = store.get_run(rid)["trace"]
    assert good and "unavailable" not in good          # a valid blob loads clean (verified against its digest)

    blobs = glob.glob(os.path.join(runlog.RUNS_DIR, "blobs", "sha256", "**", "*.json"), recursive=True)
    assert len(blobs) == 1
    with open(blobs[0], "w", encoding="utf-8") as handle:
        handle.write('{"tokens": ["TAMPERED"], "confidence": [0.01]}')   # valid JSON, wrong bytes -> mismatch

    corrupt = store.get_run(rid)["trace"]
    assert corrupt.get("unavailable") == "trace blob corrupt (digest mismatch)"
    assert corrupt.get("sha256")                        # the affected blob's digest is reported
    assert "TAMPERED" not in str(corrupt)               # tampered content is NOT served as the real trace


def test_missing_trace_blob_is_surfaced_not_silently_empty(store):
    """A dangling trace ref (its blob deleted/pruned out from under it) reads back as missing, not {}."""
    import glob
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       trace={"tokens": ["a"], "confidence": [0.95]})
    blobs = glob.glob(os.path.join(runlog.RUNS_DIR, "blobs", "sha256", "**", "*.json"), recursive=True)
    os.remove(blobs[0])
    gone = store.get_run(rid)["trace"]
    assert gone.get("unavailable") == "trace blob missing"
    assert gone.get("sha256")


# -------------------------------------------------------- evidence-write failures (BACKLOG §2, honesty)
# The old `_store_trace` let `atomic_write_json` raise straight up through `_pack`/`_put`/`record`'s
# blanket `except Exception: return None` -- a disk hiccup while writing the TRACE blob silently discarded
# the ENTIRE run (prompt, response, everything), with nothing but a bare None to show for it. The fix:
# `_store_trace` catches its own write failure, logs a warning, and hands it back via the trace_ref so
# `_pack` can mark the run "evidence missing" -- the row still lands (see BACKLOG §2 in store.py).

def test_trace_write_failure_still_persists_the_run_row(store, monkeypatch):
    """A trace blob write failure must NOT sink the whole run -- the row (prompt, response, model, ...)
    still lands; only the trace itself is honestly marked as lost."""
    def _broken_write(path, obj, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(store, "atomic_write_json", _broken_write)
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hello",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    assert rid is not None                                # the run itself was NOT discarded

    monkeypatch.undo()                                     # reads must work normally again
    rec = store.get_run(rid)
    assert rec is not None
    assert rec["response"] == "hello"                       # everything else survived intact


def test_trace_write_failure_reads_back_as_evidence_missing_not_empty(store, monkeypatch):
    """The honesty invariant: a run whose trace failed to persist must read as {"unavailable": ...}, the
    SAME shape _load_trace already uses for a corrupt/missing blob (BACKLOG §2, commit 6409535) -- never a
    plain {} that would look identical to "this run genuinely carried no trace"."""
    def _broken_write(path, obj, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(store, "atomic_write_json", _broken_write)
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hello",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    monkeypatch.undo()

    rec = store.get_run(rid)
    trace = rec["trace"]
    assert trace.get("unavailable", "").startswith("trace evidence write failed")
    assert trace.get("sha256")                              # still correlatable to a digest
    assert "TAMPERED" not in str(trace) and "a" not in trace.get("tokens", [])  # no fabricated content


def test_trace_write_failure_sets_evidence_missing_flag_and_meta_marker(store, monkeypatch):
    """Both surfaces the honesty invariant promises: the cheap `flags` list (what the Runs page filters
    on) and `meta` (the detailed record) must each independently carry the failure."""
    def _broken_write(path, obj, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(store, "atomic_write_json", _broken_write)
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hello",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    monkeypatch.undo()

    rec = store.get_run(rid)
    assert "evidence-missing" in rec["flags"]
    assert "evidence_write_failed" in rec["meta"]
    assert "simulated disk full" in rec["meta"]["evidence_write_failed"]


def test_trace_write_failure_flag_visible_in_list_runs_summary(store, monkeypatch):
    """The flag must reach the summary view too (list_runs), not just get_run's full record -- that's the
    surface the Runs page actually filters on."""
    def _broken_write(path, obj, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(store, "atomic_write_json", _broken_write)
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hello",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    monkeypatch.undo()

    rows = {r["id"]: r for r in store.list_runs()}
    assert "evidence-missing" in rows[rid]["flags"]


def test_trace_write_failure_is_logged(store, monkeypatch, caplog):
    """At minimum a logged warning (BACKLOG §2's explicit requirement) -- not just a silent marker."""
    import logging

    def _broken_write(path, obj, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(store, "atomic_write_json", _broken_write)
    with caplog.at_level(logging.WARNING, logger="clozn.runs.store"):
        store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hello",
                     trace={"tokens": ["a"], "confidence": [0.9]})
    assert any("trace blob write failed" in r.message for r in caplog.records)


def test_trace_write_success_never_gets_evidence_missing_flag(store):
    """Negative control: a normal, successful write must NOT carry the marker -- guards against the flag
    logic firing unconditionally."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hello",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    rec = store.get_run(rid)
    assert "evidence-missing" not in rec["flags"]
    assert "evidence_write_failed" not in rec["meta"]
    assert "unavailable" not in rec["trace"]


def test_trace_write_failure_does_not_orphan_a_half_written_blob_file(store, monkeypatch):
    """atomic_write_json itself already guarantees no partial file is ever left at the real path on
    failure (see clozn/_io.py) -- confirm that guarantee holds through this call path too: no blob file
    should exist at the digest's path after a simulated failure."""
    def _broken_write(path, obj, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(store, "atomic_write_json", _broken_write)
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hello",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    monkeypatch.undo()

    rec = store.get_run(rid)
    digest = rec["trace"]["sha256"]
    assert not os.path.isfile(store._blob_path(digest))


# -------- Phase 3.7 persistence: the influence map shares the trace blob machinery (store._pack) --------

def _attach_influence_map(store, rid: str, influence_map: dict) -> None:
    run = store.get_run(rid)
    run["influence_map"] = influence_map
    assert store.replace_run(run)


def test_influence_map_persists_as_a_blob_and_round_trips(store):
    import glob
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    influence_map = {
        "schema": "clozn.context_answer_influence.v1", "status": "ok", "available": True,
        "prompt_spans": [{"id": "p.m000.c000", "text": "context"}],
        "answer_spans": [{"id": "a.t0000", "text": "answer"}],
        "matrix": [[0.42]],
    }
    _attach_influence_map(store, rid, influence_map)

    rec = store.get_run(rid)
    assert rec["influence_map"] == influence_map

    # It really did go through the blob path, not stay inline in the row.
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id=?", (rid,)).fetchone()
    raw = json.loads(row["payload_json"])
    assert "influence_map" not in raw
    assert raw["influence_map_ref"]["sha256"]
    # One blob for the (empty) trace this run carries, one for the influence map -- each content-addressed.
    blobs = glob.glob(os.path.join(runlog.RUNS_DIR, "blobs", "sha256", "**", "*.json"), recursive=True)
    assert len(blobs) == 2
    assert store._blob_path(raw["influence_map_ref"]["sha256"]) in blobs


def test_influence_map_absent_by_default_not_empty(store):
    """A run that never had a map computed carries no influence_map key at all -- unlike trace (always
    {} when absent), influence_map's presence itself is the "was this ever computed" signal."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    rec = store.get_run(rid)
    assert "influence_map" not in rec


def test_corrupt_influence_map_blob_is_surfaced_not_silently_empty(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    _attach_influence_map(store, rid, {"schema": "clozn.context_answer_influence.v1", "matrix": [[1.0]]})
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id=?", (rid,)).fetchone()
    digest = json.loads(row["payload_json"])["influence_map_ref"]["sha256"]
    with open(store._blob_path(digest), "w", encoding="utf-8") as handle:
        handle.write('{"schema": "TAMPERED"}')

    corrupt = store.get_run(rid)["influence_map"]
    assert corrupt.get("unavailable") == "influence map blob corrupt (digest mismatch)"
    assert "TAMPERED" not in str(corrupt)


def test_missing_influence_map_blob_is_surfaced_not_silently_empty(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    _attach_influence_map(store, rid, {"schema": "clozn.context_answer_influence.v1", "matrix": [[1.0]]})
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id=?", (rid,)).fetchone()
    digest = json.loads(row["payload_json"])["influence_map_ref"]["sha256"]
    os.remove(store._blob_path(digest))

    gone = store.get_run(rid)["influence_map"]
    assert gone.get("unavailable") == "influence map blob missing"
    assert gone.get("sha256")


def test_influence_map_write_failure_persists_the_run_and_flags_it_distinctly_from_trace(store, monkeypatch):
    """A disk hiccup writing the influence-map blob must not sink the run OR the trace, and must be
    flagged under its own name so it's never confused with a trace evidence failure."""
    def _broken_write(path, obj, **kwargs):
        raise OSError("simulated disk full")

    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    run = store.get_run(rid)
    run["influence_map"] = {"schema": "clozn.context_answer_influence.v1", "matrix": [[1.0]]}
    monkeypatch.setattr(store, "atomic_write_json", _broken_write)
    assert store.replace_run(run)
    monkeypatch.undo()

    rec = store.get_run(rid)
    assert rec["response"] == "a"
    assert "unavailable" not in rec["trace"]                # the trace blob write was never touched
    assert rec["influence_map"].get("unavailable", "").startswith("influence map evidence write failed")
    assert "influence-evidence-missing" in rec["flags"]
    assert "evidence-missing" not in rec["flags"]           # distinct from the trace-failure flag
    assert "influence_evidence_write_failed" in rec["meta"]
    assert "simulated disk full" in rec["meta"]["influence_evidence_write_failed"]


def test_legacy_inline_influence_map_still_reads_back(store):
    """A row written before blob-backed persistence existed (influence_map inline in payload_json, no
    _ref) must still read back exactly -- unpack only resolves a ref when one is present."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    with closing(store._connect()) as db, db:
        row = db.execute("SELECT payload_json FROM runs WHERE id=?", (rid,)).fetchone()
        payload = json.loads(row["payload_json"])
        payload["influence_map"] = {"schema": "clozn.context_answer_influence.v1", "matrix": [[0.1]]}
        db.execute("UPDATE runs SET payload_json=? WHERE id=?",
                  (json.dumps(payload), rid))

    rec = store.get_run(rid)
    assert rec["influence_map"] == {"schema": "clozn.context_answer_influence.v1", "matrix": [[0.1]]}


def test_list_runs_newest_first(store):
    # ids embed a ms timestamp; pass increasing `started` so ordering is deterministic
    r1 = store.record(source="cli", messages=[{"role": "user", "content": "one"}], response="1",
                      started=1000.0, ended=1000.1)
    r2 = store.record(source="cli", messages=[{"role": "user", "content": "two"}], response="2",
                      started=2000.0, ended=2000.1)
    r3 = store.record(source="cli", messages=[{"role": "user", "content": "three"}], response="3",
                      started=3000.0, ended=3000.1)
    ids = [r["id"] for r in store.list_runs()]
    assert ids == [r3, r2, r1]                            # newest first


def test_list_runs_returns_summary_fields_only(store):
    store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="yo")
    rows = store.list_runs()
    assert len(rows) == 1
    row = rows[0]
    # The confidence group is omitted wholesale on a run with no recorded trace rather than sent as
    # nulls -- `confidence_min: null` invites `?? 0` at a call site, and 0 is a real and terrible
    # confidence value, not a missing one. Every other field is still always present, because callers
    # index those directly and have always seen null for absent.
    assert set(row.keys()) <= set(runlog.SUMMARY_FIELDS)
    assert set(runlog.SUMMARY_FIELDS) - set(row.keys()) <= set(summaries._CONFIDENCE_FIELDS)
    # the heavy fields are intentionally NOT in the summary
    assert "messages" not in row
    assert "trace" not in row


def test_list_runs_limit(store):
    for i in range(5):
        store.record(source="cli", messages=[{"role": "user", "content": str(i)}], response=str(i),
                     started=1000.0 + i, ended=1000.0 + i)
    assert len(store.list_runs(limit=3)) == 3


# ------------------------------------------------------------------ include_replays (receipt-journal spam)
# A `/runs/<id>/receipts` "prove-all" persists one child run per arm (baseline + one per fired influence +
# one per redundancy-guard pair -- clozn.replay.replay.replay(), source="replay") so each is itself an
# inspectable, diffable run. Real usage surfaced this as noise: one "prove-all" click produced ~7
# near-duplicate entries in a run-history listing. include_replays=False lets a browsing view opt out
# without losing the underlying data (still fully readable via get_run()).

def test_list_runs_include_replays_true_by_default(store):
    store.record(source="cli", messages=[{"role": "user", "content": "real"}], response="a", started=1000.0)
    store.record(source="replay", messages=[{"role": "user", "content": "real"}], response="b", started=2000.0)
    assert len(store.list_runs()) == 2


def test_list_runs_include_replays_false_drops_replay_sourced_entries(store):
    real = store.record(source="cli", messages=[{"role": "user", "content": "real"}], response="a",
                        started=1000.0)
    store.record(source="replay", messages=[{"role": "user", "content": "real"}], response="b",
                 started=2000.0, parent_run_id=real)
    rows = store.list_runs(include_replays=False)
    assert [r["id"] for r in rows] == [real]


def test_list_runs_include_replays_false_still_honors_limit(store):
    for i in range(5):
        store.record(source="cli", messages=[{"role": "user", "content": str(i)}], response=str(i),
                     started=1000.0 + i, ended=1000.0 + i)
        store.record(source="replay", messages=[{"role": "user", "content": str(i)}], response=str(i),
                     started=1000.5 + i, ended=1000.5 + i)
    rows = store.list_runs(limit=3, include_replays=False)
    assert len(rows) == 3
    assert all(r.get("source") != "replay" for r in
              [store.get_run(r["id"]) for r in rows])


def test_list_runs_include_replays_false_replay_still_readable_by_id(store):
    """The filter hides replay children from a listing view -- it must never make them unreachable."""
    real = store.record(source="cli", messages=[{"role": "user", "content": "real"}], response="a",
                        started=1000.0)
    replay_id = store.record(source="replay", messages=[{"role": "user", "content": "real"}], response="b",
                             started=2000.0, parent_run_id=real)
    store.list_runs(include_replays=False)   # must not affect what's on disk
    assert store.get_run(replay_id) is not None
    assert store.get_run(replay_id)["source"] == "replay"


def test_get_run_missing(store):
    assert store.get_run("run_does_not_exist") is None


# ---------------------------------------------------------------------------------- path traversal (security)
# GET /runs/<rid>/... hands `rid` to get_run() straight from the URL path -- a hostile `rid` like
# "../config" or an absolute path must never let a read escape RUNS_DIR onto some other readable .json
# (e.g. ~/.clozn/config.json). None of these attempts should raise either -- get_run() never raises,
# an unsafe id is just treated like "not found".

def test_get_run_rejects_dotdot_traversal(store, tmp_path):
    secret = tmp_path / "config.json"
    secret.write_text('{"api_key": "super-secret"}', encoding="utf-8")
    os.makedirs(store.RUNS_DIR, exist_ok=True)
    assert store.get_run("../config") is None
    assert store.get_run("..\\config") is None


def test_get_run_rejects_absolute_path(store, tmp_path):
    secret = tmp_path / "config.json"
    secret.write_text('{"api_key": "super-secret"}', encoding="utf-8")
    abs_no_ext = str(secret)[:-len(".json")]           # get_run appends ".json" itself
    assert store.get_run(abs_no_ext) is None
    assert store.get_run("/etc/passwd") is None


def test_get_run_rejects_non_string_or_empty_id(store):
    """The degenerate-run-record case (#3): a `{}` on disk summarizes to id=None; get_run(None) must
    return None cleanly, never raise (the old code did `None + ".json"` -> TypeError)."""
    assert store.get_run(None) is None
    assert store.get_run("") is None
    assert store.get_run(123) is None


def test_update_tiny_tests_rejects_traversal_and_writes_nothing_outside_runs_dir(store, tmp_path):
    """`clozn test --attach` reaches attachments.update_tiny_tests(rid, ...) with a CLI-supplied rid --
    same guard, same contract (False on a bad id), and this is the WRITE side: confirm nothing lands
    outside RUNS_DIR at all, not just that the return value is honest."""
    target = tmp_path / "config.json"
    target.write_text('{"api_key": "super-secret"}', encoding="utf-8")
    os.makedirs(store.RUNS_DIR, exist_ok=True)

    assert store.update_tiny_tests("../config", [{"a": 1}]) is False
    assert store.update_tiny_tests("..\\config", [{"a": 1}]) is False
    assert target.read_text(encoding="utf-8") == '{"api_key": "super-secret"}'   # untouched
    assert not os.path.isfile(os.path.join(store.RUNS_DIR, "config.json"))       # nothing spilled inside either


# -------------------------------------------------------------- round-2 pressure test #1 (HIGH): atomic writes
# record() and update_tiny_tests() are the other two user-data JSON writers sharing the same defect memory
# cards / settings had: open(path, "w") then json.dump(obj, f) truncates the target BEFORE a non-
# serializable value can raise. Both already swallow exceptions and return None/False (never raise out to
# the caller) -- the fix is that a bad write must never destroy/corrupt an existing run record, and must
# never leave a stray truncated file behind for a run that never really completed.

def test_record_bad_meta_fails_cleanly_and_leaves_other_runs_untouched(store):
    good_rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok")
    assert store.get_run(good_rid) is not None
    before = set(os.listdir(store.RUNS_DIR))

    bad_rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok",
                           meta={"bad": {1, 2, 3}})          # a set -- json can't serialize it
    assert bad_rid is None                                    # documented contract: None on failure

    after = set(os.listdir(store.RUNS_DIR))
    assert after == before                                    # no stray/truncated file left behind
    assert store.get_run(good_rid)["response"] == "ok"        # the earlier good run is untouched


def test_update_tiny_tests_bad_payload_fails_cleanly_and_leaves_the_run_record_untouched(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok")
    assert store.update_tiny_tests(rid, [{"name": "t1", "passed": True}]) is True
    assert store.get_run(rid)["tiny_tests"] == [{"name": "t1", "passed": True}]

    # a nested set inside the list -- update_tiny_tests coerces the outer value with list(), but a
    # non-serializable value nested inside an entry still reaches json.dumps
    ok = store.update_tiny_tests(rid, [{"name": "t2", "detail": {1, 2, 3}}])
    assert ok is False

    assert store.get_run(rid)["tiny_tests"] == [{"name": "t1", "passed": True}]   # prior attachment survives


def test_pruning_keeps_at_most_KEEP(store, monkeypatch):
    monkeypatch.setattr(runlog, "KEEP", 3)               # shrink the cap so the test is cheap
    ids = []
    for i in range(6):
        ids.append(store.record(source="cli", messages=[{"role": "user", "content": str(i)}],
                                response=str(i), started=1000.0 + i, ended=1000.0 + i))
    remaining = {r["id"] for r in store.list_runs(limit=100)}
    assert len(remaining) <= 3
    # the 3 most recent survived; the oldest were pruned
    assert remaining == set(ids[-3:])
    assert store.get_run(ids[0]) is None
