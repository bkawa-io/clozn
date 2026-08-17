"""Model-free contracts for the experimental native reference-match-many seam."""
from __future__ import annotations

from clozn.experiments.multi_arm import probe_reference_match_many
from clozn.server import app as cs
from clozn.server.substrates import EngineSubstrate


CONTRACT = {
    "decode_mode": "greedy",
    "sampling": None,
    "max_new": 3,
    "stop": [],
    "expected_termination": {"reason": "eos", "reason_raw": "eos"},
}


class _NativeEngine:
    def __init__(self):
        self.native_calls = []

    def reference_match_arms(self, arms, *, reference_token_ids, generation_contract,
                             parent_anchor_prompt=None):
        self.native_calls.append((arms, list(reference_token_ids), dict(generation_contract),
                                  parent_anchor_prompt))
        # Deliberately return rows out of order: the Python seam must restore
        # the caller's arm order before evidence is exposed.
        return {
            "results": [
                {"arm_id": 1, "result": {
                    "generated_token_ids": [10, 99],
                    "reply": "x",
                    "finish_reason": "length",
                    "termination": {"kind": "length"},
                    "diverged": True,
                    "diverged_at": 1,
                }},
                {"arm_id": 0, "result": {
                    "generated_token_ids": [10, 11],
                    "reply": "ok",
                    "finish_reason": "stop",
                    "termination": {"kind": "eos"},
                    "diverged": False,
                    "diverged_at": -1,
                }},
            ],
            "metrics": {"native_prefill_time_ns": 7, "proof_grade": False},
        }


def _sub(monkeypatch, engine):
    sub = object.__new__(EngineSubstrate)
    sub.engine = engine
    sub.steer = None
    sub._native_reference_match_arms = True
    sub.last_native_reference_match_metrics = None
    sub.probe_reference_match = lambda **arm: {
        "status": "diverged", "removed_source_ids": arm.get("removed_source_ids")
    }
    monkeypatch.setattr(cs, "_engine_tmpl", lambda _engine, messages, **_kwargs: "PROMPT:" + str(messages))
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    return sub


def _arms():
    return [
        {"messages": [{"role": "user", "content": "a"}],
         "reference_token_ids": [10, 11], "generation_contract": CONTRACT,
         "explicit_conditions": {}},
        {"messages": [{"role": "user", "content": "b"}],
         "reference_token_ids": [10, 11], "generation_contract": CONTRACT,
         "explicit_conditions": {}},
    ]


def test_qualified_native_many_restores_order(monkeypatch):
    class QualifiedSubstrate:
        probe_reference_match_many_proof_grade = True

        def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                                  explicit_conditions=None):
            return {"status": "diverged", "matched_token_count": 0}

        def probe_reference_match_many(self, arms, *, cancel=None):
            return [
                {"arm_index": index, "result": {
                    "status": "matched" if index == 0 else "diverged",
                    "matched_token_count": 2 if index == 0 else 1,
                    "first_divergence_index": None if index == 0 else 1,
                }}
                for index in reversed(range(len(arms)))
            ]

    rows = probe_reference_match_many(QualifiedSubstrate(), _arms())
    assert [row["status"] for row in rows] == ["matched", "diverged"]
    assert rows[1]["first_divergence_index"] == 1


def test_unqualified_native_many_stays_on_scalar_reference(monkeypatch):
    engine = _NativeEngine()
    sub = _sub(monkeypatch, engine)
    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS", "1")
    rows = probe_reference_match_many(sub, _arms())
    assert len(rows) == 2
    assert engine.native_calls == []


def test_proof_grade_default_stays_on_scalar_even_when_native_is_enabled(monkeypatch):
    engine = _NativeEngine()
    sub = _sub(monkeypatch, engine)
    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS", "1")

    rows = probe_reference_match_many(sub, _arms())

    assert len(rows) == 2
    assert engine.native_calls == []


def test_engine_client_reference_match_arms_wire_shape():
    from clozn_engine import EngineClient

    class Client(EngineClient):
        def __init__(self):
            super().__init__(port=1)
            self.seen = None

        def _post(self, path, body):
            self.seen = (path, body)
            return {"results": [{"arm_id": 0, "result": {}}]}

    client = Client()
    result = client.reference_match_arms(
        [{"arm_id": 0, "prompt": "hello"}],
        reference_token_ids=[10, 11],
        generation_contract=CONTRACT,
        parent_anchor_prompt="parent",
    )

    assert result["results"]
    assert client.seen[0] == "/v1/reference-match/arms"
    assert client.seen[1]["arms"] == [{"arm_id": 0, "prompt": "hello"}]
    assert client.seen[1]["reference_token_ids"] == [10, 11]
    assert client.seen[1]["parent_anchor_prompt"] == "parent"


def test_engine_client_persistent_parent_wire_shapes():
    from clozn_engine import EngineClient

    class Client(EngineClient):
        def __init__(self):
            super().__init__(port=1)
            self.seen = []

        def _post(self, path, body):
            self.seen.append((path, body))
            if path.endswith("/create"):
                return {
                    "session_id": "s", "parent_version": 0,
                    "parent_prompt_digest": "p", "runtime_identity": {},
                }
            if path.endswith("/probe"):
                return {"results": [{"candidate_id": "c"}]}
            if path.endswith("/promote"):
                return {"parent_version": 1}
            return {"closed": True}

    client = Client()
    client.reference_match_persistent_create(
        "prompt", reference_token_ids=[1], generation_contract=CONTRACT,
    )
    client.reference_match_persistent_probe(
        "s", expected_parent_version=0,
        children=[{"candidate_id": "c", "candidate_rank": 0, "prompt": "child"}],
    )
    client.reference_match_persistent_promote(
        "s", expected_parent_version=0, candidate_id="c",
    )
    client.reference_match_persistent_close("s")
    assert [path for path, _body in client.seen] == [
        "/v1/reference-match/persistent/create",
        "/v1/reference-match/persistent/probe",
        "/v1/reference-match/persistent/promote",
        "/v1/reference-match/persistent/close",
    ]
    assert client.seen[1][1]["children"][0]["candidate_rank"] == 0
