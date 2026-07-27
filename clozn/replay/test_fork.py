"""test_fork -- fork-at-token (clozn/replay/fork.py + the /runs/<id>/fork route), MODEL-FREE.

No model, no GPU, no live engine: a FakeEngine stands in for EngineClient (raw-prompt .complete +
optional .apply_template) and a ScoringSub adds the score_tokens seam the retokenization check
duck-types against -- the same fake-substrate pattern test_replay.py / test_timetravel.py use. The run
store is isolated to a pytest tmp dir via runlog.RUNS_DIR.

Covers the behavior contract:
  * fork at position 0 / mid / last -- prompt = final_prompt + pieces[0..p) + forced piece, greedy
    decode (temperature 0 / rep_penalty 1 / seed 0), child linkage (parent_run_id, source "fork",
    changes_applied = {"fork": {...}}), and the divergence-point fields (prefix_kept /
    forked_from_piece).
  * recorded-alternative vs free-token honesty (was_recorded_alternative), token_id resolution.
  * retokenization flag: verified-exact -> False; boundary shift -> True; no score seam -> True
    (can't prove exact => flagged) + the note says so.
  * validation ValueErrors (no trace / out of range / unresolvable token) and the route's 404/400/503
    shapes; the route module is exercised directly (it is deliberately NOT registered in app.py yet).
"""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.replay import fork as fork_mod


# =================================================================================== fakes
class FakeEngine:
    """Quacks like EngineClient for the two calls fork.py makes: raw-prompt complete() (greedy
    continuation) and, optionally, apply_template() (the no-final_prompt fallback)."""

    def __init__(self, continuation=" and onward", finish="stop", template=None):
        self.continuation = continuation
        self.finish = finish
        self.template = template          # None -> no apply_template attribute behavior via _NoTemplate
        self.calls = []

    def complete(self, prompt, **params):
        self.calls.append(dict(params, prompt=prompt))
        return {"choices": [{"text": self.continuation, "finish_reason": self.finish}]}

    def apply_template(self, messages, add_assistant=True):
        if self.template is None:
            raise RuntimeError("no template configured")
        return self.template


class FakeSub:
    """The minimal fork seam: .engine only -- NO score_tokens, so retokenization is unverifiable."""

    def __init__(self, engine=None):
        self.engine = engine if engine is not None else FakeEngine()
        self.steer = None
        self.memory = None


class ScoringSub(FakeSub):
    """Adds the score_tokens seam: 'retokenizes' any continuation text to a CANNED piece list, so a
    test chooses whether the boundary verified exact or shifted."""

    def __init__(self, pieces, engine=None):
        super().__init__(engine)
        self._pieces = list(pieces)
        self.score_calls = []

    def score_tokens(self, messages, continuation_ids=None, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.score_calls.append({"messages": messages, "continuation": continuation, "block": block})
        return [{"id": i, "piece": p, "logprob": -0.1} for i, p in enumerate(self._pieces)]


class FakeSteer:
    def __init__(self, vec, layer=7):
        self.vec = list(vec)
        self.layer = layer
        self.seen_strengths = None

    def steer_vector(self, strengths):
        self.seen_strengths = dict(strengths)
        return list(self.vec)


FINAL_PROMPT = "<|im_start|>user\nCount from 1 to 3.<|im_end|>\n<|im_start|>assistant\n"

RUN = {
    "id": "run_parent0", "model": "clozn-qwen", "substrate": "engine",
    "messages": [{"role": "user", "content": "Count from 1 to 3."}],
    "final_prompt": FINAL_PROMPT,
    "response": "One two three",
    "behavior": {"active_dials": {}},
    "trace": {
        "tokens": ["One", " two", " three"],
        "token_ids": [11, 22, 33],
        "confidence": [0.9, 0.8, 0.7],
        "alternatives": [
            [{"piece": "1", "token_id": 101, "prob": 0.05}],
            [{"piece": " 2", "token_id": 102, "prob": 0.04}],
            [{"piece": " 3", "token_id": 103, "prob": 0.03}],
        ],
    },
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return runlog


# =================================================================================== fork positions
def test_fork_mid_position_splices_prefix_and_forced_piece(store):
    sub = FakeSub()
    child = fork_mod.fork(RUN, sub, 1, token=" 2")
    assert child is not None
    call = sub.engine.calls[-1]
    assert call["prompt"] == FINAL_PROMPT + "One" + " 2"    # base + pieces[0..1) + forced piece
    assert child["response"] == "One" + " 2" + " and onward"
    assert child["parent_run_id"] == "run_parent0"
    assert child["source"] == "fork"
    assert child["client"] == "studio"
    # trace_provenance records WHY the child did or didn't get a spliced trace. Under a fake
    # substrate the traced seam is unavailable, so no continuation steps are captured and there is
    # nothing to explain -- None is the honest value here, not a missing key.
    assert child["changes_applied"] == {"fork": {"position": 1, "token": " 2",
                                                 "was_recorded_alternative": True,
                                                 "trace_provenance": None}}
    assert child["prefix_kept"] == "One"
    assert child["forked_from_piece"] == " two"
    assert "greedy" in child["note"]


def test_fork_position_zero_keeps_no_prefix(store):
    sub = FakeSub()
    child = fork_mod.fork(RUN, sub, 0, token="1")
    assert sub.engine.calls[-1]["prompt"] == FINAL_PROMPT + "1"
    assert child["prefix_kept"] == ""
    assert child["forked_from_piece"] == "One"
    assert child["response"] == "1" + " and onward"


def test_fork_last_position_keeps_all_but_last(store):
    sub = FakeSub()
    child = fork_mod.fork(RUN, sub, 2, token=" 3")
    assert sub.engine.calls[-1]["prompt"] == FINAL_PROMPT + "One two" + " 3"
    assert child["prefix_kept"] == "One two"
    assert child["forked_from_piece"] == " three"


def test_fork_decodes_greedy_deterministic(store):
    sub = FakeSub()
    fork_mod.fork(RUN, sub, 1, token=" 2")
    call = sub.engine.calls[-1]
    assert call["temperature"] == 0.0                      # a deterministic what-if, never a sample
    assert call["rep_penalty"] == 1.0
    assert call["seed"] == 0
    assert call["max_tokens"] == fork_mod.MAX_NEW


# =================================================================================== token honesty
def test_free_token_is_flagged_not_recorded(store):
    child = fork_mod.fork(RUN, FakeSub(), 1, token=" banana")
    assert child["changes_applied"]["fork"]["was_recorded_alternative"] is False
    assert child["changes_applied"]["fork"]["token"] == " banana"


def test_token_id_resolves_a_recorded_alternative(store):
    sub = FakeSub()
    child = fork_mod.fork(RUN, sub, 1, token_id=102)
    assert child["changes_applied"]["fork"]["token"] == " 2"
    assert child["changes_applied"]["fork"]["was_recorded_alternative"] is True
    assert sub.engine.calls[-1]["prompt"].endswith("One 2")


def test_token_id_of_the_committed_pick_is_not_an_alternative(store):
    child = fork_mod.fork(RUN, FakeSub(), 1, token_id=22)   # the committed token's own id
    assert child["changes_applied"]["fork"]["token"] == " two"
    assert child["changes_applied"]["fork"]["was_recorded_alternative"] is False


def test_unknown_token_id_raises(store):
    with pytest.raises(ValueError, match="not among the recorded alternatives"):
        fork_mod.fork(RUN, FakeSub(), 1, token_id=99999)


def test_token_text_wins_over_token_id(store):
    child = fork_mod.fork(RUN, FakeSub(), 1, token=" free", token_id=102)
    assert child["changes_applied"]["fork"]["token"] == " free"
    assert child["changes_applied"]["fork"]["was_recorded_alternative"] is False


def test_missing_and_empty_token_raise(store):
    with pytest.raises(ValueError, match="need a forced 'token'"):
        fork_mod.fork(RUN, FakeSub(), 1)
    with pytest.raises(ValueError, match="non-empty"):
        fork_mod.fork(RUN, FakeSub(), 1, token="")


# =================================================================================== validation
def test_position_out_of_range_raises(store):
    with pytest.raises(ValueError, match="out of range"):
        fork_mod.fork(RUN, FakeSub(), 3, token="x")
    with pytest.raises(ValueError, match="out of range"):
        fork_mod.fork(RUN, FakeSub(), -1, token="x")


def test_run_without_trace_raises(store):
    bare = {k: v for k, v in RUN.items() if k != "trace"}
    with pytest.raises(ValueError, match="no trace"):
        fork_mod.fork(bare, FakeSub(), 0, token="x")
    with pytest.raises(ValueError, match="no trace"):
        fork_mod.fork(dict(bare, trace={"tokens": []}), FakeSub(), 0, token="x")


# =================================================================================== retokenization flag
def test_retokenized_false_when_verified_exact(store):
    sub = ScoringSub(pieces=["One", " 2"])                  # score seam echoes the exact spliced pieces
    child = fork_mod.fork(RUN, sub, 1, token=" 2")
    assert child["retokenized"] is False
    assert sub.score_calls[-1]["continuation"] == "One 2"
    assert "could not be verified" not in child["note"]


def test_retokenized_true_when_boundary_shifts(store):
    sub = ScoringSub(pieces=["On", "e 2"])                  # the engine would merge across the splice
    child = fork_mod.fork(RUN, sub, 1, token=" 2")
    assert child["retokenized"] is True


def test_retokenized_true_when_unverifiable(store):
    child = fork_mod.fork(RUN, FakeSub(), 1, token=" 2")    # no score seam -> can't prove exact
    assert child["retokenized"] is True
    assert "could not be verified" in child["note"]


# =================================================================================== prompt source + dials
def test_falls_back_to_apply_template_when_no_final_prompt(store):
    run = {k: v for k, v in RUN.items() if k != "final_prompt"}
    sub = FakeSub(engine=FakeEngine(template="TPL<assistant>"))
    child = fork_mod.fork(run, sub, 1, token=" 2")
    assert sub.engine.calls[-1]["prompt"] == "TPL<assistant>" + "One" + " 2"
    assert child["prompt_source"] == "apply_template"


def test_no_final_prompt_and_no_template_fails_cleanly(store):
    run = {k: v for k, v in RUN.items() if k != "final_prompt"}
    assert fork_mod.fork(run, FakeSub(), 1, token=" 2") is None


def test_recorded_dials_ride_the_continuation(store):
    run = dict(RUN, behavior={"active_dials": {"warm": 0.5}})
    sub = FakeSub()
    sub.steer = FakeSteer(vec=[0.1, 0.2], layer=7)
    child = fork_mod.fork(run, sub, 1, token=" 2")
    call = sub.engine.calls[-1]
    assert call["steer_vec"] == [0.1, 0.2]                 # rebuilt from the RECORD, like score_tokens
    assert call["steer"] == {"coef": 1.0, "layer": 7}
    assert sub.steer.seen_strengths == {"warm": 0.5}
    assert child["behavior"]["active_dials"] == {"warm": 0.5}


# =================================================================================== child persistence
def test_child_is_persisted_and_linked(store):
    child = fork_mod.fork(RUN, FakeSub(), 1, token=" 2")
    fetched = store.get_run(child["id"])
    assert fetched is not None
    assert fetched["parent_run_id"] == "run_parent0"
    assert fetched["source"] == "fork"
    assert "replayed" in fetched["flags"]                  # runlog._flags sets this from parent_run_id
    assert fetched["final_prompt"] == FINAL_PROMPT + "One" + " 2"   # the exact spliced string
    assert fetched["finish_reason"] == "stop"
    # the response-only extensions are NOT persisted (same convention as replay's generated_ids)
    assert "prefix_kept" not in fetched


def test_generation_failure_returns_none(store):
    class BoomEngine(FakeEngine):
        def complete(self, prompt, **params):
            raise RuntimeError("engine exploded")
    assert fork_mod.fork(RUN, FakeSub(engine=BoomEngine()), 1, token=" 2") is None

    class GarbageEngine(FakeEngine):
        def complete(self, prompt, **params):
            return {"nope": True}
    assert fork_mod.fork(RUN, FakeSub(engine=GarbageEngine()), 1, token=" 2") is None


def test_substrate_without_engine_returns_none(store):
    class NoEngine:
        steer = None
        memory = None
    assert fork_mod.fork(RUN, NoEngine(), 1, token=" 2") is None


# ===================================================================================================
# the route: POST /runs/<id>/fork -- exercised directly (deliberately NOT registered in app.py yet)
# ===================================================================================================
import clozn.server.app as cs  # noqa: E402
import clozn.server.routes.fork as fork_routes  # noqa: E402


class FakeHandler:
    def __init__(self):
        self.code = None
        self.body = None

    def _json(self, code, obj, extra_headers=None):
        self.code = code
        self.body = obj


@pytest.fixture
def served(store, monkeypatch):
    """An isolated run store + a fake engine substrate installed as ctx.SUB; seeds a parent run
    through the REAL runlog.record (so the stored trace shape is the normalized on-disk one)."""
    sub = FakeSub()
    monkeypatch.setattr(cs, "SUB", sub)
    rid = runlog.record(source="openai_api", client="curl", model="clozn-qwen", substrate="engine",
                        messages=RUN["messages"], response=RUN["response"],
                        trace=RUN["trace"], final_prompt=FINAL_PROMPT)
    return sub, rid


def _post(path, body):
    h = FakeHandler()
    claimed = fork_routes.try_post(h, path, body)
    return claimed, h


def test_route_happy_path_returns_child(served):
    sub, rid = served
    claimed, h = _post(f"/runs/{rid}/fork", {"position": 1, "token": " 2"})
    assert claimed is True
    assert h.code == 200
    assert h.body["parent_run_id"] == rid
    assert h.body["prefix_kept"] == "One"
    assert h.body["forked_from_piece"] == " two"
    assert h.body["changes_applied"]["fork"] == {"position": 1, "token": " 2",
                                                 "was_recorded_alternative": True,
                                                 "trace_provenance": None}
    assert h.body["retokenized"] is True                   # FakeSub has no score seam -> flagged
    assert sub.engine.calls[-1]["prompt"] == FINAL_PROMPT + "One" + " 2"


def test_route_unknown_run_404(served):
    claimed, h = _post("/runs/run_nope/fork", {"position": 0, "token": "x"})
    assert claimed is True
    assert h.code == 404
    assert "not found" in h.body["error"]


def test_route_position_out_of_range_400(served):
    _, rid = served
    claimed, h = _post(f"/runs/{rid}/fork", {"position": 99, "token": "x"})
    assert claimed is True
    assert h.code == 400
    assert "out of range" in h.body["error"]


def test_route_missing_position_400(served):
    _, rid = served
    claimed, h = _post(f"/runs/{rid}/fork", {"token": "x"})
    assert claimed is True and h.code == 400
    assert "position" in h.body["error"]


def test_route_non_integer_position_400(served):
    _, rid = served
    claimed, h = _post(f"/runs/{rid}/fork", {"position": "abc", "token": "x"})
    assert claimed is True and h.code == 400


def test_route_missing_token_400(served):
    _, rid = served
    claimed, h = _post(f"/runs/{rid}/fork", {"position": 1})
    assert claimed is True and h.code == 400
    assert "token" in h.body["error"]


def test_route_no_substrate_503(served, monkeypatch):
    _, rid = served
    monkeypatch.setattr(cs, "SUB", None)
    claimed, h = _post(f"/runs/{rid}/fork", {"position": 1, "token": " 2"})
    assert claimed is True and h.code == 503
    assert "worker" in h.body["error"]


def test_route_substrate_without_engine_503(served, monkeypatch):
    _, rid = served
    class ChatOnly:
        engine = None
    monkeypatch.setattr(cs, "SUB", ChatOnly())
    claimed, h = _post(f"/runs/{rid}/fork", {"position": 1, "token": " 2"})
    assert claimed is True and h.code == 503


def test_route_generation_failure_500(served, monkeypatch):
    _, rid = served
    class BoomEngine(FakeEngine):
        def complete(self, prompt, **params):
            raise RuntimeError("engine exploded")
    monkeypatch.setattr(cs, "SUB", FakeSub(engine=BoomEngine()))
    claimed, h = _post(f"/runs/{rid}/fork", {"position": 1, "token": " 2"})
    assert claimed is True and h.code == 500
    assert "fork failed" in h.body["error"]


def test_route_ignores_other_paths(served):
    for path in ("/runs/run_x/branch", "/runs/run_x/replay", "/timetravel/mode", "/fork"):
        claimed, _h = _post(path, {"position": 0})
        assert claimed is False
