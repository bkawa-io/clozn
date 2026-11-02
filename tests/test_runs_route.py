from __future__ import annotations

from clozn.server.routes import runs as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status = status
        self.body = body


def test_run_list_adds_model_free_analysis_coverage(monkeypatch):
    summary = {"id": "run_a", "flags": []}
    run = {"id": "run_a"}
    monkeypatch.setattr("clozn.runs.store.list_runs", lambda _limit: [summary])
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == "run_a" else None)
    monkeypatch.setattr(
        "clozn.recipes.context_effects.plan_context_effects_reader",
        lambda value, **_kwargs: {"run_id": value["id"]},
    )
    monkeypatch.setattr(
        "clozn.experiments.context_investigation.summarize_context_investigation",
        lambda *_args, **_kwargs: {
            "status": "partial",
            "source_count": 8,
            "completed_source_count": 3,
        },
    )

    handler = Handler()
    assert route.try_get(handler, "/runs") is True

    assert handler.status == 200
    assert handler.body == {"runs": [{
        "id": "run_a",
        "flags": [],
        "analysis_status": "partial",
        "analysis_scoring_mode": "fast",
        "analysis_source_count": 8,
        "analysis_completed_source_count": 3,
    }]}


def test_run_list_does_not_claim_run_subroutes():
    handler = Handler()
    assert route.try_get(handler, "/runs/run_a") is False


def test_run_list_links_to_the_mode_with_saved_evidence(monkeypatch):
    monkeypatch.setattr("clozn.runs.store.list_runs", lambda _limit: [{"id": "run_a"}])
    monkeypatch.setattr("clozn.runs.store.get_run", lambda _id: {"id": "run_a"})
    monkeypatch.setattr(
        "clozn.recipes.context_effects.plan_context_effects_reader",
        lambda _run, *, scoring_mode: scoring_mode,
    )
    for saved_mode in ("fast", "exact"):
        monkeypatch.setattr(
            "clozn.experiments.context_investigation.summarize_context_investigation",
            lambda _run, mode, **_kwargs: {
                "status": "completed" if mode == saved_mode else "not_measured",
                "source_count": 8,
                "completed_source_count": 8 if mode == saved_mode else 0,
            },
        )
        handler = Handler()
        route.try_get(handler, "/runs")
        row = handler.body["runs"][0]
        assert row["analysis_status"] == "completed"
        assert row["analysis_scoring_mode"] == saved_mode
