from clozn.runs.diagnosis import SCHEMA, diagnose


def _by_id(report):
    return {item["id"]: item for item in report["why_slow"]["findings"]}


def test_diagnosis_reports_only_recorded_phase_context_cutoff_and_related_calls():
    run = {
        "id": "run_main", "session_key": "session_exact", "finish_reason": "length",
        "timing": {"started_at": 100.0, "ended_at": 102.4, "duration_ms": 2400,
                   "load_duration_ms": 100, "prefill_duration_ms": 400,
                   "kv_allocation_ms": 20},
        "meta": {"device": "cuda", "cpu_spill_bytes": 0, "generation_duration_ms": 1500},
        "trace": {"steps": [{"dt_ms": 10}, {"dt_ms": 20}]},
        "context_receipt": {
            "rendered": {"tokens": 90}, "context_window_tokens": 128,
            "reserved_output_tokens": 38,
            "termination": {"reason": "context_limit", "generated_tokens": 38},
        },
    }
    related = [
        {"id": "run_aux", "session_key": "session_exact", "source": "openai_api",
         "prompt_summary": "title", "timing": {"started_at": 102.5, "ended_at": 102.7,
                                                  "duration_ms": 200}},
        {"id": "run_other", "session_key": "different", "source": "openai_api",
         "timing": {"started_at": 101.0, "ended_at": 101.1}},
    ]

    report = diagnose(run, related)
    findings = _by_id(report)

    assert report["schema"] == SCHEMA
    assert findings["model_load"]["status"] == "observed"
    assert findings["prefill"]["status"] == "observed"
    assert findings["generation"]["status"] == "observed"
    assert findings["generation"]["evidence"][0]["path"] == "meta.generation_duration_ms"
    assert findings["context_allocation"]["status"] == "observed"
    assert findings["cpu_spill"]["status"] == "not_observed"
    assert report["why_cut_off"]["finding"]["status"] == "observed"
    assert "both" in report["why_cut_off"]["summary"]
    assert report["client_auxiliary_calls"]["status"] == "observed"
    assert report["client_auxiliary_calls"]["evidence"][-1]["value"][0]["run_id"] == "run_aux"


def test_sparse_run_marks_missing_evidence_unavailable_instead_of_guessing():
    report = diagnose({"id": "run_sparse", "timing": {}, "meta": {"device": "cpu"}})
    findings = _by_id(report)
    for name in ("total_wall_time", "model_load", "prefill", "generation",
                 "context_pressure", "context_allocation", "cpu_spill"):
        assert findings[name]["status"] == "unavailable"
    assert report["why_cut_off"]["finding"]["status"] == "unavailable"
    assert report["client_auxiliary_calls"]["status"] == "unavailable"
    assert "cannot prove" in findings["cpu_spill"]["text"]


def test_client_disconnect_is_reported_as_cutoff_without_normal_stop_claim():
    report = diagnose({"id": "run_disconnect", "error": "client_disconnected",
                       "finish_reason": None, "client_key": "client_exact",
                       "timing": {"started_at": 10.0, "ended_at": 11.0, "duration_ms": 1000}}, [])
    finding = report["why_cut_off"]["finding"]
    assert finding["status"] == "observed"
    assert "disconnected" in finding["text"]
    assert report["client_auxiliary_calls"]["status"] == "not_observed"
