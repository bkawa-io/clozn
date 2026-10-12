"""GET/POST /runs/<id>/context-receipt and /receipt-privacy (clozn/server/routes/context_receipt.py),
plus proof the CLOZN_ROUTE_AUTOLOAD registration actually reaches this module ahead of the generic
GET /runs/<id> fallback (docs/SEAMS.md Seam 4)."""
from __future__ import annotations

import io
import json

import clozn.runs.store as runlog
from clozn.server.routes import context_receipt as route
from clozn.runs.context_receipt import build_context_receipt

class Handler:
    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _new_run(rid="run_x"):
    receipt = build_context_receipt(
        messages=[{"role": "system", "content": "rule"}, {"role": "user", "content": "hi"}],
        assembled_messages=[{"role": "system", "content": "rule"}, {"role": "user", "content": "hi"}],
        final_prompt="EXACT PROMPT", run_id=rid,
    )
    return {"id": rid, "context_receipt": receipt}


# ---------------------------------------------------------------------------------------------------
# GET /runs/<id>/context-receipt
# ---------------------------------------------------------------------------------------------------

def test_run_not_found_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: None)
    h = Handler("/runs/run_x/context-receipt")
    assert route.try_get(h, "/runs/run_x/context-receipt") is True
    assert h.status == 404


def test_absent_receipt_is_a_clean_200_not_a_fabricated_document(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"id": rid})
    h = Handler("/runs/run_x/context-receipt")
    assert route.try_get(h, "/runs/run_x/context-receipt") is True
    assert h.status == 200
    assert h.body["shape"] == "absent"
    assert h.body["context_receipt"] == {}


def test_recorded_shape_serves_metadata_without_run_content(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: _new_run(rid))
    h = Handler("/runs/run_x/context-receipt")
    assert route.try_get(h, "/runs/run_x/context-receipt") is True
    assert h.body["shape"] == "recorded"
    receipt = h.body["context_receipt"]
    assert "survived" not in receipt
    assert "EXACT PROMPT" not in json.dumps(receipt)
    assert receipt["delivered"][0]["source_label"] == "system"


def test_unrelated_path_is_not_claimed():
    h = Handler("/unrelated")
    assert route.try_get(h, "/unrelated") is False
    assert route.try_post(h, "/unrelated", {}) is False


# ---------------------------------------------------------------------------------------------------
# GET/POST /receipt-privacy
# ---------------------------------------------------------------------------------------------------

def test_receipt_privacy_get_default_and_explicit_apply_sets_it(tmp_path, monkeypatch):
    import clozn.settings as settings
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "studio_settings.json"))

    h = Handler("/receipt-privacy")
    assert route.try_get(h, "/receipt-privacy") is True
    assert h.body == {"tier": "full", "tiers": ["full", "metadata_only", "hashes_only", "off"]}

    previewed = Handler("/receipt-privacy")
    assert route.try_post(previewed, "/receipt-privacy", {
        "action": "preview", "tier": "hashes_only",
    }) is True
    posted = Handler("/receipt-privacy")
    assert route.try_post(posted, "/receipt-privacy", {
        "action": "apply", "tier": "hashes_only",
        "expected": previewed.body["preview"]["expected"],
    }) is True
    assert posted.body["ok"] is True
    assert posted.body["tier"] == "hashes_only"
    assert posted.body["mutation"]["status"] == "applied"
    after = Handler("/receipt-privacy")
    route.try_get(after, "/receipt-privacy")
    assert after.body["tier"] == "hashes_only"


def test_receipt_privacy_post_rejects_unknown_tier(tmp_path, monkeypatch):
    import clozn.settings as settings
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "studio_settings.json"))
    h = Handler("/receipt-privacy")
    assert route.try_post(h, "/receipt-privacy", {
        "action": "preview", "tier": "telepathy",
    }) is True
    assert h.status == 400


def test_receipt_privacy_explicit_preview_apply_and_undo(tmp_path, monkeypatch):
    import clozn.settings as settings
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "studio_settings.json"))

    previewed = Handler("/receipt-privacy")
    assert route.try_post(previewed, "/receipt-privacy", {
        "action": "preview", "tier": "metadata_only",
    }) is True
    assert previewed.status == 200
    preview = previewed.body["preview"]
    assert preview["current"] == {"exists": False, "tier": "full"}
    assert preview["expected"] == {"exists": False}
    assert preview["target"] == settings.SETTINGS_PATH

    applied = Handler("/receipt-privacy")
    assert route.try_post(applied, "/receipt-privacy", {
        "action": "apply",
        "tier": "metadata_only",
        "expected": preview["expected"],
    }) is True
    assert applied.status == 200
    assert applied.body["mutation"]["status"] == "applied"
    transaction_id = applied.body["mutation"]["transaction_id"]

    undone = Handler("/receipt-privacy")
    assert route.try_post(undone, "/receipt-privacy", {
        "action": "undo", "transaction_id": transaction_id,
    }) is True
    assert undone.status == 200
    assert undone.body["mutation"]["status"] == "removed"
    assert undone.body["tier"] == "full"

    repeated = Handler("/receipt-privacy")
    route.try_post(repeated, "/receipt-privacy", {
        "action": "undo", "transaction_id": transaction_id,
    })
    assert repeated.body["mutation"]["status"] == "already_undone"


def test_receipt_privacy_explicit_apply_requires_expected_and_refuses_drift(tmp_path, monkeypatch):
    import clozn.settings as settings
    target = tmp_path / "studio_settings.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(target))

    missing = Handler("/receipt-privacy")
    route.try_post(missing, "/receipt-privacy", {
        "action": "apply", "tier": "off",
    })
    assert missing.status == 400
    assert "expected.exists" in missing.body["error"]

    previewed = Handler("/receipt-privacy")
    route.try_post(previewed, "/receipt-privacy", {
        "action": "preview", "tier": "off",
    })
    target.write_text('{"external": true}', encoding="utf-8")
    drifted = Handler("/receipt-privacy")
    route.try_post(drifted, "/receipt-privacy", {
        "action": "apply",
        "tier": "off",
        "expected": previewed.body["preview"]["expected"],
    })
    assert drifted.status == 409
    assert drifted.body["code"] == "settings_drift"
    assert json.loads(target.read_text(encoding="utf-8")) == {"external": True}


# ---------------------------------------------------------------------------------------------------
# autoload wiring: proves this module is actually discovered and dispatched, not just importable
# ---------------------------------------------------------------------------------------------------

def test_module_is_registered_before_the_runs_fallback():
    from clozn.server import app as cs
    assert route in cs._GET_ROUTES
    assert route in cs._POST_ROUTES
    assert cs._GET_ROUTES.index(route) < cs._GET_ROUTES.index(cs._runs_fallback_routes)


def _dispatch(method, path, body_obj=None):
    from clozn.server import app as cs
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


def test_end_to_end_http_get_reaches_this_route_not_the_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rid = runlog.record(source="cli", client="cli", messages=[{"role": "user", "content": "hi"}],
                        response="ok", started=1.0)
    out = _dispatch("GET", f"/runs/{rid}/context-receipt")
    assert out["run_id"] == rid
    assert out["shape"] == "recorded"
    assert isinstance(out["context_receipt"]["delivered"], list)
