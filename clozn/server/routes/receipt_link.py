"""Ambient delivery, channel 1 (AMBIENT_DELIVERY.md):

  * GET  /r/<id>        -- the per-run permalink. Redirects a receipt-footer link to that Run in
                          Studio's hash-routed Inspect surface.
  * GET  /receipt/mode  -- is the in-band footer on? (server-wide default)
  * POST /receipt/mode  -- turn it on/off, persisted. A client that can't add a body field just points
                          its tool at clozn once, flips this, and every reply carries the receipt link.

The footer itself is appended in routes/openai.py (non-stream) + sse.py (stream); see
clozn/runs/receipt_footer.py for its shape + honesty rules.
"""
import clozn.settings as settings
from clozn.server.static import APP_INDEX

RECEIPT_MODE_SETTING = "receipt_footer_mode"
RECEIPT_MODES = ("off", "exceptions", "always")


def receipt_mode() -> str:
    """Return the persisted footer mode."""
    configured = settings.get_setting(RECEIPT_MODE_SETTING, "off")
    return configured if configured in RECEIPT_MODES else "off"


def request_receipt_mode(body: dict) -> str:
    """Resolve one normalized request's footer mode without changing persisted settings.

    A per-request mode is narrow and explicit.
    """
    configured = receipt_mode()
    requested_mode = body.get("clozn_receipt_mode") if isinstance(body, dict) else None
    if requested_mode in RECEIPT_MODES:
        return requested_mode
    return configured


def _safe_run_id(raw: str) -> str:
    """Run-id charset only -- the value rides a Location header + a URL fragment, so strip anything that
    could inject a header or escape the fragment (run ids are like run_0019f...; keep [A-Za-z0-9_-]).
    This charset is also exactly what app.mjs's route regex accepts, so a sanitized id either matches
    the Lens route or is empty."""
    return "".join(c for c in (raw or "") if c.isalnum() or c in "_-")


def try_get(h, p):
    if p == "/receipt/mode":
        h._json(200, {"receipt_footer_mode": receipt_mode(), "modes": list(RECEIPT_MODES)})
        return True
    if p.startswith("/r/"):
        rid = _safe_run_id(p[len("/r/"):].strip("/"))
        if not rid:
            h._send(404, "no run id", "text/plain; charset=utf-8")
            return True
        h._send(302, "", "text/plain; charset=utf-8", {"Location": APP_INDEX + "#/inspect/" + rid})
        return True
    return False


def try_post(h, p, body):
    if p == "/receipt/mode":
        body = body if isinstance(body, dict) else {}
        changed = False
        requested_mode = body.get("receipt_footer_mode")
        if requested_mode is not None:
            if requested_mode not in RECEIPT_MODES:
                h._json(400, {"error": "receipt_footer_mode must be off, exceptions, or always"})
                return True
            settings.set_setting(RECEIPT_MODE_SETTING, requested_mode)
            changed = True
        h._json(200, {"receipt_footer_mode": receipt_mode(), "changed": changed})
        return True
    return False
