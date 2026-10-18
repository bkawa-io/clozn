"""Cancellation proxy for the selected private model worker."""
from clozn.server import app as ctx


CLOZN_ROUTE_AUTOLOAD = True


def try_post(h, p, body):
    if p != "/cancel":
        return False

    from clozn.server.model_routing import select_control_model_for_request

    selection = select_control_model_for_request(
        h,
        body.get("model"),
        field_present="model" in body,
        route="/cancel",
    )
    if selection is None:
        return True
    engine = selection.engine
    sub = selection.sub
    if engine is None or not callable(getattr(engine, "cancel", None)):
        h._json(503, {"error": "selected model worker is unavailable"})
        return True

    engine_req = str(body.get("req") or "")
    request_id = str(body.get("req_id") or "")
    if request_id and not engine_req:
        current = getattr(sub, "_request", None) if sub is not None else None
        if current is not None and current.request_id == request_id:
            current.cancel()
            engine_req = current.engine_req or ""
        if not engine_req:
            h._json(200, {"cancelled": False, "req": request_id})
            return True
    try:
        result = engine.cancel(engine_req)
    except Exception as error:
        h._json(502, {"error": f"engine cancel failed: {error}"})
        return True
    h._json(200, result)
    return True
