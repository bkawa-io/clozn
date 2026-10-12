"""Context Receipt reads and metadata-disclosure settings.

The receipt endpoint returns only the schema-governed metadata artifact. Prompt and message content
belongs to the run resource and is never conditionally embedded in this response.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_SUFFIX = "/context-receipt"


def try_get(h, p):
    if p.startswith("/runs/") and p.endswith(_SUFFIX):
        import clozn.runs.store as runlog
        from clozn.runs.context_receipt import read_receipt

        rid = p[len("/runs/"):-len(_SUFFIX)]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True

        view = read_receipt(run)
        if view["shape"] == "absent":
            h._json(200, {"run_id": rid, "shape": "absent", "context_receipt": {},
                          "note": "no context receipt was recorded for this run"})
            return True

        h._json(200, {
            "run_id": rid,
            "shape": view["shape"],
            "context_receipt": view["receipt"],
        })
        return True

    if p == "/receipt-privacy":
        from clozn.runs import receipt_privacy
        h._json(200, {"tier": receipt_privacy.tier(), "tiers": list(receipt_privacy.TIERS)})
        return True

    return False


def try_post(h, p, body):
    if p == "/receipt-privacy":
        from clozn.runs import receipt_privacy
        body = body if isinstance(body, dict) else {}
        action = str(body.get("action") or "").strip().lower()
        try:
            if action == "undo":
                mutation = receipt_privacy.undo_tier(body.get("transaction_id"))
                h._json(200, {
                    "ok": True,
                    "tier": receipt_privacy.tier(),
                    "mutation": mutation,
                })
                return True

            name = body.get("tier")
            if action == "preview":
                h._json(200, {
                    "ok": True,
                    "tier": str(name or "").strip().lower(),
                    "preview": receipt_privacy.preview_tier(name),
                })
                return True

            if action == "apply":
                expected = body.get("expected")
                if not isinstance(expected, dict) or not isinstance(expected.get("exists"), bool):
                    raise receipt_privacy.PrivacyMutationError(
                        "apply requires expected.exists from a fresh preview"
                    )
                mutation = receipt_privacy.apply_tier(
                    name,
                    expected_exists=expected["exists"],
                    expected_sha256=expected.get("sha256"),
                )
            else:
                raise receipt_privacy.PrivacyMutationError(
                    "action must be preview, apply, or undo"
                )
        except receipt_privacy.PrivacyDriftError as exc:
            h._json(409, {"ok": False, "error": str(exc), "code": "settings_drift"})
            return True
        except receipt_privacy.PrivacyMutationError as exc:
            h._json(400, {"ok": False, "error": str(exc)})
            return True
        except (OSError, ValueError) as exc:
            h._json(500, {"ok": False, "error": f"could not persist the tier setting: {exc}"})
            return True
        h._json(200, {
            "ok": True,
            "tier": str(name or "").strip().lower(),
            "mutation": mutation,
        })
        return True
    return False
