"""Experiment-result review services over authoritative local clozn.experiment.result.v0 artifacts.

The list/detail/cell/trend endpoints are read-only. Promotion is the one write service and is deliberately
separate from result storage: preview is pure; apply writes only an explicitly named regression-suite
artifact under ~/.clozn/regression-suites after expected-hash and redaction-review checks. CI preview
returns copyable text and never writes a repository workflow.

This is the backend half of notes/agent_roadmap/04-studio-experiments-workspace.md ("Studio reads and
composes existing experiment/run artifacts" -- no new experiment engine or scorer lives here). Three
roadmap features read this family: Studio's Experiments workspace, the GitHub Action model gate's verify
mode, and automatic regression triage. All three should validate what they read with
clozn.schemas.validate(doc, "clozn.experiment.result.v0") rather than trusting shape by convention.

WHY /experiment-results, NOT /experiments/<id>
------------------------------------------------
GET /experiments/types already exists (clozn/server/routes/receipts.py) as the capability catalog for
the UNRELATED single-run "change one thing" drawer primitive (clozn.experiments.experiment -- swap a
concept, edit a turn, re-roll, ...). Nesting case x variant x seed suite results under /experiments/<id>
would share a path prefix with that catalog for no reason beyond both modules containing the English
word "experiment". A distinct top-level segment avoids the collision outright instead of relying on a
real experiment_id never happening to equal "types".

TWO SIZES OF RESPONSE, ON PURPOSE
------------------------------------------------
A stored result embeds a full run record (response text, receipts, meta) per cell -- for a suite with
hundreds of cases x several seeds that is a lot of payload a matrix/summary screen does not need before
someone opens one cell (see the spec's Performance section: "Initial screen should load summaries
without downloading complete receipts... fetch a receipt only when the cell is opened"). So:

    GET /experiment-results                    thin: identity + per-variant aggregates, no cells
    GET /experiment-results/<id>                manifest + summary + THIN cells (coordinates + status
                                                 only -- no response/receipts/run)
    GET /experiment-results/<id>/cells          FULL matching cells (response/receipts/run included),
                                                 filtered by ?suite=&case=&variant=&seed= so a client
                                                 fetches exactly the cell it opened
NO CACHING (v1, DELIBERATE)
------------------------------------------------
Every request under this family re-reads and re-validates every *.json file in the results directory
(clozn.experiments.suite.load_result, which runs the full case x variant x seed completeness check, not
just this module's shape check). For a local, single-user directory of experiment artifacts this is
fast enough; it is also the honest baseline to optimize from later rather than a cache invalidation bug
to debug later. A file that fails to load is reported in the "broken" list of GET /experiment-results,
never silently dropped -- a corrupt local artifact should be visibly broken, not invisible.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

CLOZN_ROUTE_AUTOLOAD = True

_PREFIX = "/experiment-results"

# Cell fields cheap enough for the list/detail screens: coordinates, status, and the small evaluation
# facts, but never response text, receipts, or the embedded run record -- those are the "open this cell"
# payload, fetched only via GET /experiment-results/<id>/cells.
_THIN_CELL_KEYS = ("suite", "case", "variant", "variant_kind", "seed", "status", "run_id",
                   "assertions", "min_confidence", "error")


def _one(query: dict, name: str):
    values = query.get(name) or []
    return values[-1] if values else None


def _thin_cell(cell: dict) -> dict:
    return {key: cell.get(key) for key in _THIN_CELL_KEYS}


def _list_entry(result: dict) -> dict:
    from clozn.experiments import suite
    manifest = result.get("manifest") or {}
    summary = result.get("summary") or {}
    entry = {
        "experiment_id": result.get("experiment_id"),
        "name": result.get("name"),
        "created_at": result.get("created_at"),
        "baseline_variant": summary.get("baseline_variant"),
        "variants": [v.get("name") for v in manifest.get("variants") or [] if isinstance(v, dict)],
        "seeds": result.get("seeds"),
        "cell_count": len(result.get("cells") or []),
        "aggregates": summary.get("aggregates"),
        "suite_fingerprint": suite.result_fingerprint(result),
    }
    for field in ("vcs", "artifact_provenance"):
        if field in result:
            entry[field] = result[field]
    return entry


def _load_all():
    """(entries, broken): entries are (path, loaded_result) for every file that parsed and passed
    clozn.experiments.suite.load_result's full validation; broken are (path, error message) for every
    file that did not. See the module docstring on why a broken file is reported, not dropped."""
    from clozn.experiments import suite
    entries, broken = [], []
    for path in suite.list_result_paths():
        try:
            entries.append((path, suite.load_result(path)))
        except suite.ManifestError as exc:
            broken.append((path, str(exc)))
    return entries, broken


def _find(experiment_id: str):
    """The loaded result for `experiment_id`, or None. Reads every candidate file's own experiment_id
    field rather than assuming the filename is `<experiment_id>.json` -- true for
    suite.default_result_path's own naming, not guaranteed for a result saved with `--out`."""
    entries, _broken = _load_all()
    for _path, result in entries:
        if result.get("experiment_id") == experiment_id:
            return result
    return None


def _not_found(h, experiment_id: str) -> None:
    h._json(404, {"error": f"no experiment result found for id {experiment_id!r}"})


def try_get(h, p):
    if not p.startswith(_PREFIX):
        return False

    if p == _PREFIX + "/trends":
        from clozn.experiments import history, suite
        query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
        digest = _one(query, "fingerprint")
        if digest is not None and (
                len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            h._json(400, {"error": "fingerprint must be a lowercase SHA-256 digest"})
            return True
        entries, broken = _load_all()
        index = history.build_trend_index(result for _path, result in entries)
        payload = (
            history.select_compatible(
                index, {"algorithm": suite.FINGERPRINT_ALGORITHM, "sha256": digest})
            if digest is not None else index
        )
        payload["broken"] = [{"path": path, "error": error} for path, error in broken]
        h._json(200, payload)
        return True

    if p == _PREFIX:
        query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
        try:
            raw_limit, raw_offset = _one(query, "limit"), _one(query, "offset")
            limit = 50 if raw_limit is None else int(raw_limit)
            offset = 0 if raw_offset is None else int(raw_offset)
            if not 1 <= limit <= 500:
                raise ValueError("limit must be between 1 and 500")
            if offset < 0:
                raise ValueError("offset must be 0 or greater")
        except ValueError as exc:
            h._json(400, {"error": str(exc)})
            return True
        entries, broken = _load_all()
        # Newest first. created_at is a zero-padded ISO-8601 UTC string (suite.run_manifest's own
        # strftime), so lexicographic order is chronological order; missing/malformed values sort last.
        entries.sort(key=lambda pair: pair[1].get("created_at") or "", reverse=True)
        page = entries[offset:offset + limit]
        h._json(200, {
            "experiments": [_list_entry(result) for _path, result in page],
            "total": len(entries),
            "limit": limit,
            "offset": offset,
            "broken": [{"path": path, "error": error} for path, error in broken],
        })
        return True

    rest = p[len(_PREFIX) + 1:]   # everything after "/experiment-results/"

    if rest.endswith("/cells"):
        experiment_id = rest[:-len("/cells")]
        result = _find(experiment_id)
        if result is None:
            _not_found(h, experiment_id)
            return True
        from clozn.experiments import suite
        query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
        seed_raw = _one(query, "seed")
        try:
            seed = None if seed_raw is None else int(seed_raw)
        except ValueError:
            h._json(400, {"error": "seed must be an integer"})
            return True
        cells = suite.select_cells(result, suite=_one(query, "suite"), case=_one(query, "case"),
                                   variant=_one(query, "variant"), seed=seed)
        h._json(200, {"experiment_id": experiment_id, "cells": cells})
        return True

    if rest.endswith("/trends"):
        experiment_id = rest[:-len("/trends")]
        result = _find(experiment_id)
        if result is None:
            _not_found(h, experiment_id)
            return True
        from clozn.experiments import history, suite
        entries, broken = _load_all()
        index = history.build_trend_index(item for _path, item in entries)
        payload = history.select_compatible(index, suite.result_fingerprint(result))
        payload["experiment_id"] = experiment_id
        payload["broken"] = [{"path": path, "error": error} for path, error in broken]
        h._json(200, payload)
        return True

    if rest and "/" not in rest:   # the bare GET /experiment-results/<id> detail route
        experiment_id = rest
        result = _find(experiment_id)
        if result is None:
            _not_found(h, experiment_id)
            return True
        thin = dict(result)
        thin["cells"] = [_thin_cell(cell) for cell in result.get("cells") or []]
        h._json(200, thin)
        return True

    return False


def try_post(h, p, body):
    if not p.startswith(_PREFIX + "/"):
        return False
    rest = p[len(_PREFIX) + 1:]

    if rest.endswith("/promotion-preview") or rest.endswith("/promotion-apply"):
        apply = rest.endswith("/promotion-apply")
        suffix = "/promotion-apply" if apply else "/promotion-preview"
        experiment_id = rest[:-len(suffix)]
        result = _find(experiment_id)
        if result is None:
            _not_found(h, experiment_id)
            return True
        from clozn.experiments import promotion
        try:
            destination = promotion.resolve_destination(body.get("destination"))
            payload = (
                promotion.apply_promotion(result, destination, body)
                if apply else promotion.preview_promotion(result, destination, body)
            )
        except promotion.DestinationDriftError as exc:
            h._json(409, {"error": str(exc)})
            return True
        except promotion.PromotionServiceError as exc:
            h._json(400, {"error": str(exc)})
            return True
        h._json(200, payload)
        return True

    if rest.endswith("/ci-preview"):
        experiment_id = rest[:-len("/ci-preview")]
        result = _find(experiment_id)
        if result is None:
            _not_found(h, experiment_id)
            return True
        from clozn.experiments import action_contract
        try:
            payload = action_contract.ci_preview(result, body)
        except action_contract.ActionContractError as exc:
            h._json(400, {"error": str(exc)})
            return True
        h._json(200, payload)
        return True

    return False
