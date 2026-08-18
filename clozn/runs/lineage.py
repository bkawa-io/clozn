"""Parent/child lineage and run-family lookup."""
from __future__ import annotations

from . import store


def _load_runs() -> list[dict]:
    """Load full records from the authoritative SQLite journal."""
    return [run for run in store.iter_runs() if isinstance(run, dict) and run.get("id")]


def _change_label(run: dict) -> str | None:
    """Short lineage label for known replay/branch change specs."""
    changes = run.get("changes_applied") or {}
    if not isinstance(changes, dict) or not changes:
        return None
    if isinstance(changes.get("label"), str) and changes["label"].strip():
        return changes["label"].strip()

    parts = []

    corrective = changes.get("corrective_retry")
    if isinstance(corrective, dict) and corrective.get("preset"):
        arm = str(corrective.get("arm") or "candidate")
        parts.append(f"retry {corrective['preset']} ({arm})")

    if changes.get("branch_turn") is not None:
        branch = f"branched from turn {changes.get('branch_turn')}"
        if changes.get("edited_user"):
            branch += " (edited question)"
        if changes.get("kv_snapshot"):
            branch += " + KV snapshot"
        parts.append(branch)

    if changes.get("plain"):
        parts.append("re-roll")
    if parts:
        return ", ".join(parts)

    keys = [str(k) for k in sorted(changes)[:3]]
    return "changed " + ", ".join(keys) if keys else None


def _lineage_summary(run: dict, current_id: str | None = None) -> dict:
    timing = run.get("timing") or {}
    return {
        "id": run.get("id"),
        "parent_run_id": run.get("parent_run_id"),
        "created_at": run.get("created_at"),
        "created_ts": run.get("created_ts"),
        "source": run.get("source"),
        "client": run.get("client"),
        "model": run.get("model"),
        "substrate": run.get("substrate"),
        "prompt_summary": run.get("prompt_summary"),
        "response_summary": run.get("response_summary"),
        "finish_reason": run.get("finish_reason"),
        "duration_ms": timing.get("duration_ms"),
        "changes_applied": run.get("changes_applied") or {},
        "change_label": _change_label(run),
        "flags": run.get("flags") or [],
        "is_current": run.get("id") == current_id,
    }


def lineage(rid: str, limit: int = 500) -> dict | None:
    """Return ancestors, siblings, children, and a simple descendant tree for a run."""
    runs = _load_runs()
    by_id = {r.get("id"): r for r in runs if r.get("id")}
    current = by_id.get(rid)
    if not current:
        return None

    children_by_parent: dict[str, list[dict]] = {}
    for r in runs:
        parent = r.get("parent_run_id")
        if parent:
            children_by_parent.setdefault(parent, []).append(r)
    for children in children_by_parent.values():
        children.sort(key=lambda r: (r.get("created_ts") or 0, r.get("id") or ""))

    ancestors = []
    seen = {rid}
    parent_id = current.get("parent_run_id")
    while parent_id and parent_id in by_id and parent_id not in seen and len(ancestors) < limit:
        parent = by_id[parent_id]
        ancestors.append(parent)
        seen.add(parent_id)
        parent_id = parent.get("parent_run_id")
    ancestors.reverse()

    root = ancestors[0] if ancestors else current
    tree_count = 0

    def build_tree(run: dict, seen_tree: set[str]) -> dict:
        nonlocal tree_count
        tree_count += 1
        node = _lineage_summary(run, rid)
        node_id = run.get("id")
        kids = []
        if tree_count < limit and node_id not in seen_tree:
            next_seen = set(seen_tree)
            if node_id:
                next_seen.add(node_id)
            for child in children_by_parent.get(node_id, []):
                if tree_count >= limit:
                    break
                if child.get("id") in next_seen:
                    continue
                kids.append(build_tree(child, next_seen))
        node["children"] = kids
        return node

    parent_for_siblings = current.get("parent_run_id")
    siblings = []
    if parent_for_siblings:
        siblings = [r for r in children_by_parent.get(parent_for_siblings, []) if r.get("id") != rid]

    return {
        "run_id": rid,
        "root_id": root.get("id"),
        "original": _lineage_summary(root, rid),
        "current": _lineage_summary(current, rid),
        "ancestors": [_lineage_summary(r, rid) for r in ancestors],
        "children": [_lineage_summary(r, rid) for r in children_by_parent.get(rid, [])],
        "siblings": [_lineage_summary(r, rid) for r in siblings],
        "tree": build_tree(root, set()),
    }


def lineage_family(rid: str, limit: int = 2000) -> list[dict] | None:
    """Return the whole branch family of `rid` as GET /runs-shaped summaries."""
    runs = _load_runs()
    by_id = {r.get("id"): r for r in runs if r.get("id")}
    if rid not in by_id:
        return None
    children_by_parent: dict[str, list[str]] = {}
    for r in runs:
        parent, cid = r.get("parent_run_id"), r.get("id")
        if parent and cid:
            children_by_parent.setdefault(parent, []).append(cid)
    seen: set[str] = set()
    stack = [rid]
    while stack and len(seen) < limit:
        cur = stack.pop()
        if cur in seen or cur not in by_id:
            continue
        seen.add(cur)
        parent = by_id[cur].get("parent_run_id")
        if parent and parent in by_id and parent not in seen:
            stack.append(parent)
        for child in children_by_parent.get(cur, []):
            if child not in seen:
                stack.append(child)
    fam = [by_id[i] for i in seen]
    fam.sort(key=lambda r: (r.get("created_ts") or 0, r.get("id") or ""), reverse=True)
    return [store._summary(r) for r in fam]
