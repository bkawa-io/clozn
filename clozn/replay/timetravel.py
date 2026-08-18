"""timetravel -- the time-travel debugger: per-turn KV snapshots + rewind/branch, recorded as child runs.

The offline rig proved the load-bearing mechanism:
a transformer's `past_key_values`, treated as first-class addressable STATE, is byte-exact
checkpoint/branch-able (Phase 1: a branch from a kept cache == a fresh full recompute, token-for-token at
depth 2/5/10) and CPU-offloadable and nearly free (a branch re-prefills a CONSTANT ~27 tokens vs 883 at
depth 10). FINDINGS.md #3: "state is perfectly snapshottable, just not writable-once" -- so this ships the
snapshottable half (checkpoint + rewind + branch) and SKIPS state-surgery in v1 (the half-life<1-turn null;
Lab only).

This module is the product spine for that, split so the BOOKKEEPING is model-free-testable (the whole
suite stays green without a GPU) while the torch-dependent snapshot capture is optional and degrades
cleanly:

  * SnapshotStore -- a BOUNDED, CPU-offloaded ring of per-turn KV snapshots, keyed by run id. The cap is
    honest (last N turns, configurable) and the byte accounting is real: `nbytes` per snapshot, a running
    total, evict-oldest when over the count OR byte budget. Snapshots hold an OPAQUE payload (the caller
    hands us kv tensors already cloned to CPU, or -- on the stateless studio chat path, which produces no
    reusable cache -- a lightweight descriptor with n_tok only), so the store's ring/eviction/accounting
    logic is pure Python and unit-tested with fabricated payloads.
  * branch_messages -- the pure "rewind & branch from here" transcript transform: truncate a run's messages
    at turn t, optionally splice an ALTERNATE user message for that turn. Model-free.
  * branch -- re-generate the branched transcript on the live substrate and record the reply as a CHILD run
    (runlog: parent_run_id + changes_applied noting the branch turn), mirroring replay.py's substrate-safety
    (snapshot the live knobs, restore in a finally, NEVER persist). In the stateless studio a branch
    re-generates from the truncated transcript -- exactly what every normal turn already does, so it is
    correct and adds no new cost; the KV-snapshot fast path (skip the shared-prefix re-prefill) needs the
    generation path to hand back its cache, an honest v1 gap noted in the findings.

GATE: the snapshot store is behind ONE persisted setting (`timetravel_snapshots`, DEFAULT OFF) in the
shared studio_settings.json -- the same off-by-default settings-gate pattern used elsewhere in this
product (capture_mode.py, formerly the facts tier) -- because holding N KV snapshots costs CPU RAM
(measured: ~7 MB/snapshot per 128 tokens on Qwen2.5-7B nf4 bf16-KV; last-8 at ~512 tok ~= 224 MB). Branch
RECORDING (the transcript transform -> child run) does NOT need the store and works regardless of the gate;
the gate only governs whether we hold live KV state for the (future) re-prefill-skipping fast path.

Stdlib only at import time (torch imported lazily, inside the one method that needs a live cache), so this
module -- and its tests -- stay model-free. IO/among-tensors ops never raise into a request.
"""
from __future__ import annotations

from collections.abc import Mapping
import time
import uuid

import clozn.settings as settings  # the single settings file (studio_settings.json) + its never-raise get/set helpers
import clozn.runs.store as runlog

# --------------------------------------------------------------------------------------------- the gate
_ENABLED_KEY = "timetravel_snapshots"

# Defaults for the bounded ring. Tunable via set_config (persisted alongside the gate). The cap is the
# "last N turns" honesty knob; the byte budget is a hard ceiling so a very long conversation can't grow the
# per-snapshot cost without bound (a snapshot's size is O(seq)).
DEFAULT_CAP = 8                    # keep at most this many per-turn snapshots per run (the "last N turns")
DEFAULT_BUDGET_MB = 512           # ... and never exceed this many MB of offloaded KV across a store

TIME_MACHINE_SCHEMA = "clozn.time-machine-eligibility.v1"
TIME_MACHINE_VERIFICATION_SCHEMA = "clozn.time-machine-verification.v1"


def _completed_messages(run: Mapping) -> list[dict]:
    """Return a run's recorded messages with its separately stored reply reattached.

    Session runs retain the input prompt and response as separate immutable fields.  Time Machine
    source matching must compare the completed conversational prefix, but it must never mutate the
    stored run to do so.
    """
    from clozn.runs.think_tags import sanitize_messages

    messages = sanitize_messages(run.get("messages") or [])
    response = run.get("response")
    if (
        isinstance(response, str)
        and messages
        and messages[-1].get("role") == "user"
    ):
        messages = messages + [{"role": "assistant", "content": response}]
    return messages


def completed_message_turns(run: Mapping) -> list[dict]:
    """Return conversational turns from an immutable run's complete recorded output.

    Product generation stores the request messages and current assistant reply in separate fields.
    Exact source resolution must see that reply, while structural branch helpers that operate on a
    caller-supplied message list keep using ``message_turns`` directly.
    """
    if not isinstance(run, Mapping):
        return []
    return message_turns(_completed_messages(run))


def _historical_turn_sources(run: Mapping, turns: list[dict]) -> dict[int, dict]:
    """Resolve unique earlier organic session prefixes in one bounded history scan.

    ``resolve_turn_source_run`` used to scan the session once for every requested turn.  The
    eligibility projection needs all turns, so keep that inspection bounded to one session page and
    one load per candidate.  An ambiguous prefix is deliberately omitted: callers must receive the
    typed unavailable state rather than select a checkpoint by guesswork.
    """
    if not isinstance(run.get("session_key"), str):
        return {}
    try:
        from clozn.runs import sessions, store as runlog
        page = sessions.list_session_runs(run["session_key"], limit=1000)
    except Exception:
        return {}

    requested_id = run.get("id")
    requested_messages = _completed_messages(run)
    candidates: dict[int, list[dict]] = {}
    for summary in (page.get("runs") or []) if isinstance(page, Mapping) else ():
        candidate_id = summary.get("id") if isinstance(summary, Mapping) else None
        if not isinstance(candidate_id, str) or candidate_id == requested_id:
            continue
        try:
            candidate = runlog.get_run(candidate_id)
        except Exception:
            candidate = None
        if not isinstance(candidate, dict):
            continue
        # A historic source has to be an organic completed run.  A child that happens to have the
        # same text prefix is not interchangeable with the source that produced it.
        if candidate.get("parent_run_id") is not None or candidate.get("source") in {
            "branch", "replay", "fork", "execution_fork",
        }:
            continue
        candidate_messages = _completed_messages(candidate)
        candidate_turns = message_turns(candidate_messages)
        if not candidate_turns:
            continue
        turn = int(candidate_turns[-1]["turn"])
        if turn < 0 or turn >= len(turns):
            continue
        target = turns[turn]
        if target.get("assistant_idx") is None:
            continue
        prefix = requested_messages[:int(target["assistant_idx"]) + 1]
        if (
            len(candidate_turns) == turn + 1
            and candidate_turns[-1].get("assistant") == target.get("assistant")
            and candidate_messages == prefix
        ):
            candidates.setdefault(turn, []).append(candidate)

    return {
        turn: matches[0]
        for turn, matches in candidates.items()
        if len(matches) == 1
    }


def resolve_turn_source_run(run: Mapping, turn: int) -> dict | None:
    """Find the unique earlier organic session run for ``turn``'s exact completed prefix.

    The final requested run is intentionally *not* returned here.  This compatibility helper names
    only historical organic sources; ``resolve_exact_turn_source_run`` below also handles the latest
    boundary, whose source is the requested run itself.
    """
    if not isinstance(run, Mapping):
        return None
    turns = completed_message_turns(run)
    if turn < 0 or turn >= len(turns) or turn == turns[-1]["turn"]:
        return None
    return _historical_turn_sources(run, turns).get(turn)


def resolve_exact_turn_source_run(run: Mapping, turn: int) -> dict | None:
    """Return the immutable source run the exact Time Machine routes would use for ``turn``.

    The latest completed boundary is backed by the requested run.  Earlier boundaries are available
    only when a unique organic session-prefix run proves the historical provenance.
    """
    if not isinstance(run, Mapping):
        return None
    turns = completed_message_turns(run)
    if turn < 0 or turn >= len(turns) or turns[turn].get("assistant_idx") is None:
        return None
    if turn == turns[-1]["turn"]:
        return dict(run)
    return _historical_turn_sources(run, turns).get(turn)


def _durable_pin_projection(source_run_id: str) -> dict:
    """Read only the pin ledger row; do not read checkpoint bytes during GET eligibility.

    A stored manifest survives gateway/worker restart, but its blob digest and compatibility with the
    selected worker are checked only when an explicit exact action hydrates it.  Calling
    ``resolve_pin`` here would eagerly read a potentially large KV blob and would overstate a
    worker-specific result from a read-only inspection route.
    """
    try:
        from clozn.replay.checkpoint_pin_store import get_pin
        manifest = get_pin(source_run_id)
    except Exception:
        manifest = None
    if not isinstance(manifest, Mapping):
        return {
            "status": "unavailable",
            "reason": {
                "code": "durable_pin_missing",
                "message": (
                    "no durable checkpoint is recorded for this source run; restart-safe hydration "
                    "is unavailable until an explicit pin succeeds"),
            },
        }
    blob = manifest.get("blob")
    pin_id = manifest.get("pin_id")
    pinned_at = manifest.get("pinned_at")
    if not (
        isinstance(pin_id, str) and pin_id
        and isinstance(pinned_at, str) and pinned_at
        and isinstance(blob, Mapping)
        and isinstance(blob.get("kv_bytes"), int)
        and isinstance(blob.get("envelope_bytes"), int)
    ):
        return {
            "status": "unavailable",
            "reason": {
                "code": "durable_pin_manifest_invalid",
                "message": "the recorded durable checkpoint manifest is incomplete; refusing to claim restart safety",
            },
        }
    return {
        "status": "stored",
        "reason": {
            "code": "durable_pin_recorded",
            "message": (
                "a durable checkpoint is recorded for this source run; the next exact action will "
                "re-read its bytes and fail closed if integrity or runtime compatibility does not match"),
        },
        "pin": {
            "pin_id": pin_id,
            "pinned_at": pinned_at,
            "kv_bytes": blob["kv_bytes"],
            "envelope_bytes": blob["envelope_bytes"],
        },
    }


def turn_source_projection(run: Mapping, turn: int, *, source_run: Mapping | None = None) -> dict:
    """Return a typed, read-only source/pin projection for one Time Machine turn."""
    source = source_run if isinstance(source_run, Mapping) else resolve_exact_turn_source_run(run, turn)
    source_id = source.get("id") if isinstance(source, Mapping) else None
    if not isinstance(source_id, str) or not source_id:
        return {
            "status": "unavailable",
            "durable_pin": {
                "status": "unavailable",
                "reason": {
                    "code": "organic_source_unavailable",
                    "message": (
                        "no unique organic session source run proves this earlier prompt boundary; "
                        "restart-safe checkpoint hydration is unavailable"),
                },
            },
            "reasons": [{
                "code": "organic_source_unavailable",
                "message": (
                    "no unique organic session source run proves this earlier prompt boundary"),
            }],
        }
    requested_id = run.get("id")
    historical = source_id != requested_id
    return {
        "status": "available",
        "run_id": source_id,
        "scope": "session_turn_prompt_boundary" if historical else "full_run_prompt_boundary",
        "source_turn": int(turn),
        "durable_pin": _durable_pin_projection(source_id),
        "reasons": [{
            "code": "organic_session_source_resolved" if historical else "requested_run_source_resolved",
            "message": (
                "a unique organic session run exactly matches this completed historical turn"
                if historical else
                "the requested immutable run backs its latest completed prompt boundary"),
        }],
    }


def replay_eligibility(run: dict | None, store: "SnapshotStore | None" = None) -> dict:
    """Return the model-free Answer Time Machine eligibility receipt for ``run``.

    The current product branch path re-generates a truncated transcript.  It is therefore a
    *structurally reproducible* replay, even when the bounded snapshot ring happens to contain a KV
    descriptor.  A snapshot is reported as useful availability evidence, but it must never be promoted
    to ``exact_replay_eligible`` until a generation path actually restores and verifies that state.

    This is intentionally read-only and never asks a worker to load or generate.  The route can safely
    call it from run inspection and the UI can use the per-turn records to disable an exact-replay action
    with a concrete reason instead of guessing from the presence of a snapshot.
    """
    base = {
        "schema_version": TIME_MACHINE_SCHEMA,
        "run_id": str((run or {}).get("id") or ""),
        "state": "unavailable",
        "eligible": False,
        "exact_replay": {"eligible": False, "reason": {
            "code": "exact_replay_not_available",
            "message": "this Time Machine path does not restore a recorded KV state yet",
        }},
        "reasons": [],
        "turns": [],
    }
    if not isinstance(run, dict) or not base["run_id"]:
        base["reasons"] = [{"code": "run_unavailable", "message": "the run record is unavailable"}]
        return base

    turns = completed_message_turns(run)
    if not turns:
        base["reasons"] = [{"code": "no_replayable_turns",
                            "message": "the run has no user turn that can be branched"}]
        return base

    historical_sources = _historical_turn_sources(run, turns)
    has_cache = False
    for turn in turns:
        snap = store.get(base["run_id"], turn["turn"]) if store is not None else None
        descriptor = snap.descriptor() if snap is not None else None
        cache = bool(snap is not None and snap.has_cache)
        has_cache = has_cache or cache
        turn_reasons = [{
            "code": "structural_replay_only",
            "message": "branching replays the truncated transcript and does not restore exact KV state",
        }]
        if snap is not None and not cache:
            turn_reasons.append({
                "code": "snapshot_descriptor_only",
                "message": "a turn descriptor is present, but it contains no reusable KV payload",
            })
        elif cache:
            turn_reasons.append({
                "code": "snapshot_fast_path_reserved",
                "message": "a KV snapshot is retained, but the current branch implementation does not restore it",
            })
        turn_receipt = {
            "turn": int(turn["turn"]),
            "branch_eligible": True,
            "replay_fidelity": "structurally_reproducible",
            "exact_replay_eligible": False,
            "snapshot": descriptor,
            "source": turn_source_projection(
                run,
                int(turn["turn"]),
                source_run=(run if turn["turn"] == turns[-1]["turn"] else historical_sources.get(turn["turn"])),
            ),
            "reasons": turn_reasons,
        }
        # A prior verification is evidence to display, not a fresh exactness claim: the worker
        # checkpoint is scoped to its process generation and GET must remain cheap/read-only.
        try:
            from clozn.replay import timetravel_results
            previous = timetravel_results.latest_for_run(base["run_id"], int(turn["turn"]))
        except Exception:
            previous = None
        if isinstance(previous, dict):
            last_verification = {
                "verification_id": previous.get("verification_id"),
                "status": previous.get("status"),
                "fidelity": previous.get("fidelity"),
                "exact_replay": previous.get("exact_replay") is True,
                "message": ((previous.get("reasons") or [{}])[0].get("message")
                            if isinstance((previous.get("reasons") or [{}])[0], dict)
                            else None),
            }
            # Historical session-turn proofs name both sides explicitly. Preserve that
            # provenance in the read-only eligibility projection so the UI cannot imply that
            # the final run itself supplied the verified checkpoint.
            for key in ("scope", "requested_run_id", "source_run_id", "source_turn"):
                if key in previous:
                    last_verification[key] = previous[key]
            turn_receipt["last_verification"] = last_verification
        base["turns"].append(turn_receipt)

    base["state"] = "structurally_reproducible"
    base["eligible"] = True
    base["reasons"] = [{
        "code": "structural_replay_available",
        "message": "the recorded messages can be truncated and regenerated when a compatible worker is ready",
    }]
    if has_cache:
        base["reasons"].append({
            "code": "exact_restore_pending",
            "message": "the run has at least one retained KV snapshot, but exact restore is not enabled in this path",
        })
    return base


def verify_prompt_boundary(
    run: dict,
    turn: int,
    engine,
    *,
    runtime_identity,
    worker_identity,
    source_run: Mapping | None = None,
    requested_run_id: str | None = None,
    checkpoint_envelope: Mapping | None = None,
    clock=time.time,
) -> dict:
    """Run the existing exact checkpoint capture/control proof for a Time Machine turn.

    Verification always covers the latest completed boundary of its immutable source run.  For an
    earlier requested turn the route must first resolve and pass that exact organic session-prefix
    run; a final-run checkpoint can never be presented as an earlier-turn restore.  The returned
    artifact is terminal and safe to persist even when capture is unavailable or the unchanged
    control fails.
    """
    requested_id = str(requested_run_id or (run or {}).get("id") or "")
    source = source_run if isinstance(source_run, Mapping) else run
    parent_id = str((source or {}).get("id") or "")
    verification_id = "tmv_" + uuid.uuid4().hex[:20]
    turns = completed_message_turns(source or {})
    is_session_turn = bool(requested_id and requested_id != parent_id)
    base = {
        "schema_version": TIME_MACHINE_VERIFICATION_SCHEMA,
        "verification_id": verification_id,
        "parent_run_id": parent_id,
        "turn": int(turn),
        "scope": "session_turn_prompt_boundary" if is_session_turn else "full_run_prompt_boundary",
        "status": "unavailable",
        "exact_replay": False,
        "fidelity": "unavailable",
        "exactness_regime": "prompt_boundary_reprefill",
        "created_ts": float(clock()),
        "reasons": [],
        "proof": {"status": "not_run"},
        "capture": {},
    }
    if is_session_turn:
        base["requested_run_id"] = requested_id
        base["source_run_id"] = parent_id
        base["source_turn"] = int(turn)
    if not requested_id or not parent_id:
        base["reasons"] = [{"code": "run_unavailable", "message": "the parent run is unavailable"}]
    elif not turns:
        base["reasons"] = [{
            "code": "no_replayable_turns",
            "message": "the run has no conversational turn to verify",
        }]
    elif turn < 0 or turn >= len(turns):
        base["reasons"] = [{
            "code": "turn_out_of_range",
            "message": f"turn {turn} is outside the recorded conversation",
        }]
    elif turn != turns[-1]["turn"]:
        base["reasons"] = [{
            "code": "turn_state_not_available",
            "message": (
                "exact verification currently covers the full-run prompt boundary; "
                "this earlier conversational turn has no persisted worker KV state"),
        }]
    else:
        try:
            from clozn.replay.checkpoint_capture import (
                CheckpointCaptureError,
                capture_parent_checkpoint,
            )
            capture = capture_parent_checkpoint(
                source,
                engine,
                runtime_identity=runtime_identity,
                worker_identity=worker_identity,
                checkpoint_envelope=checkpoint_envelope,
            )
            base["capture"] = capture
            if isinstance(capture, dict):
                if isinstance(capture.get("parent_fingerprint_sha256"), str):
                    base["parent_fingerprint_sha256"] = capture["parent_fingerprint_sha256"]
                if isinstance(capture.get("checkpoint_reference_id"), str):
                    base["checkpoint_reference_id"] = capture["checkpoint_reference_id"]
                if isinstance(capture.get("proof"), dict):
                    base["proof"] = capture["proof"]
                if capture.get("status") == "available":
                    base["status"] = "verified"
                    base["exact_replay"] = True
                    base["fidelity"] = "exact_replay_eligible"
                    base["reasons"] = [{
                        "code": "exact_prompt_boundary_verified",
                        "message": "the unchanged exact-fork control matched the immutable parent",
                    }]
                else:
                    base["status"] = capture.get("status") if capture.get("status") in {
                        "unavailable", "failed"} else "failed"
                    base["reasons"] = list(capture.get("reasons") or [{
                        "code": "exact_verification_unavailable",
                        "message": "the exact checkpoint proof did not complete",
                    }])
        except CheckpointCaptureError as exc:
            base["reasons"] = [{"code": "verification_request_invalid", "message": str(exc)}]
        except Exception as exc:
            base["status"] = "failed"
            base["reasons"] = [{
                "code": "verification_failed",
                "message": f"exact prompt-boundary verification failed: {type(exc).__name__}: {exc}",
            }]
    from clozn import schemas
    schemas.validate(base, TIME_MACHINE_VERIFICATION_SCHEMA)
    return base


def enabled() -> bool:
    """Is per-turn KV snapshotting ON? Default OFF (the RAM rule) -- absent/garbage setting => False.
    Accepts a bool or the strings "on"/"true"/"1"/"yes" (UI persists a bool; be liberal reading)."""
    v = settings.get_setting(_ENABLED_KEY, False)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("on", "true", "1", "yes")


def set_enabled(on: bool) -> bool:
    """Persist the on/off choice into studio_settings.json (merge-write). False on IO failure (never
    raises) -- the caller reports, the request survives."""
    return settings.set_setting(_ENABLED_KEY, bool(on))


def get_config() -> dict:
    """The active ring config {cap, budget_mb}. Reads the persisted overrides if present, else the
    defaults. Values are clamped to sane ranges so a garbage setting can't make the store useless."""
    cap = settings.get_setting("timetravel_cap", DEFAULT_CAP)
    budget = settings.get_setting("timetravel_budget_mb", DEFAULT_BUDGET_MB)
    return {"cap": _clamp_int(cap, DEFAULT_CAP, 1, 128),
            "budget_mb": _clamp_int(budget, DEFAULT_BUDGET_MB, 8, 8192)}


def set_config(cap=None, budget_mb=None) -> bool:
    """Persist ring-config overrides (either/both). Clamped on read (get_config); returns False on IO
    failure. Only writes the keys actually provided."""
    ok = True
    if cap is not None:
        ok = settings.set_setting("timetravel_cap", _clamp_int(cap, DEFAULT_CAP, 1, 128)) and ok
    if budget_mb is not None:
        ok = settings.set_setting("timetravel_budget_mb",
                                     _clamp_int(budget_mb, DEFAULT_BUDGET_MB, 8, 8192)) and ok
    return ok


def _clamp_int(v, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------------------- snapshot cost
def kv_snapshot_bytes(n_tok: int, n_layers: int, n_kv_heads: int, head_dim: int,
                      bytes_per_elt: int = 2) -> int:
    """Bytes a per-turn KV snapshot occupies: keys+values, every layer, over n_tok positions. Pure; the
    single source of truth for the store's accounting AND the "measure and report it" memory number.
    bytes_per_elt defaults to 2 (bf16 -- the studio's nf4-7B runs bf16 activations, so its KV is bf16)."""
    per_pos_per_layer = 2 * int(n_kv_heads) * int(head_dim)     # keys + values
    return int(n_layers) * int(n_tok) * per_pos_per_layer * int(bytes_per_elt)


class Snapshot:
    """One per-turn KV snapshot: the run+turn it belongs to, the token length it covers, its byte size, and
    an OPAQUE payload. `kv` is whatever the caller offloaded (a tuple of CPU tensors) -- or None on the
    stateless studio path, where the turn produced no reusable cache and we keep only the descriptor
    (n_tok) so the branch bookkeeping + Run Inspector affordance still work end-to-end. The store never
    inspects `kv`; it only tracks `nbytes`, so this class is model-free."""

    __slots__ = ("run_id", "turn", "n_tok", "nbytes", "kv", "created_ts", "meta")

    def __init__(self, run_id: str, turn: int, n_tok: int, nbytes: int = 0, kv=None, meta=None):
        self.run_id = str(run_id)
        self.turn = int(turn)
        self.n_tok = int(n_tok)
        self.nbytes = int(nbytes)
        self.kv = kv
        self.created_ts = time.time()
        self.meta = dict(meta or {})

    @property
    def has_cache(self) -> bool:
        """True iff a real KV payload was offloaded (vs a descriptor-only, stateless-path snapshot)."""
        return self.kv is not None

    def descriptor(self) -> dict:
        """The JSON-safe view the API/UI reads (never the raw tensors)."""
        return {"run_id": self.run_id, "turn": self.turn, "n_tok": self.n_tok,
                "nbytes": self.nbytes, "mb": round(self.nbytes / 1048576, 3),
                "has_cache": self.has_cache, "created_ts": round(self.created_ts, 3), **self.meta}


class SnapshotStore:
    """A BOUNDED, CPU-offloaded ring of per-turn KV snapshots. Two independent ceilings, both honest:
      * cap        -- at most this many snapshots per RUN (the "last N turns" knob); evict oldest-turn.
      * budget_mb  -- a hard total-bytes ceiling across the WHOLE store; evict globally-oldest until under.
    Eviction drops the reference to a snapshot's payload so the CPU tensors are freed (GC). Pure-Python
    bookkeeping over Snapshot objects: unit-tested with fabricated payloads, no model, no GPU.

    Keyed by run id so each conversation has its own last-N window; a branch of run R starts a fresh window
    under the child's id (the parent's snapshots are untouched)."""

    def __init__(self, cap: int = DEFAULT_CAP, budget_mb: int = DEFAULT_BUDGET_MB):
        self.cap = max(1, int(cap))
        self.budget_bytes = max(1, int(budget_mb)) * 1048576
        self._by_run: dict[str, list[Snapshot]] = {}     # run_id -> [snapshots], append-order (== turn-order)
        self.total_bytes = 0

    def reconfigure(self, cap=None, budget_mb=None):
        """Apply new ceilings to a LIVE store (so a config change takes effect without a restart) and
        re-run eviction so the existing contents respect them immediately. Either/both; ignores None."""
        if cap is not None:
            self.cap = max(1, int(cap))
        if budget_mb is not None:
            self.budget_bytes = max(1, int(budget_mb)) * 1048576
        for rid in list(self._by_run):
            self._evict_run_over_cap(rid)
        self._evict_over_budget()

    # ---- writes -------------------------------------------------------------------------------------
    def put(self, snap: Snapshot) -> Snapshot:
        """Add a snapshot; enforce BOTH ceilings (per-run cap, then global byte budget). Returns the
        snapshot (so callers can chain). Never raises."""
        lst = self._by_run.setdefault(snap.run_id, [])
        lst.append(snap)
        self.total_bytes += snap.nbytes
        self._evict_run_over_cap(snap.run_id)
        self._evict_over_budget()
        return snap

    def snapshot_turn(self, run_id: str, turn: int, n_tok: int, kv=None, nbytes=None, meta=None) -> Snapshot:
        """Convenience: build + store a Snapshot for (run, turn). `nbytes` may be given explicitly (the
        model-free path / a pre-measured size); if omitted it's inferred from a real kv payload via
        _sizeof_kv (0 for a descriptor-only snapshot). The single entry point the chat path calls."""
        if nbytes is None:
            nbytes = _sizeof_kv(kv) if kv is not None else 0
        return self.put(Snapshot(run_id, turn, n_tok, nbytes=nbytes, kv=kv, meta=meta))

    # ---- eviction (both honest, both drop the payload so RAM is actually reclaimed) ------------------
    def _evict_run_over_cap(self, run_id: str):
        lst = self._by_run.get(run_id, [])
        while len(lst) > self.cap:
            old = lst.pop(0)                             # oldest turn in this run's window
            self.total_bytes -= old.nbytes
            old.kv = None                                # free the offloaded tensors
        if not lst:
            self._by_run.pop(run_id, None)

    def _evict_over_budget(self):
        """Global byte ceiling: drop the globally-oldest snapshot (by created_ts) until under budget."""
        while self.total_bytes > self.budget_bytes:
            victim_run, victim_idx, victim = None, -1, None
            for rid, lst in self._by_run.items():
                if lst and (victim is None or lst[0].created_ts < victim.created_ts):
                    victim_run, victim_idx, victim = rid, 0, lst[0]
            if victim is None:
                break
            self._by_run[victim_run].pop(victim_idx)
            self.total_bytes -= victim.nbytes
            victim.kv = None
            if not self._by_run[victim_run]:
                self._by_run.pop(victim_run, None)

    # ---- reads --------------------------------------------------------------------------------------
    def get(self, run_id: str, turn: int) -> Snapshot | None:
        for s in self._by_run.get(run_id, []):
            if s.turn == turn:
                return s
        return None

    def latest(self, run_id: str) -> Snapshot | None:
        lst = self._by_run.get(run_id, [])
        return lst[-1] if lst else None

    def turns_for(self, run_id: str) -> list[int]:
        return [s.turn for s in self._by_run.get(run_id, [])]

    def count(self) -> int:
        return sum(len(v) for v in self._by_run.values())

    def clear_run(self, run_id: str):
        for s in self._by_run.pop(run_id, []):
            self.total_bytes -= s.nbytes
            s.kv = None

    def stats(self) -> dict:
        """The honest memory receipt the UI/status shows: how many snapshots, over how many runs, and the
        exact offloaded byte total (+ the configured ceilings)."""
        return {"snapshots": self.count(), "runs": len(self._by_run),
                "bytes": self.total_bytes, "mb": round(self.total_bytes / 1048576, 3),
                "cap": self.cap, "budget_mb": round(self.budget_bytes / 1048576, 1)}


def _sizeof_kv(kv) -> int:
    """Total bytes of an offloaded kv payload: a tuple of (keys, values) tensors per layer. Defensive --
    a tensor exposes .element_size()*.nelement(); anything that doesn't contributes 0. No torch import
    (duck-typed), so the store stays model-free."""
    total = 0
    try:
        for pair in kv or ():
            for t in pair:
                try:
                    total += int(t.element_size()) * int(t.nelement())
                except Exception:
                    pass
    except TypeError:
        pass
    return total


def offload_cache(cache) -> tuple:
    """Deep-copy a live transformers Cache's per-layer keys/values to CPU (detached clone), returning the
    same tuple-of-(k,v) shape kv_timetravel.KVChat._snapshot uses. Torch imported lazily; on any failure
    (unexpected Cache internals) returns () so the store just records a descriptor-only snapshot. The one
    torch-touching function here -- everything else is model-free."""
    try:
        layers = getattr(cache, "layers", None)
        if layers is None:
            return ()
        return tuple((layer.keys.detach().clone().cpu(), layer.values.detach().clone().cpu())
                     for layer in layers)
    except Exception:
        return ()


# ------------------------------------------------------------------------- rewind & branch (transcript)
def message_turns(messages) -> list[dict]:
    """Fold a flat message list into conversational TURNS the UI rewinds to. A turn = a user message plus
    the assistant reply that followed it (the reply may be absent for a dangling final user turn). Returns
    [{turn, user, assistant, user_idx, assistant_idx}]; system messages ride with the next turn's context
    but don't start a turn of their own. Pure -- the branch UI reads this to offer 'branch from turn t'."""
    from clozn.runs.think_tags import sanitize_messages
    messages = sanitize_messages(messages)
    turns: list[dict] = []
    cur = None
    for i, m in enumerate(messages or []):
        role = (m or {}).get("role")
        content = (m or {}).get("content", "")
        if role == "user":
            if cur is not None:
                turns.append(cur)
            cur = {"turn": len(turns), "user": content, "assistant": None,
                   "user_idx": i, "assistant_idx": None}
        elif role == "assistant" and cur is not None and cur["assistant"] is None:
            cur["assistant"] = content
            cur["assistant_idx"] = i
    if cur is not None:
        turns.append(cur)
    return turns


def branch_messages(messages, turn: int, alt_user=None) -> list[dict]:
    """The 'rewind & branch from here' transcript transform (PURE). Rewind to turn `turn` and produce the
    message list to RE-GENERATE from: keep everything up to and including turn t's user message, DROP turn
    t's assistant reply and every later turn, and optionally REPLACE turn t's user message with `alt_user`.

    So branching turn t with no alt = 're-roll turn t and everything after it from the same history'; with
    an alt = 'ask something different at turn t and continue from there'. The result is a clean messages[]
    ending in a user turn -- exactly what the stateless chat path re-generates from. Raises ValueError on a
    turn index that doesn't exist (the caller validates + reports; never a silent wrong-branch)."""
    from clozn.runs.think_tags import sanitize_messages
    messages = sanitize_messages(messages)
    turns = message_turns(messages)
    if not turns:
        raise ValueError("no turns to branch from")
    if turn < 0 or turn >= len(turns):
        raise ValueError(f"branch turn {turn} out of range (have {len(turns)} turns)")
    t = turns[turn]
    kept = list(messages[:t["user_idx"] + 1])            # up to & including turn t's user message
    if alt_user is not None and str(alt_user).strip():
        kept = kept[:-1] + [{"role": "user", "content": str(alt_user)}]
    return kept


# --------------------------------------------------------------------------- branch -> child run record
def branch(run: dict, turn: int, sub, alt_user=None, sample: bool = False,
           store: "SnapshotStore | None" = None) -> dict | None:
    """Branch `run` at conversational `turn`: re-generate the truncated (optionally alt-user) transcript on
    the live substrate `sub` and record the reply as a CHILD run (parent_run_id set, changes_applied noting
    the branch turn + whether the user turn was edited). Returns the child run dict, or None on any failure
    (a branch must never raise into the request handler).

    Substrate safety mirrors replay.py: a branch only truncates the transcript and never touches any
    live substrate knob. NEVER persists (no save_state). Greedy by default (sample=False) so the
    branch is deterministic -- the receipt path.

    If a bounded `store` is passed AND a snapshot for (run, turn) holding a real cache exists, this is where
    a future fast path would restore it and skip the shared-prefix re-prefill; v1 re-generates from the
    truncated transcript (correct, and already what every stateless turn costs) and simply notes in
    changes_applied whether such a snapshot was available."""
    try:
        if not run or not isinstance(run, dict):
            return None
        chat = getattr(sub, "chat", None)
        if not callable(chat):
            return None
        try:
            branched = branch_messages(run.get("messages") or [], int(turn), alt_user=alt_user)
        except ValueError:
            return None

        snap = store.get(run.get("id"), int(turn)) if store is not None else None
        changes = {"branch_turn": int(turn),
                   "edited_user": bool(alt_user is not None and str(alt_user).strip()),
                   "kv_snapshot": bool(snap is not None and snap.has_cache)}
        if changes["edited_user"]:
            changes["alt_user"] = str(alt_user)

        t0 = time.time()
        reply = chat(branched, max_new=256, sample=bool(sample))
        reply = reply if isinstance(reply, str) else str(reply)

        rid = runlog.record(
            source="branch", client="studio",
            model=run.get("model"), substrate=run.get("substrate"),
            messages=branched, response=reply,
            parent_run_id=run.get("id"), changes_applied=changes, started=t0,
            session_key=run.get("session_key"), client_key=run.get("client_key"),
            client_key_source=run.get("client_key_source"), project_key=run.get("project_key"),
        )
        if rid is None:
            return None
        child = runlog.get_run(rid)
        return child if child is not None else {"id": rid, "response": reply,
                                                "parent_run_id": run.get("id"),
                                                "changes_applied": changes}
    except Exception:
        return None
