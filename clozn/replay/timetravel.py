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

import inspect
import time

import clozn.settings as settings  # the single settings file (studio_settings.json) + its never-raise get/set helpers
import clozn.runs.store as runlog

# --------------------------------------------------------------------------------------------- the gate
_ENABLED_KEY = "timetravel_snapshots"

# Defaults for the bounded ring. Tunable via set_config (persisted alongside the gate). The cap is the
# "last N turns" honesty knob; the byte budget is a hard ceiling so a very long conversation can't grow the
# per-snapshot cost without bound (a snapshot's size is O(seq)).
DEFAULT_CAP = 8                    # keep at most this many per-turn snapshots per run (the "last N turns")
DEFAULT_BUDGET_MB = 512           # ... and never exceed this many MB of offloaded KV across a store


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
def _snapshot_strength(steer) -> dict:
    try:
        return dict(getattr(steer, "strength", {}) or {})
    except Exception:
        return {}


def branch(run: dict, turn: int, sub, alt_user=None, sample: bool = False,
           store: "SnapshotStore | None" = None) -> dict | None:
    """Branch `run` at conversational `turn`: re-generate the truncated (optionally alt-user) transcript on
    the live substrate `sub` and record the reply as a CHILD run (parent_run_id set, changes_applied noting
    the branch turn + whether the user turn was edited). Returns the child run dict, or None on any failure
    (a branch must never raise into the request handler).

    Substrate safety mirrors replay.py: the live knobs are NOT changed by a branch (we only truncate the
    transcript), but we still snapshot .steer.strength / .memory.memory_strength and restore them in a
    finally so a future knob-carrying variant can't leave the studio mutated. NEVER persists (no
    save_state). Greedy by default (sample=False) so the branch is deterministic -- the receipt path.

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

        steer = getattr(sub, "steer", None)
        mem = getattr(sub, "memory", None) or getattr(sub, "_mem", None)
        saved_strength = _snapshot_strength(steer)
        saved_mem = getattr(mem, "memory_strength", None) if mem is not None else None

        snap = store.get(run.get("id"), int(turn)) if store is not None else None
        changes = {"branch_turn": int(turn),
                   "edited_user": bool(alt_user is not None and str(alt_user).strip()),
                   "kv_snapshot": bool(snap is not None and snap.has_cache)}
        if changes["edited_user"]:
            changes["alt_user"] = str(alt_user)

        t0 = time.time()
        try:
            call_kw = {"max_new": 256, "sample": bool(sample)}
            try:
                if "memory_scope" in inspect.signature(chat).parameters:
                    from clozn.replay.replay import _memory_scope_for_run
                    call_kw["memory_scope"] = _memory_scope_for_run(run)
            except (TypeError, ValueError):
                pass
            reply = chat(branched, **call_kw)
        finally:                                          # restore EXACTLY (never leave the studio mutated)
            if steer is not None:
                try:
                    steer.strength = dict(saved_strength)
                except Exception:
                    pass
            if mem is not None and saved_mem is not None:
                try:
                    mem.memory_strength = saved_mem
                except Exception:
                    pass
        reply = reply if isinstance(reply, str) else str(reply)

        # child memory/behavior summary: a branch keeps the live knobs, so report what's actually in force.
        dials = {}
        try:
            if steer is not None and hasattr(steer, "active"):
                dials = dict(steer.active())
        except Exception:
            dials = {}
        memd = {"strength": float(saved_mem) if saved_mem is not None else 1.0,
                "has_prefix": (getattr(mem, "prefix", None) is not None) if mem is not None else False,
                "cards_applied": [], "proposed_cards": []}

        rid = runlog.record(
            source="branch", client="studio",
            model=run.get("model"), substrate=run.get("substrate"),
            messages=branched, response=reply,
            memory=memd, behavior={"active_dials": dials},
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
