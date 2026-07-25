"""Product model adapter + the shared studio-surface base.

``EngineSubstrate`` is the only product-serving adapter (it talks to the C++ worker over HTTP -- no
Torch). ``Substrate`` is the shared base carrying the studio surface (prompt-card memory + tone dials),
inherited by both the product adapter here and the PyTorch lab adapters. The Torch lab adapters
``QwenSubstrate``/``DreamSubstrate`` have been relocated to ``clozn/lab/substrates.py`` so a product
process can never import a Torch adapter; the product gateway has no loader or route that activates them.
The app module remains the seam:
mutable server state (SUB/SUBNAME/ENGINE*/SLOTS/...) and the helpers routes and tests patch live there,
and this module reads them through `ctx` (late-bound, so a monkeypatch on the app module is always
seen). app re-exports every public name here, so `from clozn.server import app as cs; cs.EngineSubstrate`
keeps working unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

from clozn.server.config import REPO_ROOT, DEMO                        # noqa: F401
from clozn.server import app as ctx   # the seam: live server state + patchable helpers (see docstring)
from clozn.server.request_context import RequestContext   # backlog #2: per-request isolation (see EngineSubstrate)

class Substrate:
    """Shared studio surface for any substrate: the /memory/* trait cards and the /steer/* tone dials, on
    whatever model the subclass loads. A subclass sets self.steer, self._mem (a memory object exposing
    .rules / .prefix / .consolidate(rules) / .reset()), self._pers_steer, self._steer_ready/_steer_info,
    and defines _gen(prompt) -- a one-shot generate used by the /steer/check A/B (AR generate vs denoise).
    So memory + dials are written ONCE and work identically on Qwen and Dream."""

    @staticmethod
    def _requested_card_scope(body):
        """Resolve a public scope *kind* against gateway-injected private request context.

        JSON callers may say only ``scope: global|app|project``.  The opaque key is copied exclusively
        from a real ``MemoryScope`` under the private ``_memory_scope`` seam; a JSON object cannot
        impersonate that frozen dataclass.  This keeps association keys out of the public mutation
        contract and makes a missing gateway association fail closed instead of widening the card.
        """
        from clozn.memory.scope import MemoryScope, app_scope, global_scope, project_scope

        if not isinstance(body, dict):
            return None, "memory request body must be an object"
        if any(field in body for field in ("key", "scope_key", "app_key", "project_key")):
            return None, "raw memory scope keys are not accepted"

        trusted = body.get("_memory_scope")
        if "_memory_scope" in body and not isinstance(trusted, MemoryScope):
            return None, "memory scope context was not injected by the gateway"
        requested = body.get("scope", "global")
        if not isinstance(requested, str) or requested.strip().lower() not in ("global", "app", "project"):
            return None, "scope must be global, app, or project"
        requested = requested.strip().lower()
        if requested == "global":
            return global_scope(), None
        if not isinstance(trusted, MemoryScope):
            return None, f"{requested} scope is unavailable for this request"
        if requested == "app":
            if trusted.app_key is None:
                return None, "app scope is unavailable for this request"
            return app_scope(trusted.app_key), None
        if trusted.project_key is None:
            return None, "project scope is unavailable for this request"
        return project_scope(trusted.project_key), None

    def _memory(self, path, body):
        """Card-backed memory (D2 + E1). Cards carry the metadata + review status; m.rules stays == the
        ACTIVE-card texts and drives the prefix via consolidate(). Status changes go through _mem_sync_rules,
        which only retrains when the active set actually moved -- so pending/no-op edits never touch the prefix."""
        import clozn.memory.cards as memory_cards
        m = self._mem
        self._ensure_cards_migrated()           # one-time seed of legacy rules -> active cards (no retrain)

        if path == "/memory/cards":             # OBJECTS now (not bare strings) -- the review layer
            return {"cards": memory_cards.list_cards(), "has_prefix": m.prefix is not None,
                    "mode": ctx._memory_mode(),     # the UI adapts its copy / hides retrain chrome on this
                    "retraining": self._retrain_status_mode()}   # fold the in-flight signal in (one reload sees it)

        if path == "/memory/retrain-status":    # the poll target: is a background consolidate() running?
            return self._retrain_status_mode()      # prompt mode: never ({active:false, mode:"prompt"})

        if path == "/memory/add":               # propose a card as PENDING -> does NOT affect the prefix
            text = str(body.get("text", "")).strip()
            if not text:
                return {"ok": False, "reason": "empty trait"}
            scope, scope_error = self._requested_card_scope(body)
            if scope_error:
                return {"ok": False, "reason": scope_error}
            card = memory_cards.create(text, status="pending", kind="preference",
                                       risk=ctx._risk_of(text), source_run_id=body.get("source_run_id"),
                                       evidence=str(body.get("evidence", "")), scope=scope)
            if not card:
                return {"ok": False, "reason": "could not create card"}
            # If this is really a STYLE preference, surface the tone DIAL that delivers it (the trained
            # prefix carries topical prefs well but style ones weakly). Card is still created + pending;
            # this only SUGGESTS the better mechanism -- null when the text isn't a style match.
            return {**card, "dial_suggestion": ctx._dial_suggestion(text)}

        if path == "/memory/scope":             # rebind by kind; opaque key comes only from private context
            cid = str(body.get("id", "")).strip()
            if not cid:
                return {"ok": False, "reason": "need a card id"}
            if "scope" not in body:
                return {"ok": False, "reason": "need scope: global, app, or project"}
            scope, scope_error = self._requested_card_scope(body)
            if scope_error:
                return {"ok": False, "reason": scope_error}
            if memory_cards.get(cid) is None:
                return {"ok": False, "reason": "no such card"}
            card = memory_cards.update(cid, scope=scope)
            if card is None:
                return {"ok": False, "reason": "could not update card scope"}
            return card

        if path == "/memory/remove":            # delete by id -> if it was active, rebuild from the rest
            cid = str(body.get("id", "")).strip()
            if not cid:                          # (index removed -- ids are the stable handle now)
                return {"ok": False, "reason": "need a card id"}
            was_active = (memory_cards.get(cid) or {}).get("status") == "active"
            ok = memory_cards.delete(cid)
            if not ok:
                return {"ok": False, "reason": "no such card"}
            # delete is synchronous+fast; the retrain (only if an ACTIVE card left the set) is backgrounded.
            resync = self._start_retrain(m, "remove", cid) if was_active else {"retraining": False}
            return {"ok": True, "removed": cid, "resync": resync}

        if path in ("/memory/approve", "/memory/reject", "/memory/disable", "/memory/enable"):
            return self._card_status(path.rsplit("/", 1)[1], str(body.get("id", "")).strip())

        if path == "/memory/edit":              # change a card's text; if active, retrain on the new text
            cid = str(body.get("id", "")).strip()
            new_text = str(body.get("text", "")).strip()
            if not (cid and new_text):
                return {"ok": False, "reason": "need id and text"}
            card = memory_cards.update(cid, text=new_text, risk=ctx._risk_of(new_text))
            if card is None:
                return {"ok": False, "reason": "no such card"}
            if card.get("status") == "active":   # editing an active card's text retrains -> in the background
                card = {**card, "resync": self._start_retrain(m, "edit", cid)}
            return card

        if path == "/memory/strength":          # the memory dial. Internalized: scales how hard the prefix
            # bites (0 = off, >1 = stronger). PROMPT mode: on/off only -- 0 never injects the block, any
            # >0 injects when the topic gate lets it in (nothing scales continuously; the UI hint says so).
            if "value" in body and hasattr(m, "memory_strength"):
                m.memory_strength = max(0.0, min(2.0, float(body["value"])))
                if hasattr(m, "save"):
                    try:
                        m.save()                             # persists inside the .pt (needs a prefix)
                    except Exception:
                        pass
                try:                                         # mirror to settings so the dial survives a
                    import clozn.memory.mode as memory_mode                       # restart in prompt mode (no .pt to carry it)
                    memory_mode.set_setting("memory_strength", float(m.memory_strength))
                except Exception:
                    pass
            return {"strength": float(getattr(m, "memory_strength", 1.0)), "has_prefix": m.prefix is not None,
                    "mode": ctx._memory_mode()}

        if path == "/memory/gatecheck":         # DEBUG (live calibration): the topic-relevance gate for a prompt
            # Exposes both raw signals + the final gate + per-rule cosines for the active rules, so the bands
            # (lo_t/hi_t/lo_o/hi_o) can be tuned against real on/off-topic prompts. Fully guarded: on any
            # failure (or no embedder) it reports the no-gating baseline (gate 1.0) rather than raising.
            prompt = str(body.get("prompt", ""))
            rules = list(getattr(m, "rules", []) or [])
            try:
                from clozn.memory.topic_gate import get_gate
                dbg = get_gate().debug(prompt, rules)
            except Exception as e:
                dbg = {"gate": 1.0, "topic": 0.0, "openness": 0.0, "relevance": {},
                       "ok": False, "error": f"{type(e).__name__}: {e}"}
            # `gate` here is the RAW topic gate (relevance only); the applied scale is memory_strength x
            # gate (internalized) or include-iff gate >= ctx.PROMPT_GATE_MIN (prompt mode) -- mode says which.
            return {"prompt": prompt, "rules": rules, "mode": ctx._memory_mode(),
                    "strength": float(getattr(m, "memory_strength", 1.0)), **dbg}
        return None

    # ---- E1 review lifecycle: a status change rebuilds m.rules from the active set, retrains iff it moved -
    def _card_status(self, action, cid):
        """approve->active, reject->rejected, disable->disabled, enable->active. The STATUS flip (fast) is
        synchronous; the RETRAIN it may trigger (rebuild the prefix from active_texts) is backgrounded so
        the response returns immediately. The card keeps its FINAL status; a separate _RETRAIN flag carries
        the in-flight signal. self._start_retrain no-ops when the active set didn't actually move (prefix safe).

        PROVENANCE GATE: 'approve' is refused for a card that CLAIMS a run (source_run_id
        set) but carries no quoted_span to back that claim up -- memory_cards.is_provenance_claim_unbacked.
        This is never auto-approvable; the reviewer sees why via the reason string (the Memory page also
        flags it so this should rarely even be attempted). reject/disable/enable are NOT gated -- you must
        always be able to discard or de-activate a card regardless of its provenance."""
        import clozn.memory.cards as memory_cards
        if not cid:
            return {"ok": False, "reason": "need a card id"}
        if action == "approve":
            existing = memory_cards.get(cid)
            if existing is not None and memory_cards.is_provenance_claim_unbacked(existing):
                return {"ok": False, "reason": "no provenance -- this card cites a run but has no quoted "
                                                "span backing it up, so it can't be approved"}
        target = {"approve": "active", "reject": "rejected",
                  "disable": "disabled", "enable": "active"}[action]
        card = memory_cards.set_status(cid, target)
        if card is None:
            return {"ok": False, "reason": "no such card"}
        resync = self._start_retrain(self._mem, action, cid)  # retrains on a thread iff the active set changed
        return {**card, "resync": resync}

    def _ensure_cards_migrated(self):
        """Seed the card store from this substrate's legacy rule-strings exactly once per process."""
        if getattr(self, "_cards_migrated", False):
            return
        ctx._mem_migrate(self._mem)
        self._cards_migrated = True

    # ---- retrain dispatch: PRODUCT path (prompt mode -- never retrains) -----------------------------
    # The internalized soft-prefix RETRAIN machinery (background consolidate + the in-flight banner) is a
    # LAB thing now: it lives on clozn/lab/substrates.py's _InternalizedRetrain mixin, which QwenSubstrate/
    # DreamSubstrate inherit and which OVERRIDES these four. The product (EngineSubstrate) carries only the
    # trivial prompt-mode versions below -- the cards ARE the memory, so a mutation is instant bookkeeping,
    # no thread, no _TRAIN_LOCK, no retrain banner. _memory/_card_status dispatch through self.<method>, so
    # the same call site is instant here and a backgrounded consolidate on the lab substrates.
    def _retrain_status_mode(self):
        """The retrain signal the UI polls. Prompt mode never retrains -> a constant idle
        ({active: false, mode: "prompt"}). Byte-identical to the old app-module helper's prompt branch."""
        return {"active": False, "mode": "prompt"}

    def _start_retrain(self, m, action, card_id, force=False):
        """Prompt-mode card mutation: ONLY bookkeeping -- sync m.rules to the active-card texts (runlog +
        /state read it), instantly. No consolidate, no thread, no _TRAIN_LOCK, no retrain banner; a trained
        prefix (if one exists from a lab session) is left completely untouched. Byte-identical to the old
        app-module _start_retrain's prompt short-circuit (which also ignored `force`)."""
        r = ctx._mem_sync_rules(m, reconsolidate=False)      # instant: rules bookkeeping only
        return {"retraining": False, "changed": r["changed"], "mode": "prompt"}

    def _retrain_in_flight(self):
        return False

    def _join_retrain(self, timeout=None):
        return True

    def _ensure_steer(self):
        """Compute the axis vectors once, race-safe (double-checked lock). Two dial calls racing on first
        use could otherwise both run compute() on the shared model at once and corrupt it (IndexError).

        A dead engine surfaces here as a raw urllib.error.URLError -- self.steer.compute()'s very first
        harvest() round-trip fails to even connect (EngineClient._request only translates an HTTPError,
        i.e. the engine responding with a JSON 4xx, into EngineError; a connection refusal propagates
        unwrapped). Caught here and re-raised as ctx.EngineUnavailable so the caller gets the same clean
        502 every other engine-touching route uses, instead of a bare URLError reaching app.py's generic
        500 fallback (engine-down pressure test finding #1)."""
        if not self._steer_ready:
            with self._steer_lock:
                if not self._steer_ready:
                    import urllib.error
                    try:
                        self._steer_info = self.steer.compute()
                    except urllib.error.URLError as e:
                        raise ctx.EngineUnavailable(ctx._engine_unreachable_message()) from e
                    self._steer_ready = True

    def _steer(self, path, body):
        from clozn.behavior.steering.axes import AXES
        if path == "/steer/axes":
            calib = ctx._dial_calibration()   # {} when uncalibrated/offline -- ctx._with_calibration no-ops per axis
            axes = [ctx._with_calibration(
                        {"name": k, "poles": AXES[k]["poles"], "value": self.steer.strength.get(k, 0.0),
                         "max": AXES[k].get("max", 1.5)}, calib.get(k))
                    for k in AXES]
            lib_names = ctx._library_dial_names()   # shipped-library custom dials -- NOT user-made, never "yours"
            for k, v in getattr(self.steer, "custom", {}).items():   # user-defined + shipped-library dials
                axis = {"name": k, "poles": v["poles"], "value": self.steer.strength.get(k, 0.0), "max": v["max"]}
                if k in lib_names:
                    axis["library"] = True     # shipped, curated dial -- distinct from a user's own custom
                else:
                    axis["custom"] = True      # unchanged: a genuine user-made dial ("yours" + deletable)
                axes.append(ctx._with_calibration(axis, calib.get(k)))
            return {"axes": axes, "ready": self._steer_ready, "substrate": self.name}
        if path == "/steer/custom_delete":      # a pure dict-pop -- doesn't touch the engine at all, so it
            # must work (or fail on its OWN terms) even while the engine is down; must NOT sit behind
            # _ensure_steer()'s unrelated ~35-round-trip calibration harvest (pressure test finding #1b).
            if hasattr(self.steer, "remove_custom"):
                self.steer.remove_custom(str(body.get("name", "")))
                self.steer.save_custom(ctx._pers(f"studio_custom_{self.name}.json"))
                if self._pers_steer:
                    self.steer.save_state(self._pers_steer)
            return {"custom": list(getattr(self.steer, "custom", {}))}
        if path == "/steer/concept/set":         # Tier-1 #1: any-concept dial (dir(c)) -- ZERO calibration.
            # A DIFFERENT mechanism (ConceptSteer) from the diff-of-means EngineSteer _ensure_steer()
            # calibrates -- must not be gated behind that unrelated subsystem's readiness either.
            import clozn.behavior.steering.concept_dir as concept_dir
            cs = ctx._engine_concept_steer()
            if cs is None:
                return {"error": "concept dials need the product model worker (CLOZN_ENGINE_PORT)"}
            concept = str(body.get("concept", "")).strip()
            if not concept:
                return {"error": "need a concept word"}
            strength = float(body.get("strength", concept_dir.DEFAULT_STRENGTH))
            result = cs.steer_toward(concept, strength)
            result["active"] = cs.active()
            return result
        if path == "/steer/concept/check":       # A/B: baseline vs dir(concept)-steered (mirrors /steer/check)
            import clozn.behavior.steering.concept_dir as concept_dir
            cs = ctx._engine_concept_steer()
            if cs is None:
                return {"error": "concept dials need the product model worker (CLOZN_ENGINE_PORT)"}
            concept = str(body.get("concept", "")).strip()
            if not concept:
                return {"error": "need a concept word"}
            strength = float(body.get("strength", concept_dir.DEFAULT_STRENGTH))
            prompt = str(body.get("prompt", ""))[:300]
            max_new = int(body.get("max_new", 90))
            base = concept_dir._text_of(cs.ec.complete(prompt, max_tokens=max_new))
            built = cs.steer_toward(concept, strength)
            if not built.get("ok"):
                return {"prompt": prompt, "concept": concept, "strength": strength,
                        "baseline": base, "steered": None,
                        "blocked": built.get("blocked"), "note": built.get("note")}
            steered = concept_dir._text_of(cs.ec.intervene(
                prompt, vector=built["vector"], coef=built["coef"], layer=built["layer"], max_tokens=max_new))
            return {"prompt": prompt, "concept": concept, "strength": strength, "layer": built["layer"],
                    "token_id": built["token_id"], "coef": built["coef"],
                    "baseline": base, "steered": steered, "note": built.get("note")}
        self._ensure_steer()                    # compute the axis vectors once on first real use (race-safe)
        if path == "/steer/compute":
            return {"ready": True, **self._steer_info}
        if path == "/steer/set":
            warning = self.steer.set(str(body["name"]), float(body.get("value", 0.0)))
            if self._pers_steer:
                self.steer.save_state(self._pers_steer)
            result = {"active": self.steer.active()}
            if warning:
                # Same message EngineSteer.set() already put on stderr (visible to whoever's running
                # `clozn serve`) -- also handed back on the wire so an HTTP/Studio caller with no
                # terminal in view sees it too. One warning, computed once, two audiences.
                result["warning"] = warning
            return result
        if path == "/steer/check":              # A/B one dial: baseline vs steered (subclass _gen)
            prompt = str(body.get("prompt", ""))[:300]
            base = self._gen(prompt)
            # The check is diagnostic, not a settings mutation. Preserve every pre-existing value (including
            # explicit zeros used by the UI) and the engagement state. The old implementation cleared the
            # whole live persona after every A/B check, so merely inspecting one dial silently erased all
            # persisted in-process tone settings until the next restart/profile switch.
            prior = dict(getattr(self.steer, "strength", {}) or {})
            was_engaged = bool(getattr(self.steer, "_engaged", False))
            self.steer.clear()
            warning = self.steer.set(str(body["name"]), float(body.get("value", 1.0)))
            self.steer.engage()
            try:
                steered = self._gen(prompt)
            finally:
                self.steer.disengage()
                self.steer.clear()
                for name, value in prior.items():
                    self.steer.set(name, value)   # restoring prior state, not a new user action -- its
                                                   # own warning (if any) already fired when it was set
                if was_engaged:
                    self.steer.engage()
            result = {"prompt": prompt, "axis": body.get("name"), "value": body.get("value", 1.0),
                      "baseline": base, "steered": steered}
            if warning:
                result["warning"] = warning
            return result
        if path == "/steer/custom":             # USER-DEFINED dial: compute mean(+pole)-mean(-pole) live
            if not hasattr(self.steer, "add_custom"):
                return {"error": "custom dials are not supported on this substrate yet"}
            name = str(body.get("name", "")).strip()[:24]
            pos, neg = str(body.get("pos", "")).strip(), str(body.get("neg", "")).strip()
            if not (name and pos and neg):
                return {"error": "need a name and both poles (pos, neg)"}
            info = self.steer.add_custom(name, pos, neg, float(body.get("max", 0.5)))
            self.steer.save_custom(ctx._pers(f"studio_custom_{self.name}.json"))
            return {"name": name, "max": info["max"], "custom": list(self.steer.custom)}
        return None


class _EngineMemory:
    """Thin prompt-mode memory for the engine substrate: the CARD STORE *is* the memory. No model, no
    learned prefix (the soft-prefix TTT is a lab experiment now, not shipped in the engine product -- see
    RUNTIME_SPLIT.md). Exposes exactly the surface the base Substrate._memory handler, ctx._prompt_block_for,
    and the receipts/replay stack read: .rules (active-card texts), .prefix (always None), .memory_strength,
    ._exclude_card_ids (replay sets this for per-card receipts), .consolidate/.reset (no-ops -- prompt mode
    never trains), .state(), .lock."""

    def __init__(self):
        self.prefix = None
        self._exclude_card_ids = None
        self.lock = threading.Lock()
        try:
            import clozn.memory.mode as memory_mode                    # 0.35 == the shipped product default (commit f3e9f60, the
            self.memory_strength = float(memory_mode.get_setting("memory_strength", 0.0))    # off by default;
        except Exception:                          # cards are opt-in via the UI strength slider, not always-on
            self.memory_strength = 0.0             # prompt injection into unrelated topics.

    @property
    def rules(self):
        import clozn.memory.cards as memory_cards
        return [c["text"] for c in (memory_cards.list_cards() or []) if c.get("status") == "active"]

    @rules.setter
    def rules(self, _value):
        # The card store IS the memory here, so `rules` is derived and has nothing to set. The shared
        # _mem_sync_rules() assigns m.rules for the soft-prefix (SelfTeach) backend; make that a harmless
        # no-op on the engine substrate instead of an AttributeError -- otherwise every approve/reject/
        # disable/enable/remove crashed AFTER already mutating the store (scrappy error toast, action
        # silently succeeded). The store stays the single source of truth.
        pass

    def consolidate(self, rules):
        return {"ok": True, "mode": "prompt"}      # prompt mode never trains a prefix

    def reset(self):
        pass

    def state(self):
        import clozn.memory.cards as memory_cards
        return {"mode": "prompt", "has_prefix": False,
                "cards": len(memory_cards.list_cards() or []), "rules": self.rules}


class EngineSubstrate(Substrate):
    """PURE-ENGINE substrate: chat + prompt-mode memory + tone dials on the C++ GGUF runtime, NO PyTorch
    model resident. THIS is the class that brings the whole torch-free Server tier -- /v1/chat/completions,
    replay, receipts, explain, narrate, counterfactual -- onto the fast engine, because every one of those
    routes through SUB.chat(). Memory is prompt-mode only (the card store as a topic-gated system block);
    dials apply via EngineSteer's steer_vec. See RUNTIME_SPLIT.md (the keystone)."""

    name = "engine"

    # IDENTITY LAZY RE-RESOLUTION (engine-down pressure test finding #2): a down-at-startup engine pays
    # this ~2s connect-refused tax (this host's control fact) at most once per cooldown window on lazy
    # re-resolve attempts (see _maybe_reresolve_identity) -- never on every request, so a persistently-dead
    # engine never adds latency to ordinary calls.
    _IDENTITY_RETRY_COOLDOWN_S = 30.0

    def __init__(self):
        if ctx.ENGINE is None:
            raise RuntimeError("engine substrate needs the supervised GGUF worker (set CLOZN_ENGINE_PORT)")
        self.engine = ctx.ENGINE
        self.steer = ctx._engine_steer()            # an EngineSteer on the GGUF (tone dials via steer_vec)
        if self.steer is not None:               # metadata-only: the shipped library's names/poles/max, so
            try:                                  # they show up in /steer/axes immediately (their direction
                self.steer.load_library(ctx._pers("studio_library.json"))   # vectors are computed lazily by compute())
                self.steer.load_custom(ctx._pers(f"studio_custom_{self.name}.json"))  # + the user's own custom dials
            except Exception:
                pass
        self._mem = _EngineMemory()
        self.memory = self._mem                 # the studio reads SUB.memory in a few places
        self._steer_ready = False
        self._steer_info = {}
        self._steer_lock = threading.Lock()
        self.brain = None                       # no SAE/brain on the pure-engine substrate (concepts 409 cleanly)
        # T0.2: reflect the ACTUALLY-LOADED GGUF, not a hardcoded Qwen assumption. Derive the family from
        # the engine's /health model file (best-effort -- never blocks boot if the engine isn't up yet)
        # and pin the tone-dial steer tap to THIS model's mid-depth: Qwen-7B -> 14 (unchanged), Llama-3.2-1B
        # -> 8, an unrecognized GGUF keeps EngineSteer's generic default. run_meta() re-derives this lazily
        # too, so the run record is correct even when the engine comes up after the substrate.
        #
        # model_family/model_id/model_sha256/_pers_steer are PROPERTIES (below), backed by the _val fields
        # here: a startup-time engine outage must not permanently disable per-model dial persistence for
        # the rest of the process's life -- every read retries _resolve_identity() (cooldown-gated) while
        # unresolved, and never re-fetches once it resolves. See _resolve_identity/_maybe_reresolve_identity.
        self._model_family_val = None
        self._model_id_val = None
        self._model_sha256_val = None
        self._pers_steer_val = None
        self._identity_lock = threading.Lock()
        self._identity_last_attempt = 0.0
        self._resolve_identity()

    def _resolve_identity(self):
        """One best-effort attempt to derive model_family/model_id/model_sha256 from the engine's /health,
        pin the tone-dial steer layer, and -- once a sha256 is actually known -- load this exact GGUF's
        persisted dial state and enable J-transport. Never blocks boot, never raises (a down/old engine
        just leaves everything at its unresolved default). Called once at construction and retried lazily
        by _maybe_reresolve_identity whenever the engine was down at the previous attempt."""
        self._identity_last_attempt = time.time()
        h = {}
        try:
            h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
            fam, _info = _engine_model_info((h or {}).get("model", ""))
            self._model_family_val = fam
            self._model_id_val = _info["model_id"]
            if self.steer is not None and _info["steer_layer"] is not None:
                self.steer.layer = _info["steer_layer"]
        except Exception:
            return
        sha256 = str((h or {}).get("model_sha256") or "") or None
        if not sha256:
            return
        self._model_sha256_val = sha256
        self._pers_steer_val = ctx._pers(os.path.join("models", sha256, "studio_personality.json"))
        if self.steer is not None:              # restore values only from this exact GGUF's state file
            try:
                self.steer.load_state(self._pers_steer_val)
            except Exception:
                pass
            # J-TRANSPORT (engine_adapter.EngineSteer's class docstring / jlens_transport.py, see
            # notes/JLENS_SAE_FINDINGS.md finding #1): auto-enable using the running engine's OWN
            # reported model digest -- the strongest identity this substrate actually has (no local
            # GGUF file path to re-derive full contracts.gguf_identity() metadata from). Safe to
            # always attempt: a byte-identical no-op (self.steer.last_j_transport["applied"] is
            # False) whenever no compact-eligible J artifact claims this exact GGUF sha256 -- true
            # for every model shipped today -- never a silent substitution of a mismatched J.
            try:
                self.steer.enable_j_transport(model_sha256=sha256)
            except Exception:
                pass

    def _maybe_reresolve_identity(self):
        """Retry _resolve_identity() iff identity is still unresolved (no model_sha256 yet) AND the
        cooldown has elapsed since the last attempt. A no-op once resolved -- never re-fetches, matching
        the pre-existing "resolve once" behavior for a healthy startup -- and a no-op within the cooldown
        window so a persistently-down engine doesn't add a health() round-trip to every request."""
        if self._model_sha256_val:
            return
        if time.time() - self._identity_last_attempt < self._IDENTITY_RETRY_COOLDOWN_S:
            return
        with self._identity_lock:               # double-checked, so concurrent callers don't stack retries
            if self._model_sha256_val:
                return
            if time.time() - self._identity_last_attempt < self._IDENTITY_RETRY_COOLDOWN_S:
                return
            self._resolve_identity()

    @property
    def model_family(self):
        self._maybe_reresolve_identity()
        return self._model_family_val

    @property
    def model_id(self):
        self._maybe_reresolve_identity()
        return self._model_id_val

    @property
    def model_sha256(self):
        self._maybe_reresolve_identity()
        return self._model_sha256_val

    @property
    def _pers_steer(self):
        self._maybe_reresolve_identity()
        return self._pers_steer_val

    def _gen(self, prompt):                     # one-shot generate for the /steer/check A/B (base _steer)
        if self.steer is not None:
            return self.steer.generate(prompt, max_new=90)
        from clozn.behavior.steering.engine_adapter import EngineSteer
        return EngineSteer._text(self.engine.complete(prompt, max_tokens=90))

    # ---- per-request context: request isolation (backlog #2) ------------------------------------------
    # chat()/chat_stream() each start with self._new_request(), then write everything the call learns
    # about ITSELF onto that one object (see request_context.RequestContext's docstring for why). The
    # properties below are the back-compat SEAM: every existing reader of sub._last_generation_meta /
    # _last_finish_reason / _last_diverged / _last_diverged_at / _last_stream_trace keeps working
    # unchanged, unaware that the piecemeal attributes became views onto self._request. Deliberately
    # EngineSubstrate-only (not on the shared Substrate base): QwenSubstrate/DreamSubstrate (clozn/lab/
    # substrates.py) still WRITE these same names as plain instance attributes -- putting a property with
    # no setter on the shared base would break that assignment with `AttributeError: can't set attribute`
    # the moment a lab substrate's chat() ran. Read-only on purpose: the only legitimate writers are
    # chat()/chat_stream() below, and they now write through `self._request` instead.
    def _new_request(self) -> RequestContext:
        """Start this call's own RequestContext and publish it as 'the current one' in a single attribute
        assignment. Must be the FIRST thing chat()/chat_stream() do, mirroring exactly where the old code
        used to reset self._last_generation_meta/_last_diverged/_last_diverged_at at call start."""
        self._request = RequestContext()
        return self._request

    @property
    def _last_generation_meta(self):
        req = getattr(self, "_request", None)
        return req.generation_meta if req is not None else None

    @property
    def _last_finish_reason(self):
        req = getattr(self, "_request", None)
        return req.finish_reason if req is not None else None

    @property
    def _last_diverged(self):
        req = getattr(self, "_request", None)
        return req.diverged if req is not None else None

    @property
    def _last_diverged_at(self):
        req = getattr(self, "_request", None)
        return req.diverged_at if req is not None else None

    @property
    def _last_stream_trace(self):
        req = getattr(self, "_request", None)
        return req.trace if req is not None else []

    @property
    def _last_prompt_tokens(self):
        req = getattr(self, "_request", None)
        return req.prompt_tokens if req is not None else None

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None,
             reference_tokens=None, apply_anchored=False, memory_scope=None):
        """One stateless chat completion on the engine with memory (prompt-mode card block) + tone dials
        applied. Mirrors QwenSubstrate.chat's contract EXACTLY (same signature, same trace_out/mem_out
        fill) so the receipts/replay stack is backend-agnostic.

        `sample`: the caller's request to sample (True), force greedy (False), or override this request's
        sampling fields with a dict. REPRODUCE_AND_PROVE_PLAN S5: `sample=True` (the default) resolves via
        ctx._resolve_sampling against the persisted "sampling" setting (default ON,
        Ollama/llama.cpp's canonical temperature=0.8/top_p=0.9/top_k=40/repeat_penalty=1.1, a FRESH seed
        every turn); the setting off degrades to greedy, byte-identical to pre-S5 behavior.
        `sample=False` ALWAYS decodes greedy (temperature 0) regardless of the setting -- this is what the
        receipt/replay/forced-scoring stack relies on: it forces the STORED token ids over this generation
        (replay.py passes `sample=False` for every `{"greedy": True}` change spec, which every receipt
        path uses), so a sampled interactive run's receipts are computed exactly like a greedy run's.

        `reference_tokens` (optional): the baseline reply's committed token ids. When present, the engine
        EARLY-STOPS this generation at the first token that differs from the reference (prove-all ablated
        arms) -- so the reply is a bit-exact PREFIX of what full generation would produce, plus a divergence
        verdict stashed for last_divergence(). This is a pure termination check -- decode/sampling (greedy
        or not) are otherwise untouched, so a diverged reply is still a bit-exact prefix either way.

        REQUEST ISOLATION (backlog #2): this call's own RequestContext (self._new_request()) replaces the
        old piecemeal self._last_generation_meta/_last_diverged/_last_diverged_at instance writes -- see
        request_context.RequestContext's docstring. _last_generation_meta/_last_diverged/_last_diverged_at
        stay readable exactly as before (now read-only views onto self._request); nothing about this
        call's CONTROL FLOW or the reply it returns changed."""
        # `apply_anchored` is explicit so live OpenAI chat can use X7 anchored memory, while receipts/replay
        # keep the pre-existing deterministic baseline unless they intentionally opt in.
        req = self._new_request()
        samp = ctx._resolve_sampling(sample)
        req.sampling = samp
        req.generation_meta = ctx._engine_generation_meta(max_new, stream=False, sample=samp)
        # MEMORY: the active cards as a topic-gated system block (omitted off-topic / when strength 0).
        decision = (ctx._prompt_block_for(self.memory, ctx._last_user(messages))
                    if memory_scope is None else
                    ctx._prompt_block_for(
                        self.memory, ctx._last_user(messages), request_scope=memory_scope))
        block, applied, gate = decision
        ctx._capture_prompt_decision(mem_out, decision)
        if applied and mem_out is not None:
            baseline_tokens = ctx._baseline_prompt_tokens(self.engine, messages)
            if baseline_tokens is not None:
                mem_out["baseline_prompt_tokens"] = baseline_tokens
        assembled = ctx._inject_block(messages, block)
        template_usage = {}
        prompt = ctx._engine_tmpl(self.engine, assembled, usage_out=template_usage)
        if mem_out is not None and isinstance(template_usage.get("prompt_tokens"), int):
            mem_out["actual_prompt_tokens"] = template_usage["prompt_tokens"]
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); assembled_messages is its
            # pre-template form. Both recorded so the run is inspectable at either level.
            mem_out.update(mode="prompt", applied=applied, gate=gate,
                           prompt_block=block, assembled_messages=assembled, final_prompt=prompt)
        # TONE: dials from self.steer.strength (replay toggles this in place), falling back to disk.
        kw = {}
        st = (getattr(self.steer, "strength", None) if self.steer is not None else None) or ctx._disk_dials()
        req.steering_snapshot = dict(st) if st else {}      # what THIS call used, decoupled from the live dict
        if self.steer is not None and st and any(st.values()):
            sv = self.steer.steer_vector(st)
            if sv:
                kw["steer_vec"] = sv
                kw["steer"] = {"coef": 1.0, "layer": self.steer.layer}
        comp = ((ctx._apply_anchored_memory(kw, mem_out, ctx._last_user(messages))
                 if memory_scope is None else ctx._apply_anchored_memory(
                     kw, mem_out, ctx._last_user(messages), request_scope=memory_scope))
                if apply_anchored else None)
        if reference_tokens:                                # prove-all early-stop: halt when the answer changes
            kw["reference_tokens"] = [int(t) for t in reference_tokens if t is not None]
        usage = {}
        traced_kw = {"sample": samp}
        try:
            import inspect
            params = inspect.signature(ctx._engine_complete_traced).parameters.values()
            if ("usage_out" in inspect.signature(ctx._engine_complete_traced).parameters
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)):
                traced_kw["usage_out"] = usage
        except Exception:
            pass
        reply_raw, steps, finish, divinfo = ctx._engine_complete_traced(
            self.engine, prompt, max_new, kw, **traced_kw)
        if isinstance(usage.get("prompt_tokens"), int):
            req.prompt_tokens = usage["prompt_tokens"]
        req.generation_timing = dict(usage.get("generation_timing") or {})
        req.finish_reason = finish                          # stash for last_finish_reason() (the log path)
        req.diverged, req.diverged_at = divinfo             # stash for last_divergence()
        if comp is not None:                                 # LOOP GUARD: only when anchored memory was
            reply_raw, steps, finish = ctx._anchored_loop_guard(  # ACTUALLY injected this turn (comp is not
                self.engine, prompt, max_new, kw, samp, comp, reply_raw, steps, finish, mem_out)
            req.finish_reason = finish                      # None) -- see ctx._anchored_loop_guard's docstring
            if mem_out is not None and mem_out.get("anchored_loop_guard"):
                # The guard may have replaced the first generation with one or two retries.  Those older
                # helper calls do not return timing, so discard the initial pass's timing rather than attach
                # it to a different final reply.
                req.generation_timing = {}
        if trace_out is not None:
            trace_out.extend(steps)
        req.trace = list(steps)
        if mem_out is not None:
            req.memory_manifest = dict(mem_out)
        return reply_raw.strip()

    def _complete_chat_native(self, messages, *, tools=None, tool_choice="auto", json_schema=None,
                              parallel_tool_calls=False, max_new=256, sample=True,
                              trace_out=None, mem_out=None, apply_anchored=False,
                              add_generation_prompt=True, enable_thinking=True,
                              reasoning_format="none", memory_scope=None) -> dict:
        """Private atomic model-native structured chat on the C++ worker.

        This is deliberately a substrate seam, not an OpenAI route or a qualification claim.  The
        worker owns the model's chat template, grammar, generation, and native output parser for the
        whole request via ``EngineClient.complete_chat``; keeping those operations atomic prevents a
        client-held prepared descriptor from drifting between rendering and generation.

        Clozn still owns the layers around that native operation.  Prompt-card memory is injected into
        the message list before the worker renders it, tone dials (and explicitly requested anchored
        memory) use the same raw steering channel as :meth:`chat`, and sampling resolves through the
        same per-request policy.  The worker's buffered response contains the actual native event JSON,
        so it is folded through ``accumulate_ar_events`` rather than reconstructed from a final board.

        The return value keeps ``raw_model_output`` byte-for-byte as supplied by the worker and exposes
        its parsed OpenAI message separately.  The same atomic response also carries the exact rendered
        tool/schema prompt, so ``final_prompt`` is recorded from worker evidence rather than from a
        second, potentially drifting render request.
        """
        req = self._new_request()
        samp = ctx._resolve_sampling(sample)
        req.sampling = samp
        req.generation_meta = ctx._engine_generation_meta(max_new, stream=False, sample=samp)

        decision = (ctx._prompt_block_for(self.memory, ctx._last_user(messages))
                    if memory_scope is None else
                    ctx._prompt_block_for(
                        self.memory, ctx._last_user(messages), request_scope=memory_scope))
        block, applied, gate = decision
        assembled = ctx._inject_block(messages, block)
        memory_manifest = {
            "mode": "prompt",
            "applied": applied,
            "gate": gate,
            "prompt_block": block,
            "assembled_messages": assembled,
        }
        ctx._capture_prompt_decision(memory_manifest, decision)
        if applied:
            # Native structured chat adds tools/schema material inside its atomic renderer.  A plain
            # apply_template(messages) baseline would charge that unrelated material to memory, so leave
            # the per-card token delta unavailable until the atomic endpoint exposes a matched baseline.
            memory_manifest["prompt_token_cost_unavailable_reason"] = \
                "structured_prompt_baseline_not_captured"
        if mem_out is not None:
            mem_out.update(memory_manifest)

        options = {}
        st = (getattr(self.steer, "strength", None) if self.steer is not None else None) or ctx._disk_dials()
        req.steering_snapshot = dict(st) if st else {}
        if self.steer is not None and st and any(st.values()):
            steer_vec = self.steer.steer_vector(st)
            if steer_vec:
                options["steer_vec"] = steer_vec
                options["steer"] = {"coef": 1.0, "layer": self.steer.layer}

        # Anchored memory is opt-in, matching chat().  The atomic response now carries real token events,
        # but this private seam intentionally does not run chat()'s multi-pass loop guard: retrying would
        # require a second atomic structured generation and could produce a different parsed call.  The
        # public route must not opt in until that policy is explicitly designed and qualified.
        if apply_anchored:
            if memory_scope is None:
                ctx._apply_anchored_memory(options, memory_manifest, ctx._last_user(messages))
            else:
                ctx._apply_anchored_memory(
                    options, memory_manifest, ctx._last_user(messages), request_scope=memory_scope)
            if mem_out is not None:
                mem_out.update(memory_manifest)

        if samp and samp.get("on"):
            options.update(
                temperature=float(samp["temperature"]),
                rep_penalty=float(samp["repeat_penalty"]),
                top_k=int(samp["top_k"]),
                top_p=float(samp["top_p"]),
                seed=int(samp["seed"]),
            )
        else:
            options.update(temperature=0.0, rep_penalty=1.0, top_k=0, top_p=1.0, seed=0)

        # Publish the memory decision even if the worker fails.  It describes what was assembled for
        # this request, not a claim that generation succeeded.
        req.memory_manifest = dict(memory_manifest)
        response = self.engine.complete_chat(
            assembled,
            tools=tools,
            tool_choice=tool_choice,
            json_schema=json_schema,
            parallel_tool_calls=parallel_tool_calls,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_format=reasoning_format,
            max_tokens=int(max_new),
            **options,
        )

        choice = response["choices"][0]
        chat_io = response["chat_io"]
        usage = dict(response.get("usage") or {})
        finish = choice.get("finish_reason")
        req.finish_reason = finish if isinstance(finish, str) else None
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            req.prompt_tokens = prompt_tokens

        native_events = chat_io.get("trace")
        if not isinstance(native_events, list):
            native_events = []
        import clozn.runs.store as runlog
        steps = runlog.accumulate_ar_events(native_events)
        req.generation_timing = runlog.generation_timing_from_frames(native_events)
        req.trace = list(steps)
        if trace_out is not None:
            trace_out.extend(steps)

        # Unlike the earlier descriptor, the hardened atomic response carries the exact rendered prompt
        # from the same in-worker prepare/generate transaction.  It is now valid evidence for the normal
        # context receipt and replaces the pre-generation manifest snapshot on both channels.
        rendered_prompt = chat_io["rendered_prompt"]
        memory_manifest["final_prompt"] = rendered_prompt
        req.memory_manifest = dict(memory_manifest)
        if mem_out is not None:
            mem_out["final_prompt"] = rendered_prompt

        parse_error = chat_io.get("parse_error")
        parsed_message = chat_io.get("message")
        return {
            "raw_model_output": chat_io["raw_model_output"],
            "rendered_prompt": rendered_prompt,
            "model_sha256": chat_io["model_sha256"],
            "message": dict(parsed_message) if isinstance(parsed_message, dict) else None,
            "openai_json": chat_io.get("openai_json"),
            "format": chat_io["format"],
            "pipeline": dict(chat_io.get("pipeline") or {}),
            "parse_error": dict(parse_error) if isinstance(parse_error, dict) else None,
            "finish_reason": req.finish_reason,
            "usage": usage,
            "trace": list(steps),
        }

    def last_divergence(self):
        """The early-stop verdict from the most recent chat(): (diverged, diverged_at). (None, None) when
        the last chat carried no reference_tokens. Read by replay to record whether an ablated arm's reply
        was truncated at the point it provably changed."""
        return (getattr(self, "_last_diverged", None), getattr(self, "_last_diverged_at", None))

    def score_tokens(self, messages, continuation_ids=None, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        """Teacher-forced per-token logprob of a continuation under EXPLICIT (block,
        steer_strengths) conditions -- the seam the forced-scoring
        stack (rederive.py, forced receipts) builds on. Assembles the prompt EXACTLY like chat()
        (ctx._inject_block + ctx._engine_tmpl -- the loaded model's own chat template) and the steer_vec EXACTLY
        like chat() (self.steer.steer_vector),
        but from the CALLER's `block`/`steer_strengths` -- NEVER from live self.memory/self.steer.strength
        -- so a with/without arm is reconstructed purely from a run record (memory ablation = recompile
        the block without a card; dial ablation = zero a strength and recompute) rather than from
        whatever the live substrate happens to be doing right now. That's what makes receipt arms
        reconstructable: two calls with different explicit `block`/`steer_strengths`, same messages
        and continuation_ids, are directly comparable. No sampling anywhere; deterministic.

        `block`: a prompt-mode memory block string (or None to omit it), e.g. run.memory.prompt_block.
        `steer_strengths`: a {dial_name: strength} dict (or None for no steer), e.g. run.behavior.dials.
        `continuation_ids`: the PRIMARY continuation form (token ids, e.g. from a stored trace) --
        takes precedence over `continuation` when both are given (mirrors EngineClient.score).
        `continuation`: a TEXT fallback (S3's rederive.py, for a run whose trace lacks per-token ids) --
        the engine retokenizes it independently of the prompt, which can drift at the prompt/
        continuation BPE boundary (flagged `boundary_approximate` by /score itself; see
        REPRODUCE_AND_PROVE_PLAN.md's tokenization-boundary caveat).
        `steer_vec`: an explicit RAW steer direction, ADDED on top of whatever `steer_strengths`
        produces (or used alone if `steer_strengths` is falsy) -- the S3 null-floor control needs a
        direction with no named dial behind it ("a random vector of equal norm at the same layer").

        Returns [{"id", "piece", "logprob"}, ...] (+ "topk" per token when topk>0), one entry per
        continuation token, in the SAME order as continuation_ids (or the engine's own retokenization
        of `continuation` text).
        """
        assembled = ctx._inject_block(messages, block)
        prompt = ctx._engine_tmpl(self.engine, assembled)
        kw = {}
        sv = None
        if self.steer is not None and steer_strengths and any(steer_strengths.values()):
            sv = self.steer.steer_vector(steer_strengths)
        if steer_vec is not None:
            sv = [a + b for a, b in zip(sv, steer_vec)] if sv else list(steer_vec)
        if sv:
            kw["steer_vec"] = sv
            # self.steer.layer is model-aware (pinned per-family in __init__); with no steer built, pass
            # layer 0 so the ENGINE picks its own calibrated mid-depth band -- not a hardcoded Qwen 14.
            kw["steer"] = {"coef": 1.0, "layer": self.steer.layer if self.steer is not None else 0}
        if continuation_ids is not None:
            kw["continuation_ids"] = [int(t) for t in continuation_ids]
        elif continuation is not None:
            kw["continuation"] = str(continuation)
        r = self.engine.score(prompt=prompt, topk=int(topk), **kw)
        return r.get("tokens", [])

    def jlens(self, text, layer=None, topk=5):
        """Proxy the engine's /jlens for the Run Inspector J-lens panel -- mirrors score_tokens' /score
        proxy. Returns a NORMALIZED dict (never raises): the engine's {layer, n_tokens, tokens, readouts}
        plus available_layers (from /health's jlens.layers). Graceful absence: if the engine was started
        WITHOUT --jlens (no jlens block in /health), returns {available:False, reason:...} so the panel
        shows a clean 'lens not loaded' instead of an error. An unknown layer surfaces the engine's 400
        body (the available layers) cleanly rather than throwing.

        The health() probe itself is wrapped SEPARATELY from parsing its body (pressure test finding #5):
        a connection failure (the engine isn't running at all) is factually different from a reachable
        engine whose /health simply carries no jlens block (it's up, just started without --jlens), and
        the two used to collapse into the same wrong "started without --jlens" reason whenever the engine
        was fully down."""
        try:
            h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
        except Exception:
            return {"available": False, "reason": ctx._engine_unreachable_message()}
        try:
            jl = (h or {}).get("jlens") or {}
            avail = [int(x) for x in (jl.get("layers") or [])]
        except Exception:
            avail = []
        if not avail:
            return {"available": False, "reason": "the engine was started without --jlens"}
        try:
            r = self.engine.jlens(text, layer=layer, topk=int(topk))
        except ctx.EngineError as e:
            # e.g. an unknown layer -> the engine's 400 {error, available}. Surface it cleanly (the panel
            # can offer the loaded layers); available_layers already comes from /health above.
            return {"available": True, "error": str(e), "available_layers": avail,
                    "layer": layer, "n_tokens": 0, "tokens": [], "readouts": []}
        return {"available": True, "layer": r.get("layer"), "available_layers": avail,
                "n_tokens": int(r.get("n_tokens", 0) or 0),
                "tokens": r.get("tokens", []), "readouts": r.get("readouts", [])}

    def last_stream_trace(self):
        """The per-token trace captured during the most recent chat_stream (raw step list, or []) --
        same contract as QwenSubstrate.last_stream_trace: the SSE handler reads this AFTER the generator
        is exhausted, to log the run's Run Inspector timeline."""
        return list(getattr(self, "_last_stream_trace", []) or [])

    def last_finish_reason(self):
        """The stop cause ("stop"|"length"|...) from the most recent chat()/chat_stream, or None. Same
        stash-and-read contract as last_stream_trace: the handler reads it AFTER generation, so the run
        logs WHY the engine stopped instead of a hard-coded 'stop'."""
        return getattr(self, "_last_finish_reason", None)

    def last_prompt_tokens(self):
        """The prompt token count the engine's own `gen_started` frame reported for the most recent
        chat_stream, or None (a non-streaming chat() call, an engine build too old to send the field, or
        nothing streamed yet). Same stash-and-read contract as last_stream_trace/last_finish_reason --
        read by clozn.server.ndjson to fill an honest `prompt_eval_count` on the Ollama NDJSON shim's
        final chunk (roadmap Phase 2 #1); never guessed when absent."""
        return getattr(self, "_last_prompt_tokens", None)

    def run_meta(self):
        """Reproducibility metadata -- WHAT produced a run -- for the run record. Fetched once from
        /health (model file -> quant, engine mode) and cached; the STATIC baseline here is the honest
        greedy default (temperature 0) -- the ACTUAL regime a specific reply used (greedy, or S5 sampled
        with its params + seed) rides in _last_generation_meta, filled by chat()/chat_stream() and merged
        in below, so a call made before any generation still reports the honest baseline rather than a
        guess. Health-derived fields are omitted when unavailable rather than guessed. Never raises:
        metadata never breaks a run."""
        health_meta = getattr(self, "_run_meta", None)
        if health_meta is None:
            health_meta = {}
            h: dict = {}
            mp = ""
            try:
                h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
                mp = str((h or {}).get("model", ""))
                if mp:
                    health_meta["model_file"] = mp.replace("\\", "/").rsplit("/", 1)[-1]
                    q = _quant_from_name(health_meta["model_file"])
                    if q:
                        health_meta["quant"] = q
                    # T0.2: which model actually produced this run (derived from the loaded GGUF, not a
                    # hardcoded id). family is the registry key; model_id the friendly HF name when known.
                    fam, info = _engine_model_info(mp)
                    if fam:
                        health_meta["family"] = fam
                    if info.get("model_id"):
                        health_meta["model_id"] = info["model_id"]
                if (h or {}).get("mode"):
                    health_meta["mode"] = h["mode"]
                for k in ("n_ctx", "device", "gpu_layers"):
                    v = (h or {}).get(k)
                    if v is not None:
                        health_meta[k] = v
            except Exception:
                pass
            # roadmap S4.3: immutable reproduction identity, assembled from the SAME /health fetch above --
            # never a second round trip, never a fresh file hash on the request path. Prefers the engine's
            # own reported model_sha256 (computed once at boot, see clozn.runs.identity's module
            # docstring); only falls back to hashing model_path itself when the engine didn't report one,
            # and even that fallback is cached process-wide by _identity_meta_val (below) plus
            # clozn.runs.identity's own on-disk cache. A failure here must not cost the health_meta this
            # call already earned, so it gets its own try/except rather than sharing the one above.
            try:
                from clozn.runs import identity as run_identity
                self._identity_meta_val = run_identity.runtime_identity(
                    model_path=mp or None,
                    model_sha256_hint=(h or {}).get("model_sha256"),
                    apply_template_fn=getattr(self.engine, "apply_template", None),
                    engine_health=h if isinstance(h, dict) else None,
                )
            except Exception:
                self._identity_meta_val = {}
            self._run_meta = dict(health_meta)
        meta = ctx._engine_generation_meta()
        meta.update(dict(health_meta))
        meta.update(getattr(self, "_last_generation_meta", None) or {})
        request = getattr(self, "_request", None)
        meta.update(dict(getattr(request, "generation_timing", None) or {}))
        prompt_tokens = getattr(self, "_last_prompt_tokens", None)
        if isinstance(prompt_tokens, int):
            meta["prompt_tokens"] = prompt_tokens
        return dict(meta)

    def identity_meta(self) -> dict:
        """The run record's top-level `identity` block (roadmap S4.3): model_sha256, model_path,
        model_size_bytes, template_fingerprint, engine_build, clozn_version, captured_at -- whichever of
        those this process could actually establish. Computed once per process inside run_meta()'s single
        /health fetch (see its comment) and cached in self._identity_meta_val, so calling this never adds
        a network round trip or a file read beyond what run_meta() already pays for. Calls run_meta()
        first if it hasn't run yet this process; never raises."""
        if getattr(self, "_identity_meta_val", None) is None:
            try:
                self.run_meta()
            except Exception:
                pass
        return dict(getattr(self, "_identity_meta_val", None) or {})

    def chat_stream(self, messages, max_new=256, mem_out=None, lens=None, on_frame=None, sample=True,
                    memory_scope=None):
        """Streaming twin of chat(): the SAME memory-block + tone-dial + anchored-memory construction
        (kept in lockstep -- see chat()'s comments; do not let this drift from that logic), but opens the engine's
        /v1/completions with stream:True (mirrors _engine_complete_traced's request) and yields text as
        the engine commits it, instead of waiting on one blocking call. This is what makes /v1/chat/
        completions's SSE branch (_sse_chat, gated on `getattr(SUB, "chat_stream", None)`) fire on the
        pure-engine substrate too -- before this existed, a streaming request here silently fell through
        to one non-streamed chat() reply. mem_out: as in chat() -- prompt mode records what memory
        actually rode this turn.

        F1 LIVE LENS: lens = a dict {layer?, topk?, every?} (or {} for engine defaults) rides to the
        engine as body["lens"]; the engine then interleaves `jlens_live` frames (the J-lens
        "disposed to say" readout for each committed token, computed mid-generation) with the token
        frames. Each one is handed to `on_frame(obj)` as it arrives -- a side-channel, because this
        generator's yields are text pieces and must stay that way for every existing consumer. A
        failing on_frame is dropped (never kills generation).

        Per-token trace (mirrors QwenSubstrate.chat_stream's B3 contract): every parsed SSE frame is
        collected, then folded into self._last_stream_trace via runlog.accumulate_ar_events once the
        stream ends -- normal completion OR an early GeneratorExit (the consumer stopped early) -- so a
        partial stream still logs whatever trace it managed. Wrapped so any parse hiccup just leaves it
        [], same as the non-streaming path's fallback.

        SAMPLING (S5): `sample` has the same bool-or-override-dict contract as chat(). The OpenAI SSE route
        uses the dict form so explicit temperature/top_p/top_k/repeat_penalty/seed values affect the stream;
        ordinary callers use True and inherit the persisted defaults. The master setting off degrades to
        greedy, byte-identical to pre-S5 behavior.

        REQUEST ISOLATION + CANCELLATION (backlog #2): this call's own RequestContext (self._new_request())
        replaces the old piecemeal self._last_generation_meta/_last_stream_trace/_last_finish_reason
        instance writes (see request_context.RequestContext's docstring); the piecemeal names stay readable
        exactly as before. The context also carries a cancellation Event: sse.py's caller sets it (via
        self._request.cancel()) the instant it detects the CLIENT is gone (a failed write to the far end),
        and the read loop below checks it between worker frames as a second, belt-and-suspenders stop
        alongside the GeneratorExit an explicit `gen.close()` throws at the `yield` below -- either one
        aborts the worker's chunked send promptly instead of draining a reply nobody will read."""
        import urllib.error
        import urllib.request
        import clozn.runs.store as runlog
        req = self._new_request()
        samp = ctx._resolve_sampling(sample)
        req.sampling = samp
        req.generation_meta = ctx._engine_generation_meta(max_new, stream=True, sample=samp)
        # MEMORY + TONE: built EXACTLY as chat() builds them.
        decision = (ctx._prompt_block_for(self.memory, ctx._last_user(messages))
                    if memory_scope is None else
                    ctx._prompt_block_for(
                        self.memory, ctx._last_user(messages), request_scope=memory_scope))
        block, applied, gate = decision
        ctx._capture_prompt_decision(mem_out, decision)
        if applied and mem_out is not None:
            baseline_tokens = ctx._baseline_prompt_tokens(self.engine, messages)
            if baseline_tokens is not None:
                mem_out["baseline_prompt_tokens"] = baseline_tokens
        assembled = ctx._inject_block(messages, block)
        template_usage = {}
        prompt = ctx._engine_tmpl(self.engine, assembled, usage_out=template_usage)
        if mem_out is not None and isinstance(template_usage.get("prompt_tokens"), int):
            mem_out["actual_prompt_tokens"] = template_usage["prompt_tokens"]
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); kept in lockstep with chat().
            mem_out.update(mode="prompt", applied=applied, gate=gate,
                           prompt_block=block, assembled_messages=assembled, final_prompt=prompt)
        kw = {}
        st = (getattr(self.steer, "strength", None) if self.steer is not None else None) or ctx._disk_dials()
        req.steering_snapshot = dict(st) if st else {}      # what THIS call used, decoupled from the live dict
        if self.steer is not None and st and any(st.values()):
            sv = self.steer.steer_vector(st)
            if sv:
                kw["steer_vec"] = sv
                kw["steer"] = {"coef": 1.0, "layer": self.steer.layer}
        # F6 ANCHORED MEMORY (X7): active bags compose into ONE gated steer_vec at L21 and ride live chat.
        if memory_scope is None:
            ctx._apply_anchored_memory(kw, mem_out, ctx._last_user(messages))
        else:
            ctx._apply_anchored_memory(
                kw, mem_out, ctx._last_user(messages), request_scope=memory_scope)
        body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_new)
        if samp and samp.get("on"):     # S5: real sampling -- Ollama-style temperature/top_k/top_p/rep_penalty/seed
            body["temperature"] = float(samp["temperature"])
            body["rep_penalty"] = float(samp["repeat_penalty"])
            body["top_k"] = int(samp["top_k"])
            body["top_p"] = float(samp["top_p"])
            body["seed"] = int(samp["seed"])
        else:
            body["temperature"] = 0.0; body["rep_penalty"] = 1.0; body["seed"] = 0
        body["stream"] = True
        if lens is not None:                # F1 live lens: opt-in passthrough (engine validates layer etc.)
            body["lens"] = lens if isinstance(lens, dict) else True
        wreq = urllib.request.Request(self.engine.base + "/v1/completions",
                                      data=json.dumps(body).encode("utf-8"),
                                      headers={"Content-Type": "application/json"})
        frames = []
        try:
            resp = urllib.request.urlopen(wreq, timeout=getattr(self.engine, "timeout", 600))
        except urllib.error.HTTPError as he:
            # surface the engine's own error text (e.g. a bad lens layer's 400) instead of a bare code
            try:
                detail = json.loads(he.read()).get("error") or str(he)
            except Exception:
                detail = str(he)
            raise RuntimeError(f"engine: {detail}")
        try:
            for raw in resp:
                if req.is_cancelled():          # CANCELLATION: the caller already gave up on this request
                    break                        # (client gone) -- stop pulling from the worker between frames
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                if req.engine_req is None:
                    # StreamEnvelope (server_shared.hpp) stamps `req` on EVERY frame of this stream, so
                    # the first one parsed already carries it -- capture once here rather than waiting
                    # for a specific frame type. This is the req_ <-> worker-req correlation
                    # request_context.py's new_request_id() describes: routes/engine.py's POST /cancel
                    # reads it off self._request to resolve a gateway id to the worker's own.
                    engine_req = obj.get("req")
                    if engine_req:
                        req.engine_req = str(engine_req)
                if req.prompt_tokens is None and obj.get("type") == "gen_started":
                    # roadmap Phase 2 #1 (Ollama NDJSON streaming): the engine's own accounting of how
                    # many prompt tokens this generation evaluated (engine/core/include/cloze/events.hpp),
                    # captured once off the first gen_started frame -- honest source for the Ollama shim's
                    # `prompt_eval_count` (clozn.server.ndjson), never derived/guessed elsewhere.
                    prompt_tokens = obj.get("prompt_tokens")
                    if isinstance(prompt_tokens, int):
                        req.prompt_tokens = prompt_tokens
                if obj.get("type") == "jlens_live":     # F1: side-channel to the SSE relay, never yielded
                    if on_frame is not None:
                        try:
                            on_frame(obj)
                        except Exception:
                            on_frame = None             # a dead callback must never kill generation
                    continue
                if obj.get("type") == "tokens_committed":
                    for it in obj.get("items") or []:
                        piece = it.get("piece", "")
                        if piece:
                            yield piece
        finally:
            # ALWAYS release the engine connection -- whether the stream ran to [DONE] or the consumer
            # stopped early (this `finally` also runs when the caller .close()s us mid-stream, via a
            # GeneratorExit at the `yield` above); guarded so a close() hiccup can never mask a
            # propagating GeneratorExit -- it must reach the caller, never be swallowed here. (The
            # engine-side crash-on-disconnect is a separate C++-side task; this just closes cleanly.)
            try:
                resp.close()
            except Exception:
                pass
            try:
                req.trace = runlog.accumulate_ar_events(frames)
            except Exception:
                req.trace = []
            try:
                req.finish_reason = runlog.finish_reason_from_frames(frames)
            except Exception:
                req.finish_reason = None
            try:
                req.generation_timing = runlog.generation_timing_from_frames(frames)
            except Exception:
                req.generation_timing = {}
            # LOOP GUARD, streaming twin: the engine sets the anchored
            # steer at generation-START (body["steer_vec"] above) and every piece is yielded to the
            # caller live over SSE -- by the time this `finally` runs, the client has ALREADY received
            # the whole reply. There is no seamless mid-stream re-injection here (unlike chat()'s
            # auto-retry-at-half-strength): the honest thing this path can do is DETECT the degeneracy
            # after the fact and FLAG the run -- never fake a retry/self-heal capability streaming
            # doesn't have. Only checked when anchored memory actually rode this turn (mem_out["anchored"]
            # set, not anchored_skipped/absent) -- a degeneracy safety, never a claim the memory "worked".
            if mem_out is not None and mem_out.get("anchored"):
                try:
                    from clozn.memory import anchored as _anchored_lg
                    pieces = [str(s.get("piece", "")) for s in (req.trace or [])]
                    if _anchored_lg.detect_loop(pieces):
                        mem_out["anchored_loop_guard"] = {
                            "fired": True, "action": "flagged-only", "resolved": False,
                            "note": ("streaming reply already reached the client -- detected after the "
                                     "fact, no mid-stream retry is possible on this path")}
                except Exception:
                    pass
            if mem_out is not None:
                req.memory_manifest = dict(mem_out)

    def handle(self, path, body):
        r = self._memory(path, body)
        if r is not None:
            return r
        return self._steer(path, body)

    def state(self):
        import clozn.memory.cards as memory_cards
        return {"dials": dict(getattr(self.steer, "strength", {}) or {}),
                "cards": len(memory_cards.list_cards() or [])}


def _quant_from_name(name):
    """Pull the GGUF quant tag (Q4_K_M, Q8_0, IQ4_XS, F16, ...) out of a model filename, or None. GGUF
    files name their quantization in the basename, so this is the one bit of repro metadata we can read
    for free (no engine change) off /health's model path."""
    import re
    m = re.search(r"(IQ\d+[A-Z0-9_]*|Q\d+(?:_[A-Z0-9]+)+|Q\d+|BF16|F16|F32)", str(name), re.IGNORECASE)
    return m.group(1).upper() if m else None


# --- engine model registry (T0.2) ---------------------------------------------------------------------
# The engine substrate reflects the ACTUALLY-LOADED GGUF, not a hardcoded "Qwen2.5-7B" id/assumption.
# The ONE Qwen-specific assumption the engine substrate carried was the tone-dial steer TAP LAYER
# (mid-depth: 14 for Qwen-7B's 28 layers). This tiny registry keys that -- plus a friendly model_id for
# run_meta -- off the loaded model's family (derived from its /health filename), with a sensible default
# for any unrecognized GGUF (steer_layer None => don't pin a layer; let the engine use its OWN per-model
# calibrated mid-depth steer band). Everything else the engine already calibrates per-model server-side
# (the C++ concept/steer probe taps at startup, and the chat template via /apply_template). This is NOT a
# framework -- it is the minimal table that removes the last hardcoded-Qwen coupling from the engine path.
_ENGINE_MODELS = {
    "qwen2.5-7b":   {"model_id": "Qwen/Qwen2.5-7B-Instruct",         "steer_layer": 14},  # 28L -> mid 14 (unchanged)
    "qwen2.5-0.5b": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",       "steer_layer": 12},  # 24L -> mid 12
    "qwen3.5-9b":   {"model_id": "Qwen/Qwen3.5-9B",                  "steer_layer": None},
    "llama-3.1-8b": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "steer_layer": None},
    "llama-3.2-1b": {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "steer_layer": 8},   # 16L -> mid 8
    "llama-3.2-3b": {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "steer_layer": 14},  # 28L -> mid 14
    "gemma4-e4b":   {"model_id": "google/gemma-4-E4B-it",            "steer_layer": None},
    "ministral3-3b": {"model_id": "mistralai/Ministral-3-3B-Instruct-2512",
                        "steer_layer": None},
}
_ENGINE_MODEL_DEFAULT = {"model_id": None, "steer_layer": None}  # unknown GGUF: nothing pinned; engine picks


def _model_family_from_name(name):
    """Coarse model family key ('qwen2.5-7b', 'llama-3.2-1b', ...) from a GGUF filename, or None -- the
    engine substrate looks up per-model assumptions in _ENGINE_MODELS by this key instead of hardcoding
    Qwen's. Same free derive-off-/health-filename trick as _quant_from_name (no engine change needed)."""
    import re
    s = str(name or "").lower()
    m = re.search(r"qwen[._]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"qwen{m.group(1)}-{m.group(2)}b"
    m = re.search(r"llama[._-]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"llama-{m.group(1)}-{m.group(2)}b"
    if re.search(r"gemma[._-]?4[._-]?e4b", s):
        return "gemma4-e4b"
    m = re.search(r"ministral[._-]?3[._-]?(\d+(?:\.\d+)?)b", s)
    if m:
        return f"ministral3-{m.group(1)}b"
    return None


def _engine_model_info(name):
    """(family, {model_id, steer_layer}) for the loaded GGUF -- the engine substrate's per-model
    assumptions -- or (None, the default with nothing pinned) for an unrecognized model."""
    fam = _model_family_from_name(name)
    return fam, dict(_ENGINE_MODELS.get(fam, _ENGINE_MODEL_DEFAULT))


def _engine_complete_traced(engine, prompt, max_tokens, kw, sample=None, usage_out=None):
    """Generate on the engine and ALSO capture a per-token trace (issue B3), returning
    (reply, steps, finish, divinfo).

    The engine's non-streaming /v1/completions carries only the final text -- no per-token confidence. To
    populate the Run Inspector timeline we ask the SAME request with stream:True and fold its per-token
    `tokens_committed`/`step_lens` frames into steps via runlog.accumulate_ar_events. Generation defaults to
    greedy (temperature 0) so the reassembled text is identical to the blocking call; passing a
    ctx._resolve_sampling() dict as `sample` (S5) switches BOTH the streaming attempt and the fallback below to
    the same temperature/rep_penalty/seed, so the two paths always agree. Either way we only capture
    ALONGSIDE; the client still receives the same single JSON reply (this streams engine<->server, never to
    the client). Any streaming hiccup falls back to the plain complete() so a run is never lost -- just
    without a trace. (AR GGUFs only; a diffusion engine commits out of reading order and emits no such
    per-token stream.)

    `divinfo` is (diverged, diverged_at) when the request carried `reference_tokens` (prove-all early-stop):
    diverged True/False is the engine's verdict, else (None, None). The reply on a diverged run is a
    BIT-EXACT PREFIX of the full generation (the engine only adds a stop check -- decode is unchanged).

    `sample`: None (or a falsy dict) -- greedy, temperature=0/rep_penalty=1/seed=0/top_k=0/top_p=1,
    byte-identical to pre-S5 behavior. A ctx._resolve_sampling() dict -- temperature/repeat_penalty/seed
    plus the full Ollama nucleus (top_k/top_p) ride the request as the engine's own SampleConfig keys;
    the engine's sampler (engine/core/src/sample.cpp) truncates to top-k then the top-p nucleus before
    the draw, so a sampled chat lands on the same distribution the user knows from Ollama. Greedy
    (temperature 0) ignores all of them -- the argmax path is untouched, receipts stay exact."""
    on = bool(sample and sample.get("on"))
    temperature = float(sample["temperature"]) if on else 0.0
    rep_penalty = float(sample["repeat_penalty"]) if on else 1.0
    top_k = int(sample["top_k"]) if on else 0
    top_p = float(sample["top_p"]) if on else 1.0
    seed = int(sample["seed"]) if on else 0
    import urllib.request
    body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_tokens)
    body["temperature"] = temperature; body["rep_penalty"] = rep_penalty; body["seed"] = seed
    body["top_k"] = top_k; body["top_p"] = top_p
    body["stream"] = True
    try:
        req = urllib.request.Request(engine.base + "/v1/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        frames, text = [], ""
        diverged, diverged_at = None, None
        with urllib.request.urlopen(req, timeout=getattr(engine, "timeout", 600)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                if (usage_out is not None and obj.get("type") == "gen_started"
                        and isinstance(obj.get("prompt_tokens"), int)):
                    usage_out["prompt_tokens"] = obj["prompt_tokens"]
                ch = obj.get("choices")                     # the final OpenAI-style frame carries the full text
                if ch and isinstance(ch, list) and ch[0].get("text"):
                    text = ch[0]["text"]
                if "diverged" in obj:                       # early-stop verdict rides the final frame
                    diverged = obj.get("diverged")
                    diverged_at = obj.get("diverged_at")
        import clozn.runs.store as runlog
        steps = runlog.accumulate_ar_events(frames)
        finish = runlog.finish_reason_from_frames(frames)   # the engine's real stop cause (else None)
        if not text:                                        # no final frame text -> reassemble from the pieces
            text = "".join(s.get("piece", "") for s in steps)
        if steps or text:
            if usage_out is not None:
                timing = runlog.generation_timing_from_frames(frames)
                if timing:
                    usage_out["generation_timing"] = timing
            return text, steps, finish, (diverged, diverged_at)
    except Exception:
        pass
    # Fallback: the original blocking path, reply preserved, trace simply empty. The non-streaming
    # /v1/completions carries the same `diverged`/`diverged_at` when a reference was sent. Same
    # temperature/rep_penalty/seed as the streaming attempt above -- the fallback must never silently
    # decode under a DIFFERENT regime than the one recorded in the run's meta.
    r = engine.complete(prompt, max_tokens=max_tokens, temperature=temperature, rep_penalty=rep_penalty,
                        top_k=top_k, top_p=top_p, seed=seed, **kw)
    prompt_tokens = (r.get("usage") or {}).get("prompt_tokens") if isinstance(r, dict) else None
    if usage_out is not None and isinstance(prompt_tokens, int):
        usage_out["prompt_tokens"] = prompt_tokens
    ch = r.get("choices") if isinstance(r, dict) else None
    finish = ch[0].get("finish_reason") if (ch and isinstance(ch[0], dict)) else None
    divinfo = (r.get("diverged"), r.get("diverged_at")) if isinstance(r, dict) else (None, None)
    return (ch[0].get("text", "") if ch else str(r)), [], finish, divinfo
