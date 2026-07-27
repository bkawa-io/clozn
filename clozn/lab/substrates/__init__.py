"""Model substrate adapters: PyTorch lab substrates -- the Qwen-7B (AR) workbench.

This package holds both halves of the lab's torch-loading surface:
  * the raw building blocks, one per submodule (qwen.py, self_teach.py) -- each importable on its own,
    torch-heavy internals lazily loaded inside their own functions/classes;
  * QwenSubstrate / _InternalizedRetrain, defined directly in this __init__ (folded in from the former
    clozn/lab/substrates.py -- a package and a same-named sibling module can't coexist in the same parent
    package; the package wins the import and the module becomes silently unreachable, so the fix is one
    file, not two), the classes that compose those building blocks into the shared Substrate contract
    (memory cards + tone dials) clozn/lab/app.py actually serves.

Relocated out of clozn/server/substrates.py so the product package physically cannot reach a Torch
adapter. QwenSubstrate is instantiated only by `clozn lab` (clozn/lab/app.py); the product gateway serves
EngineSubstrate exclusively. torch/transformers stay lazily imported inside methods (including the
submodule imports below -- self-referential but safe: they're all in-function, so by the time any of them
run, this package's own __init__ has already finished executing and is cached in sys.modules; there is no
circular import at load time), so importing this module is cheap -- but a product process never imports
it at all.

They still lean on the shared studio surface: the Substrate base (memory cards + tone dials) and the
prompt-memory / generation-meta helpers on clozn.server.app (`ctx`). That helper web is shared domain
code, not product-serving state; decoupling it further is a later step.
"""
from __future__ import annotations

import os
import threading
import time

from clozn.server.config import REPO_ROOT, DEMO                        # noqa: F401
from clozn.server import app as ctx                                    # shared prompt-memory + gen-meta helpers
from clozn.server.substrates import Substrate                          # the shared studio-surface base


_RETRAIN_INIT_LOCK = threading.Lock()   # guards the double-checked lazy init of each instance's retrain state


def _render_final_prompt(memory, messages):
    """Best-effort exact text twin of SelfTeach._chat_ids, for receipts and think-prefill detection."""
    try:
        rendered_messages = list(messages)
        if not memory._supports_system() and any(m.get("role") == "system" for m in rendered_messages):
            rendered_messages = memory._fold_system(rendered_messages)
        return memory.tok.apply_chat_template(
            rendered_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return None


class _InternalizedRetrain:
    """The internalized soft-prefix RETRAIN machinery, moved here VERBATIM out of clozn.server.app so the
    PRODUCT module carries none of it. Mixed into QwenSubstrate (before Substrate in the MRO, so this
    OVERRIDES the base's trivial prompt-only versions).

    Mutating a memory card in internalized mode retrains the soft-prefix via consolidate() -- ~4-5 min on
    the 4-bit 7B -- so we must NOT block the HTTP handler: the card STATUS flip (fast) stays synchronous
    and the RETRAIN runs on a daemon thread. Three PER-INSTANCE guards (created lazily so a bare
    object.__new__(QwenSubstrate) in tests, which skips __init__, never crashes on attribute access):
      _train_lock  -- held for the WHOLE consolidate(); this substrate's chat/generate paths acquire+
                      release it (self._train_lock) so a reply can't race the shared model+gradients
                      mid-retrain (they queue, they don't error).
      _retrain     -- the in-flight signal the UI polls: {active, card_id, action, started_at, error}.
      _retrain_meta -- guards reads/writes of the _retrain dict (a tiny critical section, distinct from
                      the long _train_lock); we don't launch a 2nd retrain while one runs."""

    def _ensure_retrain_state(self):
        if "_train_lock_" not in self.__dict__:
            with _RETRAIN_INIT_LOCK:
                if "_train_lock_" not in self.__dict__:
                    self.__dict__["_train_lock_"] = threading.RLock()
                    self.__dict__["_retrain_meta_"] = threading.Lock()
                    self.__dict__["_retrain_"] = {"active": False, "card_id": None, "action": None,
                                                  "started_at": None, "error": None}

    @property
    def _train_lock(self):
        self._ensure_retrain_state()
        return self.__dict__["_train_lock_"]

    @property
    def _retrain_meta(self):
        self._ensure_retrain_state()
        return self.__dict__["_retrain_meta_"]

    @property
    def _retrain(self):
        self._ensure_retrain_state()
        return self.__dict__["_retrain_"]

    def _retrain_status(self):
        """A snapshot of the in-flight retrain signal (copy -- never hand out the live dict)."""
        with self._retrain_meta:
            return dict(self._retrain)

    def _retrain_status_mode(self):
        """The retrain signal the UI polls, MODE-aware: prompt mode never retrains, so it reports a
        constant idle ({active: false, mode: "prompt"}); internalized reports the live flag."""
        if ctx._memory_mode() == "prompt":
            return {"active": False, "mode": "prompt"}
        return dict(self._retrain_status(), mode="internalized")

    def _retrain_in_flight(self):
        with self._retrain_meta:
            return bool(self._retrain["active"])

    def _join_retrain(self, timeout=None):
        """Block until no retrain is in flight (acquire+release _train_lock). Used by tests to await the
        background consolidate deterministically, and available for a graceful shutdown. Returns True once
        the lock was momentarily held with nothing active; False on timeout."""
        if not self._train_lock.acquire(timeout=timeout if timeout is not None else -1):
            return False
        try:
            return not self._retrain_in_flight()
        finally:
            self._train_lock.release()

    def _start_retrain(self, m, action, card_id, force=False):
        """Launch _mem_sync_rules(m) -- the SLOW consolidate() -- on a daemon thread and return immediately.

        PROMPT MODE short-circuits the whole machinery: the cards ARE the memory there, so a mutation only
        syncs m.rules (bookkeeping -- runlog + /state read it) and returns instantly. No consolidate, no
        _train_lock, no thread, no retrain banner; the trained prefix is left completely untouched (it
        stays internalized mode's artifact, preserved for a toggle back).

        Internalized: returns {retraining: True} once the thread is running, or {retraining: False} if
        there's nothing to do (the active set didn't move -- checked synchronously first, so a no-op
        transition never spins a thread) or a retrain is already in flight (we refuse to stack them, like
        _ensure_steer refuses a double compute). The worker holds _train_lock for the whole consolidate so
        chats serialize behind it, and clears _retrain on finish (success OR error) so the UI's poll always
        terminates. `force` skips the no-op pre-check AND forces the consolidate (the mode-switch catch-up:
        rules are synced but the prefix is stale)."""
        import clozn.memory.cards as memory_cards
        if ctx._memory_mode() == "prompt":
            r = ctx._mem_sync_rules(m, reconsolidate=False)          # instant: rules bookkeeping only
            return {"retraining": False, "changed": r["changed"], "mode": "prompt"}
        # cheap synchronous pre-check: would the active set actually change? if not, do NOT spawn a thread.
        if not force and list(getattr(m, "rules", []) or []) == list(memory_cards.active_texts()):
            return {"retraining": False, "changed": False}
        with self._retrain_meta:
            if self._retrain["active"]:                  # a retrain is already running -> don't stack a second
                return {"retraining": True, "busy": True, "queued": False}
            self._retrain.update(active=True, card_id=card_id, action=action,
                                 started_at=time.time(), error=None)

        def _work():
            err = None
            try:
                with self._train_lock:                   # hold across consolidate() -> chats wait, never race
                    ctx._mem_sync_rules(m, reconsolidate=True, force=force)
            except Exception as e:                        # a failed retrain must still clear the flag
                err = f"{type(e).__name__}: {e}"
            finally:
                with self._retrain_meta:
                    self._retrain.update(active=False, error=err)

        threading.Thread(target=_work, daemon=True).start()
        return {"retraining": True, "action": action, "card_id": card_id}


class QwenSubstrate(_InternalizedRetrain, Substrate):
    """One Qwen-7B + SAE behind the concept readout AND the memory + tone dials."""
    name = "qwen"

    def __init__(self):
        from clozn.readouts.brain import BrainReadout
        from clozn.readouts.sae7b import GpuSAE, load7b
        from clozn.lab.substrates.self_teach import SelfTeach
        from clozn.behavior.steering.hf_adapter import SteeringControl
        sae = GpuSAE()
        tok, model = load7b()
        self.brain = BrainReadout(model, tok, sae, DEMO, os.path.join(REPO_ROOT, "research"))
        self.memory = SelfTeach("Qwen/Qwen2.5-7B-Instruct", model=model, tok=tok,   # shares the model
                                persist_path=ctx._pers("studio_memory.pt"))
        self.steer = SteeringControl(model, tok)            # tone dials on the same model
        self._mem = self.memory
        self._steer_ready, self._steer_info, self._steer_lock = False, {}, threading.Lock()
        self._pers_steer = ctx._pers("studio_personality.json")
        self.steer.load_state(self._pers_steer)             # restore the personality dials across restarts
        self.steer.load_custom(ctx._pers(f"studio_custom_{self.name}.json"))    # + any user-defined dials
        # + the shipped dial library (research/deploy_dial_library.py), if it has ever been deployed on
        # this install -- a no-op (load_custom returns immediately) until that script runs. Loaded SECOND
        # so a rare --force-deployed name collision (a user custom + a library dial sharing a name)
        # resolves to the library's direction on every subsequent boot, matching what --force just did live.
        self.steer.load_custom(ctx._pers("studio_library.json"))
        if ctx._memory_mode() == "prompt":
            # PROMPT MODE boots from the CARD STORE (the prefix isn't applied): adopt the active-card
            # texts as m.rules right away so /state + runlog bookkeeping don't lag until the first
            # /memory call. sync_cards never touches the prefix; it also runs the one-time migration.
            self.memory.sync_cards()
            self._cards_migrated = True
            try:                                            # in prompt mode the strength dial persists in
                import clozn.settings as settings                                # the .pt needs a prefix to save; a
                s = settings.get_setting("memory_strength")               # fresh install has none
                if s is not None:
                    self.memory.memory_strength = max(0.0, min(2.0, float(s)))
            except Exception:
                pass

    def handle(self, path, body):
        if path == "/think":
            return self.brain.think(str(body.get("text", ""))[:500], str(body.get("sid", "default")))
        if path == "/concepts":                 # read what fired inside (no generation) -> annotate a reply
            return self.brain.concepts_only(str(body.get("text", ""))[:500])
        if path == "/say":
            with self._train_lock:                   # studio chat touches the shared model -> wait out a retrain
                # body["_trace_out"] / body["_mem_out"] (optional, server-side only): collectors the handler
                # passes for the Run Inspector trace + the per-turn memory record; never echoed to the client.
                if ctx._memory_mode() == "prompt":
                    return {"reply": self._say_prompt(body["message"], body.get("max_new", 200),
                                                      trace_out=body.get("_trace_out"),
                                                      mem_out=body.get("_mem_out"))}
                return {"reply": self.memory.say(body["message"], body.get("max_new", 200),
                                                 trace_out=body.get("_trace_out"))}
        if path == "/consolidate":               # a manual retrain -> the same shared-model lock as card retrains
            with self._train_lock:
                return self.memory.consolidate(body.get("rules"), body.get("steps", 120), body.get("lr", 0.012),
                                               body.get("n_probe", 8), body.get("max_norm", 14.0))
        if path == "/whatlearned":
            if ctx._memory_mode() == "prompt":
                return self._whatlearned_prompt()
            return {"report": self.memory.what_learned(), "mode": "internalized"}
        if path == "/check":                     # generates on the shared model -> wait out a retrain
            with self._train_lock:
                if ctx._memory_mode() == "prompt":
                    return self._check_prompt(body["prompt"], body.get("max_new", 200))
                return self.memory.check(body["prompt"], body.get("max_new", 200))
        if path == "/reset":
            with self._train_lock:                    # mutates the prefix/model state -> don't race a retrain
                self.brain.reset(str(body.get("sid", "default")))
                return self.memory.reset(body.get("keep_prefix", False))
        if path.startswith("/memory/"):
            return self._memory(path, body)
        if path.startswith("/steer/"):
            return self._steer(path, body)
        return None

    # ---- prompt-mode twins of the SelfTeach conversation endpoints ---------------------------------
    # Same surfaces, same shapes -- but memory rides as the gated system block and the model runs
    # prefix-free (use_prefix=False). SelfTeach itself is untouched: it stays the internalized-mode
    # engine and the research instrument (the self-audit experiments REQUIRE a non-text memory).

    def _say_prompt(self, message, max_new=200, trace_out=None, mem_out=None):
        """One /say turn in prompt mode: history grows exactly as SelfTeach.say, but the memory is the
        compiled block (topic-gated on THIS user turn). Runs under the caller's self._train_lock; takes
        m.lock like say() so concurrent history appends can't interleave."""
        m = self.memory
        with m.lock:
            m.history.append({"role": "user", "content": message})
            block, applied, gate = ctx._prompt_block_for(m, message)
            assembled = ctx._inject_block(m.history, block)
            if mem_out is not None:
                mem_out.update(mode="prompt", applied=applied, gate=gate,
                               prompt_block=block, assembled_messages=assembled,
                               final_prompt=_render_final_prompt(m, assembled))
            reply = m._generate(assembled, use_prefix=False, max_new=max_new,
                                sample=True, trace_out=trace_out)
            from clozn.runs.think_tags import sanitize_reply
            m.history.append({"role": "assistant", "content": sanitize_reply(reply).public_text})
            return reply

    def _whatlearned_prompt(self):
        """Prompt-mode /whatlearned: ask from a fresh context WITH the block injected, ungated (the
        self-view shows the full memory, mirroring what_learned's apply_gate=False). Honesty: in this
        mode the model is READING its cards out of context, not introspecting a trained prefix -- the
        `mode` field is there so the UI labels it as reading, not self-knowledge."""
        m = self.memory
        cards = ctx._prompt_mem_cards(m)
        if not cards:
            return {"report": "(no active memory cards yet -- add or approve one on the Memory page)",
                    "mode": "prompt"}
        import clozn.memory.mode as memory_mode
        block = memory_mode.compile_prompt_block([c["text"] for c in cards])
        ask = ("What have you picked up about me so far -- my interests, anything I seem to care about, "
               "and how I like you to respond? List what you know, one item per line.")   # == what_learned's
        with m.lock:
            report = m._generate([{"role": "system", "content": block}, {"role": "user", "content": ask}],
                                 use_prefix=False, max_new=200, sample=False)
        return {"report": report, "mode": "prompt"}

    def _check_prompt(self, prompt, max_new=200):
        """Prompt-mode /check, mirroring check()'s response shape: baseline vs block-in-context. The
        block is binary per turn, so `ungated` == block always in, and `gated` == what a real chat does
        (identical when the gate lets it in, the plain baseline when the topic gates it out -- greedy
        decode makes the reuse exact, no second generation needed)."""
        m = self.memory
        with m.lock:
            msgs = [{"role": "user", "content": prompt}]
            base = m._generate(msgs, use_prefix=False, max_new=max_new, sample=False)
            cards = ctx._prompt_mem_cards(m)
            if not cards:
                return {"prompt": prompt, "gate": None, "baseline": base, "mode": "prompt",
                        "ungated": "(no active memory cards)", "gated": "(no active memory cards)"}
            texts = [c["text"] for c in cards]
            import clozn.memory.mode as memory_mode
            block = memory_mode.compile_prompt_block(texts)
            g = round(ctx._prompt_gate(prompt, texts), 3)
            ungated = m._generate(ctx._inject_block(msgs, block), use_prefix=False, max_new=max_new, sample=False)
            gated = ungated if g >= ctx.PROMPT_GATE_MIN else base
            return {"prompt": prompt, "gate": g, "baseline": base, "ungated": ungated, "gated": gated,
                    "mode": "prompt"}

    def _gen(self, prompt):                     # AR generate for the /steer/check A/B
        return self.steer.generate(prompt, 90)

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        """One stateless chat completion with the WHOLE tunable self applied: the memory (as the trained
        prefix in internalized mode, as the topic-gated system block in prompt mode) AND the active
        tone-steering sliders, both on the shared model. This is what the OpenAI-compatible endpoint
        serves -- normal chat on the surface, legible and tunable underneath. Serializes behind an
        in-flight memory retrain (self._train_lock) so a reply can't race the shared model+gradients
        mid-consolidate -- it waits, briefly, rather than corrupting.
        trace_out (optional list): filled with the per-token trace for the Run Inspector; reply unchanged.
        mem_out (optional dict): prompt mode fills {mode, applied, gate} -- what memory ACTUALLY rode
        this turn -- so the run log records per-turn application, not just the active set."""
        with self._train_lock:                       # wait out any background retrain, then hold for this reply
            self._last_generation_meta = ctx._qwen_generation_meta(max_new, sample=sample, stream=False)
            self._last_finish_reason = None
            if self.steer.strength:             # persisted personality -> ensure vectors are ready (race-safe)
                self._ensure_steer()
            self.steer.engage()
            try:
                if ctx._memory_mode() == "prompt":
                    # PROMPT MODE: the cards ride as the system block (omitted when the topic gate says
                    # this turn is off-memory); the model runs prefix-free. The block wording is the
                    # exact distillation target the prefix trains toward, so behavior stays comparable.
                    block, applied, gate = ctx._prompt_block_for(self.memory, ctx._last_user(messages))
                    assembled = ctx._inject_block(messages, block)
                    if mem_out is not None:
                        mem_out.update(mode="prompt", applied=applied, gate=gate,
                                       prompt_block=block, assembled_messages=assembled,
                                       final_prompt=_render_final_prompt(self.memory, assembled))
                    reply = self.memory._generate(assembled, use_prefix=False,
                                                  max_new=max_new, sample=sample, trace_out=trace_out)
                    self._last_finish_reason = getattr(self.memory, "_last_finish_reason", None)
                    return reply
                # gate="auto" -> _generate scales the memory prefix by memory_strength x TOPIC RELEVANCE, so
                # the OpenAI /v1/chat path gets the same on-topic gating as /say (fixes the always-on
                # over-bleed). memory_strength 0 still zeroes it; a missing embedder falls back to no-gating.
                reply = self.memory._generate(messages, use_prefix=True, max_new=max_new, sample=sample,
                                              gate="auto", trace_out=trace_out)
                if mem_out is not None:
                    mem_out.update(mode="internalized", final_prompt=_render_final_prompt(self.memory, messages))
                self._last_finish_reason = getattr(self.memory, "_last_finish_reason", None)
                return reply
            finally:
                self.steer.disengage()

    def last_finish_reason(self):
        return getattr(self, "_last_finish_reason", None)

    def run_meta(self):
        return dict(getattr(self, "_last_generation_meta", None) or
                    ctx._qwen_generation_meta(sample=True))

    def last_stream_trace(self):
        """The per-token trace captured during the most recent chat_stream (raw step list, or []). The SSE
        handler reads this AFTER the generator is exhausted to log it -- streaming yields text, not tokens,
        so the trace is assembled from the recorder's rows + the generated ids, not from the chunks."""
        return list(getattr(self, "_last_stream_trace", []) or [])

    def chat_stream(self, messages, max_new=256, mem_out=None):
        """Streaming chat: yields text chunks as the AR model generates -- memory + tone steering
        applied -- via a TextIteratorStreamer with generate() in a thread. Local AR is slow, so this is
        the big UX win the diffusion side doesn't need (diffusion is trace-based, not left-to-right).
        mem_out: as in chat() -- prompt mode records what memory actually rode this turn.

        Per-token trace (B3): a pure pass-through RecordingLogitsProcessor rides along and the generated
        ids are captured from generate()'s return -> after the stream ends we assemble the trace into
        self._last_stream_trace for the SSE handler to log. Pass-through means the streamed chunks are
        byte-identical to before; the whole capture is wrapped so any failure just leaves the trace empty."""
        import threading
        import torch
        from transformers import TextIteratorStreamer
        self._train_lock.acquire()                   # serialize behind an in-flight retrain (released in finally)
        self._last_generation_meta = ctx._qwen_generation_meta(max_new, sample=True, stream=True)
        self._last_finish_reason = None
        if self.steer.strength:
            self._ensure_steer()
        m = self.memory
        if ctx._memory_mode() == "prompt":
            # PROMPT MODE: the gated system block replaces the prefix concat below -- it simply becomes
            # part of the chat template, so the streaming mechanics are untouched.
            block, applied, gate = ctx._prompt_block_for(m, ctx._last_user(messages))
            assembled = ctx._inject_block(messages, block)
            if mem_out is not None:
                mem_out.update(mode="prompt", applied=applied, gate=gate,
                               prompt_block=block, assembled_messages=assembled,
                               final_prompt=_render_final_prompt(m, assembled))
            e = m._embed(m._chat_ids(assembled))
        else:
            if mem_out is not None:
                mem_out.update(mode="internalized", final_prompt=_render_final_prompt(m, messages))
            e = m._embed(m._chat_ids(messages))
            if m.prefix is not None:                        # prepend the consolidated memory prefix, scaled by
                # memory_strength x TOPIC RELEVANCE (same on-topic gating as _generate's gate="auto"; this path
                # inlines generate() for streaming so it can't call _generate, but the scale must match). A
                # missing embedder makes rel==1.0 (no gating); memory_strength 0 zeroes the prefix entirely.
                last_user = next((mm["content"] for mm in reversed(messages) if mm.get("role") == "user"), "")
                g = m.memory_strength * m._gate(last_user)
                e = torch.cat([(g * m.prefix.detach()).to(e.dtype)[None], e], 1)
        att = torch.ones(e.shape[:2], device=e.device, dtype=torch.long)
        streamer = TextIteratorStreamer(m.tok, skip_prompt=False, skip_special_tokens=True)
        kw = dict(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new, do_sample=True,
                  temperature=0.7, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3,
                  pad_token_id=m.eos or 0, streamer=streamer)            # trim steering-induced loops
        self._last_stream_trace = []                        # reset; filled after the stream if capture succeeds
        recorder = None
        try:                                                # observe-only trace capture (never affects output)
            from clozn.lab.substrates.qwen import RecordingLogitsProcessor
            from transformers import LogitsProcessorList
            recorder = RecordingLogitsProcessor()
            kw["logits_processor"] = LogitsProcessorList([recorder])
        except Exception:
            recorder = None
        gen_out = {}                                        # holder so the thread can hand back generate()'s ids

        def _gen():
            with torch.no_grad():
                out = m.model.generate(**kw)
                try:
                    gen_out["ids"] = [int(t) for t in out[0].tolist()]   # inputs_embeds -> generated ids only
                except Exception:
                    pass

        self.steer.engage()                                 # tone dials apply during the streamed generation
        th = threading.Thread(target=_gen, daemon=True)
        th.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            th.join()
            raw_ids = list(gen_out.get("ids", []) or [])
            try:
                from clozn.lab.substrates.qwen import finish_reason_from_generated_ids
                self._last_finish_reason = finish_reason_from_generated_ids(raw_ids, m.eos, max_new)
            except Exception:
                self._last_finish_reason = None
            if recorder is not None:                        # assemble the trace from rows + emitted ids
                try:
                    from clozn.lab.substrates.qwen import steps_from_records
                    gen_ids = list(raw_ids)
                    while gen_ids and gen_ids[-1] == (m.eos or -1):
                        gen_ids.pop()
                    self._last_stream_trace = steps_from_records(recorder.records, gen_ids, m.tok)
                except Exception:
                    self._last_stream_trace = []
            self.steer.disengage()
            self._train_lock.release()                          # done streaming -> let a queued retrain proceed

    def state(self):
        return self.memory.state()
