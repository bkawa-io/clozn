"""Request-local generation state and cancellation.

Each EngineSubstrate chat call creates and publishes one RequestContext. Generation metadata, trace,
termination, timing, prompt facts, and cancellation therefore move together as one coherent request
record; readers cannot observe fields combined from different calls.
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field


def new_request_id() -> str:
    """A short, log-friendly per-request id, minted the moment a generation call BEGINS -- unlike the runs
    store's run_<ts>_<hex> id (runs/store.py's record()), which only exists after a run is successfully
    persisted, well after generation finished. Distinct namespace (`req_` vs `run_`) so the two are never
    confused; correlates one gateway call with the worker's own per-request `req` stamp on native SSE
    frames (engine/core/serve/server_shared.hpp's StreamEnvelope) -- EngineSubstrate.chat_stream (see
    substrates.py) reads that `req` off the first parsed frame and stashes it on this call's
    RequestContext.engine_req, so routes/engine.py's POST /cancel can resolve a gateway `req_` id to the
    worker's own id before proxying the cancel (body key `req_id`, distinct from the `req` key that
    carries a raw worker id)."""
    return "req_" + secrets.token_hex(8)


@dataclass
class RequestContext:
    """One generation call's isolated state -- constructed fresh per chat()/chat_stream() call (see
    EngineSubstrate._new_request), then published onto the substrate as `self._request` in one shot. Never
    mutated by a SECOND call once a new one begins: a new call always builds and publishes its own instance,
    so a reference held by a caller (e.g. sse.py, to call .cancel() on the in-flight request) keeps
    describing THAT call even after a later one starts and replaces self._request with a different object.

    Fields (mirrors the backlog's "id, sampling, prompt manifest, trace, finish reason, cancellation"
    list exactly):
      request_id          -- new_request_id(), for log correlation.
      sampling             -- ctx._resolve_sampling()'s dict, or None for the greedy/forced-deterministic path.
      generation_meta      -- ctx._engine_generation_meta(...)'s reproducibility block.
      generation_timing    -- worker-measured aggregate decode timing from its terminal gen_finished
                              event; empty when that event was unavailable or the final reply was retried.
      gateway_timing       -- request-local gateway spans (template rendering/worker dispatch). These
                              use the gateway monotonic clock and are never aligned to worker offsets.
      prompt_manifest       -- a snapshot of mem_out once the call has finished mutating it
                              (assembled_messages/final_prompt, representing what was actually sent); the
                              live mem_out dict remains the primary,
                              already-per-call-isolated channel (callers own and read it directly) -- this is
                              an additional, consolidated copy on the context object for symmetry.
      trace                -- the per-token step list returned by last_stream_trace().
      finish_reason         -- the engine's stop cause, or None (missing reads as missing, never as "stop").
      finish_reason_raw     -- the worker event's pre-normalization stop cause, when available.
      diverged/diverged_at  -- the prove-all early-stop verdict returned by last_divergence().
      engine_req            -- the worker's OWN per-request id (StreamEnvelope's `req`, stamped on every
                              native SSE frame), captured off the FIRST frame chat_stream() parses -- or
                              None until that first frame arrives (nothing streamed yet, or a non-streaming
                              chat() call, which never sees native frames). This is the other half of the
                              request_id correlation new_request_id() describes: routes/engine.py's
                              POST /cancel resolves a gateway `req_` id to this field before proxying to
                              EngineClient.cancel(), so a cancel request never has to already know the
                              worker's own id scheme.
      prompt_tokens         -- the engine's own `gen_started` frame reports `prompt_tokens` verbatim
                              (engine/core/include/clozn/events.hpp); chat_stream() captures it off the
                              first such frame it parses, the same way it already captures `engine_req`.
                              None until that frame arrives (or on a substrate/test double whose stream
                              never emits one) -- retained as private worker accounting,
                              roadmap Phase 2 #1) to fill an honest `prompt_eval_count`, never fabricated.
      cancelled             -- a threading.Event; see cancel()/is_cancelled().
    """

    request_id: str = field(default_factory=new_request_id)
    sampling: dict | None = None
    generation_meta: dict = field(default_factory=dict)
    generation_timing: dict = field(default_factory=dict)
    gateway_timing: dict = field(default_factory=dict)
    prompt_manifest: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    finish_reason: str | None = None
    finish_reason_raw: str | None = None
    diverged: bool | None = None
    diverged_at: int | None = None
    engine_req: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    termination: dict = field(default_factory=dict)
    cancelled: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Signal that the client this request serves is gone. chat_stream's read loop checks
        is_cancelled() between worker frames and stops promptly instead of draining a reply nobody will
        receive; sse.py additionally calls gen.close() (throwing GeneratorExit at the generator's
        suspended `yield`) for an IMMEDIATE stop rather than waiting for the next frame boundary -- this
        Event is the belt to that close()'s suspenders, and the durable record that THIS request was
        cancelled (as opposed to merely having stopped) for anything that inspects the context later."""
        self.cancelled.set()

    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()
