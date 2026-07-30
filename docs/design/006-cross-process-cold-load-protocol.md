# ADR 006 — Cross-process cold-load, eviction, and coalescing protocol

Status: proposed; not implemented
Date: 2026-07-30

## Context

[ADR 004](004-multi-model-routing-contract.md) specifies cold loading, coalescing, and eviction as
part of the routing contract. RT-04 built it — `WorkerRegistry.ensure_loaded`
(`clozn/cli/worker_registry.py:943`) implements single-flight coalescing and idle-LRU eviction, and
`ProjectionFileRouter`/`PreloadedModelRouter` (`clozn/server/model_routing.py`) accept a `loader=`
callback that drives it (`ColdLoader = Callable[[str, float], ColdLoadOutcome]`, line 83). Both halves
are proven independently: `tests/test_worker_registry.py::test_ten_concurrent_cold_requests_cause_exactly_one_load`
(line 507) proves the registry's single-flight guarantee with real threads; `tests/test_projection_file_router_loader.py`
proves the router seam end to end through the real HTTP `do_POST` dispatch path, with the file's own
docstring calling out its flake-hunted acceptance test
(`test_ten_concurrent_http_posts_through_projection_file_router_cause_exactly_one_load`, line 500):
25 independently-rebuilt iterations × 10 concurrent threads = 250 real HTTP requests through the live
dispatch path per run, printed and asserted with zero tolerated failures (lines 513–616).

None of that is reachable in production. `clozn/server/app.py::main()` constructs `ProjectionFileRouter`
with no `loader=` argument (line 1326), and the comment at that construction site (lines 1310–1325)
states why: `clozn serve`'s gateway (`python -m clozn.server.app`) is a separate OS process, launched by
`subprocess.Popen` from `clozn/cli/runtime_process.py::_spawn_managed_runtime` (lines 497–512) — no
`fork`, no shared memory, no shared Python objects. A loader living in the gateway needs a live
`WorkerRegistry`, and the only live one is owned by the supervisor (the foreground `clozn serve` CLI
process, `clozn/cli/commands/serve.py::cmd_serve`, blocking in `RuntimeStack.wait()`,
`clozn/cli/runtime_process.py:389–410`). Verified by grep: `WorkerRegistry(...)` is constructed in
exactly two places — `clozn/cli/runtime_process.py:243,446` (the second one is the live registry;
the first, in `RuntimeConfig.__post_init__`, is validation-only, "no process starts here", same file
line 242) — and once more in `clozn/cli/managed_models.py:307`, whose own comment says why: "Let RT-02
remain the authority for cross-definition/default/preload invariants without starting any process."
Every mutating call (`start_preloaded`, `recover_failed`, `stop_all`) is grep-confirmed to originate
only from `clozn/cli/runtime_process.py`. `clozn/server` never imports `clozn/cli` anywhere in this
tree (grep for `clozn.cli` under `clozn/server` returns zero hits besides comments), and the converse
is also true — `clozn/cli` never imports `clozn/server`. This is a real, currently-clean one-way wall.

Building a loader inside the gateway process therefore means one of: (a) importing `clozn.cli` into
`clozn.server` — explicitly forbidden, stated at `clozn/models/__init__.py:4` ("the server must never
import clozn.cli") and re-stated at the `MODEL_ROUTER` construction comment in `app.py` (line 1318,
"forbidden -- see ColdLoadOutcome's docstring"); or (b) constructing a second, disconnected
`WorkerRegistry` inside the gateway process, which has no access to the model file paths/flags needed
to spawn a worker (only the supervisor's env-var handoff at boot gives the gateway `CLOZN_ENGINE_PORT`
and `CLOZN_MODEL_ROUTING_FILE` — see `_ENGINE_DISCOVERY_ENV_KEYS`'s comment in
`clozn/server/substrates.py:194–199`) and would spawn duplicate, desynchronized workers instead of
coalescing onto the supervisor's single-flight guarantee. `tests/test_model_routing_gateway.py`'s
`_loader_from_registry` docstring (line 845) already calls the real wiring "separately owned." This
ADR is that separately-owned decision.

**What the supervisor already publishes, one-way.** `RoutingProjectionTransport`
(`clozn/cli/runtime_process.py:140–207`) atomically publishes `WorkerRegistry.routing_projection()`
to a private temp-directory file (`os.replace` after `fsync`, lines 171–188); the gateway's
`ProjectionFileRouter._read_projection`/`refresh()` (`clozn/server/model_routing.py:1041–1101`)
re-reads it, comparing a SHA-256 fingerprint of the raw bytes to skip unnecessary rebuilds, on
(almost) every request via `_current()` (line 1103). This is supervisor→gateway only. The file's path
crosses to the gateway exactly once, via `CLOZN_MODEL_ROUTING_FILE`, set on the child's environment
before `subprocess.Popen` (`clozn/cli/runtime_process.py:494,506–512`) — env vars are a one-shot,
boot-time handoff, not a channel for anything ongoing.

**Two "busy" trackers exist today, and they don't know about each other.** `WorkerHandle.track_call()`/
`.busy`/`wait_until_idle()` (`clozn/cli/worker_handle.py:107–151`) is real, real-threaded, and unit
tested — and lives entirely in the *supervisor* process. `WorkerRegistry.track_call`
(`clozn/cli/worker_registry.py:828–841`) and the idle-LRU candidate filter
(`_select_eviction_candidate`, lines 900–915, `other.handle.busy`) already consult it correctly. But
grepped across the whole tree, `registry.track_call`/`handle.track_call` have **no caller outside
their own definitions and tests** — because today's actual generation dispatch never touches the
supervisor process at all: the gateway's `EngineSubstrate`/`EngineClient` talk to the private worker's
loopback port directly (`clozn/server/substrates.py`), bypassing the supervisor entirely. The gateway's
own, *separate*, already-wired-and-live per-worker busy tracker is `WorkerGateRegistry`
(`clozn/server/request_gate.py:131–223`, RT-05), used by `PreloadedModelRouter.select`'s
`gate.acquire_generation` (`clozn/server/model_routing.py:759–777`) on every real generation. So the
supervisor's `busy` signal is correct machinery wired to nothing, and the gateway's real traffic signal
lives in a process the supervisor cannot see into. Today this is harmless because cold-load-triggered
eviction (`_ensure_capacity`, `clozn/cli/worker_registry.py:917–941`) is itself unreachable — but it is
exactly the gap this ADR's backpressure section has to close before eviction can safely cross the
boundary. Section 5 below addresses this directly; it is a load-bearing finding, not a footnote.

**The cancellation-token claim needs a precise correction.** `docs/MANAGED_MODELS.md:262` states
"Cooperative cancellation cannot interrupt an in-flight private worker call. The worker protocol
carries no request ID for it," and the same sentence (nearly verbatim) appears in
`WorkerBusyError`'s docstring (`clozn/cli/worker_registry.py:70–76`) and `WorkerHandle.track_call`'s
(`clozn/cli/worker_handle.py:119–125`). Read literally, this is not quite what the code does.
`protocol/fixtures/handshake.json` pins `"stream_frame_envelope": ["req", "seq"]` — every SSE frame
the worker emits during generation *does* carry a worker-minted `req`. `clozn/server/routes/engine.py`'s
`POST /cancel` (lines 29–69) forwards that `req` (or resolves a gateway-side `req_id` to it via
`RequestContext.engine_req`, `clozn/server/request_context.py:106`) to `ctx.ENGINE.cancel(engine_req)`,
and `request_context.py:110–117`'s `RequestContext.cancel()` is real, live cancellation of the one
active generation stream. So "no request ID, ever, anywhere in the protocol" is too broad.

What *is* true, and is what actually matters for this ADR: that cancellation token (a) only exists for
the single **streaming generation** call — grepped across `clozn/server/substrates.py`, none of
`engine.score(...)`, `engine.jlens(...)`, `engine.health()`, checkpoint/restore/branch, or the
spawn+handshake sequence of a cold load itself carry or accept any per-call id, so none of them are
cancellable by any mechanism that exists; and (b) the token that *does* exist lives in **gateway**-process
memory (`RequestContext.engine_req`), not in the **supervisor**'s `WorkerHandle` — the process that
would need it to cancel something on eviction's behalf. So the practical conclusion the docs draw
("the registry never guesses here; it refuses instead of silently waiting") is correct, but the reason
is narrower and more precise than "no request ID exists." Section 6 carries this forward honestly.

## Decision

### 1. Who arbitrates

The supervisor's `WorkerRegistry` remains the **only** mutator of worker lifecycle state, unchanged
from today. Nothing in this ADR moves `ensure_loaded`, `_ensure_capacity`, `_evict_entry`, spawn, or
handshake qualification into the gateway process, and nothing gives the gateway a second, competing
copy of that state. The gateway's role is unchanged in kind, only extended in reach: today it *reads*
a point-in-time snapshot (the routing projection file) and fails closed on anything not already
resident; after this ADR it additionally *asks* the supervisor to act and waits for a typed answer. It
still never acts on worker lifecycle itself. Two owners is exactly the duplicate-spawn failure named
in `docs/MANAGED_MODELS.md`'s Limitations section and in this ADR's Context above; this decision does
not reopen that question, it answers the question left open by refusing it: *how does the gateway ask
without becoming an owner*.

Concretely, the gateway process gets a thin, pure client that speaks a small documented wire contract
and imports nothing from `clozn.cli`; the supervisor process gets a thin server that wraps the existing,
unchanged `WorkerRegistry` methods and translates real `LoadResult`/`WorkerBusyError`/exception outcomes
into that same wire contract. Neither side gains new authority — the client can only ask, and the only
new code the server side needs is translation, not decision logic. `WorkerRegistry.ensure_loaded`'s
existing single-flight/eviction/qualification behavior does not change at all.

### 2. The transport

Two calls need to cross the boundary, in **opposite directions**, and they should not share one
mechanism naively:

- **Gateway → supervisor: "load this model."** Rare (only on a cold model), latency-tolerant (a real
  spawn + handshake already takes seconds), and needs a genuine round trip with a bounded wait — this
  is exactly `ColdLoader`'s existing shape, `Callable[[str, float], ColdLoadOutcome]`.
- **Supervisor → gateway: "is anyone using this worker right now?"** Only needed at the moment the
  supervisor is about to evict a candidate to free capacity for a load (see section 5) — rare, and
  needs a quick, cheap, read-only answer.

**Recommended: a private loopback HTTP control endpoint on the supervisor, plus one new internal route
on the gateway's existing server.**

The supervisor (today single-threaded, blocking in `RuntimeStack.wait()`'s 0.25s poll loop) starts a
second `ThreadingHTTPServer` — the exact class the gateway itself already uses
(`clozn/server/app.py:36,1351`) — bound to `127.0.0.1` on a freshly allocated port
(`clozn/cli/engine_process.py:211`'s `_free_port()`, already used for worker ports), in a background
thread, before the gateway subprocess is spawned. Its address crosses to the gateway exactly the way
`CLOZN_ENGINE_PORT` and `CLOZN_MODEL_ROUTING_FILE` already do today — one more entry in the `env` dict
passed to `subprocess.Popen` (`clozn/cli/runtime_process.py:487–512`), e.g.
`CLOZN_SUPERVISOR_CONTROL_URL`. No new discovery mechanism is needed; the existing one-shot env-var
handoff already does this job for two other facts.

`POST /ensure_loaded {"model_id": "...", "timeout_s": ...}` on that server calls
`self.worker_registry.ensure_loaded(model_id, timeout=timeout_s)` directly and serializes the result.
This is nearly free: `LoadResult`'s own docstring (`clozn/cli/worker_registry.py:320–329`) already
states its field names "mirror `clozn.model-routing.v1`'s `LoadEvent` object exactly... so a caller can
copy them straight into a routing receipt" — the existing test adapter `_loader_from_registry`
(`tests/test_model_routing_gateway.py:845–872`) already does this translation for in-process callers;
the HTTP handler does the identical translation, just serialized to JSON instead of returned as a
Python object.

The reverse direction reuses a listener that already exists: the gateway's own `ThreadingHTTPServer`
(`clozn/server/app.py`) gets one new internal route, e.g. `GET /internal/worker-busy/<model_id>`,
backed directly by `MODEL_ROUTER.gate.snapshot()` (`WorkerGateRegistry.snapshot()`,
`clozn/server/request_gate.py:221–222`, already returns exactly `{"active": int, "waiting": int, ...}`
per worker key — no new state to build, only a route to expose it). This needs no new listener at all
on the gateway side.

Both contracts should get the same treatment `protocol/SPEC.md` + `protocol/fixtures/handshake.json`
already establish as this codebase's precedent for a pinned wire contract: a versioned JSON shape,
a fixture both sides can be tested against, and a test that fails the moment either side drifts
(`clozn/protocol.py`'s docstring names the exact same three-way pin for worker protocol 1.1). Unlike
protocol 1.1, both sides here are Python in the same repository, so no cross-language pin is needed —
but "supervisor-control.v1" (name TBD at implementation time) should still be a single source of truth
both `clozn/cli` (real handler) and `clozn/server` (thin client) import, not two independently-typed
JSON shapes that happen to agree today.

**Alternative A — filesystem-mediated two-way request/claim, extending `RoutingProjectionTransport`.**
The gateway atomically writes a request file (same temp-write-then-`os.replace` pattern already used
for the projection, `clozn/cli/runtime_process.py:171–188`) naming a `model_id` and a request id; the
supervisor's existing poll loop lists that directory each tick and answers with a claim/result file.

This avoids a second listening socket, reuses a proven atomic-write primitive, and needs no new port.
But it does not avoid the real cost this ADR has to pay: the supervisor's poll loop still cannot call
`ensure_loaded` (which blocks for the full boot timeout, up to 180s per model by default) from its own
thread without stalling `maintain()`'s dead-worker-restart duties for that whole window — so this
alternative *still* needs a background thread/pool inside the supervisor, it just polls a directory
instead of listening on a socket to feed it. On top of that: there is no push, so both request pickup
and result delivery are bounded by whatever poll interval is chosen (shrinking it below 0.25s to keep
latency reasonable increases disk churn for every tick, not just cold-load ticks); "the supervisor
died" has no clean signal — a request file just sits unclaimed, and the gateway has to invent a
staleness timeout that cannot distinguish "supervisor is gone" from "supervisor's listing pass hasn't
run yet," where a refused TCP connection is unambiguous; and multiple gateway threads requesting the
same cold model need de-duplication logic invented on the supervisor's file-processing side, where
`ThreadingHTTPServer` dispatching concurrently into `ensure_loaded` gets that from code that already
exists and is already proven. Genuinely usable for a first version if "zero new listening sockets" is
weighted above low latency and clean failure detection, but not the recommendation.

**Alternative B — a second `WorkerRegistry` in the gateway process.** Not a real alternative; this is
the exact failure mode named in `docs/MANAGED_MODELS.md`'s Limitations section and restated in this
ADR's Context — duplicate, desynchronized spawns, no shared single-flight guarantee. Included only
because the prompt for this decision explicitly names it as the thing "who arbitrates" must rule out.

**Recommendation: the loopback HTTP control endpoint (Alternative, first option above).** It reuses a
transport pattern already proven twice in this exact codebase (gateway↔worker protocol 1.1; the CLI's
own `gateway_health`/`gateway_liveness` polling via `urllib.request`,
`clozn/cli/runtime_process.py:44–67`) instead of introducing a third transport family; failure modes
map directly onto ordinary socket semantics (refused / reset / timed out — see section 4); and, as
section 3 shows, the existing single-flight guarantee composes for free under a threaded HTTP server
in a way that a poll-bounded file exchange cannot cleanly claim without inventing its own dispatch
layer anyway.

### 3. Coalescing across the boundary

The single-flight guarantee already lives **entirely** inside `WorkerRegistry.ensure_loaded`, keyed on
`_WorkerEntry.condition` per `model_id` (`clozn/cli/worker_registry.py:943–1095`). Its own docstring is
explicit: "of any number of concurrent callers naming the same cold... model, exactly one becomes the
loader... and every other concurrent caller waits on that same attempt" (lines 946–954) — and this is
proven with real OS threads today (`tests/test_worker_registry.py:507`). This ADR does not need to
reimplement that guarantee; it needs to not accidentally defeat it. That is the whole reason the
transport must dispatch concurrently: ten concurrent gateway HTTP client calls, hitting a
`ThreadingHTTPServer` on the supervisor side, become ten concurrent supervisor-side threads calling
`ensure_loaded(model_id)` on the *same* entry — already proven to coalesce to one spawn, no new
synchronization required. A single-threaded transport (a naive poll loop processing one request per
tick, or a non-threaded HTTP server) would still avoid a *duplicate spawn* — but it would silently
serialize what today is proven-concurrent, quietly degrading the coalescing latency guarantee (nine
waiters getting their answer as soon as the one load finishes, not one-per-poll-tick) without
technically breaking correctness. This is a hard requirement on the transport, not an implementation
nicety.

On the gateway side, nothing changes: `PreloadedModelRouter._materialize_ready_binding`
(`clozn/server/model_routing.py:866–901`) already guards, via `self._upgrade_lock`, against N gateway
threads that all received the same successful `ColdLoadOutcome` each redundantly rebuilding an
engine/substrate pair — proven today by `tests/test_projection_file_router_loader.py`. That code does
not know or care whether the `ColdLoadOutcome` came from an in-process test double or a real HTTP call;
this ADR does not touch it.

**A correctness requirement this ADR does surface, that the naive design misses:** `RuntimeStack.wait()`
(`clozn/cli/runtime_process.py:389–410`) only republishes the routing projection file when its own
`before`/`after` diff around its own `maintain()` call detects a change (lines 396–401) — it has no way
to notice that a *different* thread (the new control-endpoint handler, running concurrently) mutated
the same `WorkerRegistry` in between ticks. If the control handler calls `ensure_loaded` and it
succeeds without also publishing, the on-disk projection stays stale until the next *unrelated* change
happens to trigger a republish — and when it does, `ProjectionFileRouter.refresh()`
(`clozn/server/model_routing.py:1070–1101`) will see a changed fingerprint and rebuild
`PreloadedModelRouter.from_projection(...)` from scratch, which would silently discard the in-memory
`_materialize_ready_binding` upgrade in favor of whatever the (stale) file said at that moment.
Concretely: **the control-endpoint handler must call `transport.publish(registry.routing_projection())`
itself, synchronously, immediately after every state-changing `ensure_loaded`/eviction outcome** — the
same discipline `recover_worker()` already follows today (`clozn/cli/runtime_process.py:367–374`,
`_publish_registry`). This is a small, specific requirement, but it is easy to miss and would produce
an intermittent "loaded model reverts to unloaded" bug if skipped.

### 4. Failure semantics

| Scenario | How it's detected | Typed outcome |
|---|---|---|
| Supervisor process not running, or control endpoint refuses the connection | `ConnectionRefusedError`/`URLError` at first connect — fails fast, before any of the timeout budget is spent | New internal distinction (`ColdLoadOutcome.failure_code = "supervisor_unreachable"`), surfaced to the client through the **existing** `model_load_failed` code (503, retryable) with a message naming the cause. A new *top-level* error code is a schema change ADR 004 already gates behind `clozn.model-routing.v2` ("Changing required fields... or error mapping requires clozn.model-routing.v2", ADR 004 line 170) — not proposed here. See versioning note below. |
| Load call reaches the supervisor but doesn't answer within `load_timeout_ms` | client-side socket timeout on the control HTTP call | existing `model_load_timeout` (504, retryable) — meaning unchanged |
| Gateway process dies while its own load-triggering request is in flight | nothing on the supervisor side changes: `ensure_loaded` runs to completion on a supervisor-owned thread regardless of the calling connection's fate, mirroring the already-established rule that a call already dispatched runs to completion (`WorkerHandle.track_call`'s docstring, `clozn/cli/worker_handle.py:119–125`) | no new state; the *next* gateway request (or a restarted gateway) for that `model_id` calls `ensure_loaded` again and finds the entry already `ready`/`failed`/still-`loading` (and coalesces onto it if still loading) |
| Supervisor process dies mid-load (not just the connection) | the gateway's connection either resets mid-call or the next call gets `ConnectionRefusedError` | same as row 1, `supervisor_unreachable` → `model_load_failed`. Cleanup of a possibly-orphaned, half-initialized worker process is a **pre-existing** gap — `clozn/cli` has no parent-death-signal or job-object mechanism today (grepped: no `PR_SET_PDEATHSIG`, no Windows job object, no `atexit` child reaper) even for the single-model runtime — this ADR neither creates nor fixes it |
| Worker starts but fails its handshake | unchanged: `_qualify_handshake` (`clozn/cli/worker_registry.py:521–630`) already exhaustively types this | existing `worker_identity_mismatch` (502, not retryable) — the control-endpoint handler only needs to forward `outcome.failure_code` verbatim, which `PreloadedModelRouter._cold_load` already maps correctly (`clozn/server/model_routing.py:846–852`) |
| Registry has no evictable idle worker to free capacity | unchanged: `_ensure_capacity` already returns the typed string `"no_evictable_worker"` (`clozn/cli/worker_registry.py:917–941`) | existing `no_evictable_worker` (503, retryable) |

Every row either reuses an ADR 004 code unchanged, or maps a new *internal* distinction onto an
existing code's `message`/`failure_code` detail rather than widening the closed `oneOf` in
`clozn/schemas/defs/clozn.model-routing.v1.json`. That is a deliberate, conservative choice: it keeps
this ADR's protocol entirely inside v1's existing contract. A future `v2` could promote
`supervisor_unreachable` to its own top-level code (useful for operators who want to alert on "the
supervisor is down" differently from "the model failed to load") — not proposed here, and not needed
for correctness.

### 5. Backpressure and the resident limit

16GB VRAM is real, and `_ensure_capacity` (`clozn/cli/worker_registry.py:917–941`) already refuses to
exceed `max_loaded_workers`, evicting the least-recently-used *idle* resident first via
`_select_eviction_candidate` (lines 900–915). The Context section above states the actual gap plainly:
that candidate filter already checks `handle.busy`, but nothing populates `busy` today, because real
generation traffic never touches the supervisor process — it flows gateway↔worker directly. Wiring
cold-load requests through the supervisor does not, by itself, fix this: the supervisor could now
correctly decide *when* to try evicting something, but it still has no ground truth for whether the
candidate it's about to stop is mid-generation, because that truth lives in the gateway's
`WorkerGateRegistry` (`clozn/server/request_gate.py`), not in the supervisor's `WorkerHandle`.

The design in section 2 already carries the fix: before finalizing an eviction chosen by
`_select_eviction_candidate`, the supervisor calls the gateway's new
`GET /internal/worker-busy/<model_id>` and only proceeds if the gateway reports zero active and zero
waiting requests for that worker. If the gateway is unreachable at that moment (itself informative —
something is already wrong), the supervisor fails closed exactly as `no_evictable_worker` already does
today for "no idle candidate exists" — never guesses idle. This preserves ADR 004's existing wording
verbatim: "A worker with an active generation, an admitted queue entry, or an active mutation is not
evictable" (`docs/design/004-multi-model-routing-contract.md:102`) — the only change is *which
process's records* answer that question.

This reverse-direction call is deliberately placed on the **rare** path (eviction, which only happens
when the registry is already at capacity and a new cold load needs room) rather than the **hot** path
(every generation request). An earlier design that had every gateway-side `track_call`/`release` phone
the supervisor was considered and rejected: it would add a cross-process round trip to every single
generation, for a fact (worker busy-ness) the supervisor only ever needs to know at the rare moment it
is about to evict something.

### 6. Cancellation, honestly

The corrected claim from Context: no protocol verb exists to cancel a non-generation private-worker
call (score, harvest, intervene, checkpoint/restore/branch, or the spawn+handshake sequence of a cold
load itself), and the one cancellation token that does exist (the worker-minted `req` on a generation
stream, `protocol/fixtures/handshake.json`'s `stream_frame_envelope`) lives in gateway-process memory
(`RequestContext.engine_req`) and is reachable only by the gateway's own `POST /cancel`
(`clozn/server/routes/engine.py:29–69`) — not by the supervisor.

This ADR does not change that. Specifically:

- It does not add a request id to the cold-load spawn/handshake sequence, and does not attempt to make
  an in-progress spawn abortable. There is no "un-spawn" primitive today, and inventing one is a
  larger, separate piece of engine-process work this ADR is not proposing.
- It does not route eviction through the gateway's existing generation-cancel path. Even if it could
  (by having the supervisor ask the gateway to cancel, rather than just report busy), that would still
  only cover the single active *generation* — not a mutation, checkpoint call, or handshake — so it
  would not actually let `_select_eviction_candidate` treat more workers as evictable; it would only
  let it evict *slightly* sooner in the one case (an in-flight chat completion) where a cancel already
  exists. Given section 5 already gives the supervisor a correct busy/idle signal without this, adding
  a partial, generation-only forced-cancel path here would be complexity without a matching payoff.
- The practical behavior is unchanged from today's documented rule: a worker with anything in flight —
  generation or otherwise — is not evictable; the caller either gets `no_evictable_worker` or the
  request queues for capacity. Cooperative cancellation of an *already-dispatched* private-worker call
  remains a capability this system does not have, for any call type, from either process. If a future
  version of the worker protocol added per-call request ids to *every* call type (not just generation)
  as an explicit prerequisite, cancellation-driven eviction would become possible — that prerequisite
  is out of scope here and not assumed.

### 7. Out of scope for a first version

- **Orphan-worker cleanup on ungraceful supervisor death.** Pre-existing gap (Context, section 4);
  unaffected by this ADR either way.
- **Forced cancellation of any in-flight private-worker call**, generation or otherwise, whether
  triggered by eviction or anything else (section 6).
- **Remote (non-loopback) exposure** of either new endpoint. Both bind `127.0.0.1` only, matching
  every existing precedent in this codebase (`docs/RUNTIME_SPLIT.md`: "Loopback is the supported
  deployment. Remote exposure requires an explicit auth/TLS design.") — that design is not this one.
- **A new top-level `clozn.model-routing.v1` error code.** Section 4 deliberately reuses the existing
  closed `oneOf`; a dedicated `supervisor_unreachable` code is a `v2` question, not decided here.
- **Multiple concurrent supervisors, or a gateway reconnecting to a different supervisor mid-run.**
  There is exactly one supervisor per gateway process today (one `clozn serve` invocation, one
  `CLOZN_MODEL_ROUTING_FILE`, one control URL); this ADR does not generalize to N supervisors.
- **CLI-triggered explicit eviction** (a hypothetical `clozn evict MODEL`). `WorkerRegistry.evict()`
  (`clozn/cli/worker_registry.py:864–898`) already exists as a library method but is grep-confirmed to
  have no CLI command calling it today. This ADR only wires the *automatic* LRU-on-cold-load path
  already implemented inside `_ensure_capacity`; it does not add a new user-facing command.
- **Changing `WorkerRegistry.ensure_loaded`, `_ensure_capacity`, `_select_eviction_candidate`, or
  `_qualify_handshake`.** All are already correct and already tested; this ADR is purely about how a
  second process reaches them.

## How we would know this is working

The existing bar, precisely: `tests/test_projection_file_router_loader.py`'s
`test_ten_concurrent_http_posts_through_projection_file_router_cause_exactly_one_load` drives ten real
concurrent HTTP `POST`s through the real `do_POST` dispatch path, against a real (in-process)
`WorkerRegistry`, for 25 independently-rebuilt iterations (250 requests total), asserting `call_count
== 1`, a `9:1` coalesced/non-coalesced split read back from persisted run receipts, and zero tolerated
failures across all 25 iterations (lines 500–616). That is the standard for the *router* seam. It does
not, and cannot, prove the cross-process case: everything in it runs inside one Python process.

The cross-process equivalent needs a genuinely separate gateway OS process and a genuinely separate
supervisor OS process, which this repository already has a live-acceptance pattern for:
`scripts/smoke/managed_runtime_smoke.py` drives the real `clozn serve --models-config` / `clozn ps` /
`clozn stop` CLI boundary against real GGUFs and the real GPU engine build — "the same external process
boundary a user gets," per its own module docstring — and honestly degrades to a reported `SKIPPED`
(never a silent pass) when hardware or models aren't available.

A cross-process acceptance test for this ADR should extend that pattern with a scenario that:

1. Boots a managed gateway with one model in `preload_model_ids` and a second model present in
   `models[]` but deliberately **not** preloaded — genuinely cold, not reachable at all today per
   `docs/MANAGED_MODELS.md`'s Limitations section.
2. Fires ten concurrent real HTTP `POST`s at the live gateway's public port for the cold model, and
   asserts: exactly one worker process is spawned for it (observable via `clozn ps`/`GET /runtime/models`
   worker PID, or a single boot banner in the worker log); all ten HTTP responses succeed; all ten
   persisted run receipts' `worker_identity.worker_id` are identical (proving all ten were served by
   the one worker that came up, not ten redundant ones); the `coalesced` flag distribution is exactly
   one `False` and nine `True`, matching the existing in-process assertion shape
   (`tests/test_projection_file_router_loader.py:596`).
3. Repeats step 2 across multiple independently-rebuilt runs (matching the existing 25-iteration
   convention) — cross-process timing has genuinely different jitter (OS scheduling, TCP handshakes,
   real process spawn latency) than in-process threading, so the in-process flake-hunt does not
   transfer automatically and needs its own repetition budget.
4. Adds the failure-mode cases section 4 introduces and section 3's in-process tests cannot reach at
   all: `SIGKILL` the supervisor process mid-load and assert every in-flight and subsequent gateway
   request for that model gets a typed `model_load_failed` (never a hang, never a duplicate spawn,
   never a silently-served-by-default response — matching `test_unselected_run_engine_routes_never_use_default_worker`'s
   existing fail-closed standard in `tests/test_managed_model_bootstrap.py`); kill the gateway process
   mid-load and confirm the supervisor's load completes or fails on its own, and a fresh gateway
   process (or a retried request) observes the correct final state without redoing the spawn.
5. Confirms the eviction-safety query from section 5: with `max_loaded_workers` at capacity and a
   genuine in-flight generation on the LRU candidate, trigger a cold load for a different model and
   assert the busy worker is never stopped — the request either queues or returns
   `no_evictable_worker`, and the in-flight generation completes normally.

Until an acceptance test at this standard passes — real separate processes, real concurrency, flake-hunted,
with the failure-injection cases above — this remains a decision record, not a shipped capability, per
this ADR's own status line.

## Compatibility and versioning

This ADR changes no public API, no route, and no `clozn.model-routing.v1` receipt shape or error code.
Every failure case in section 4 is expressed through codes ADR 004 already defines; the new distinction
(`supervisor_unreachable`) is carried as internal detail (`failure_code`/`message`), not a new
top-level code, so no existing v1 receipt or client integration changes meaning. The new
supervisor↔gateway control contract (both directions) is an internal, unversioned-at-the-public-API-level
artifact — it should still be defined once, in a location both `clozn/cli` and `clozn/server` can import
without violating the existing one-way boundary, and pinned the way `clozn/protocol.py` already pins
worker protocol 1.1, so drift between the two sides fails a test rather than failing silently at
runtime. `docs/MANAGED_MODELS.md`'s Limitations section should be updated once this ships to state that
cold loading, eviction, and coalescing are reachable through `clozn serve` — not before.
