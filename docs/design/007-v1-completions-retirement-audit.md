# ADR 007 — `/v1/completions` retirement audit (M1/M2/M3)

Status: accepted — public route retired 2026-07-31 with a typed HTTP 410 migration response.
Date: 2026-07-30

## Decision (2026-07-31)

The public Python gateway route is removed as a generation surface. Requests to `POST
/v1/completions` receive HTTP 410 with `error.code = "endpoint_retired"` and are directed to
`POST /v1/chat/completions`. The private C++ worker's same-named loopback protocol remains unchanged;
it is still the generation primitive used by the gateway, CLI, and offline research tools. Historical
run readers remain unchanged because they do not branch on the route's write path.

## Scope

This was `docs/PRODUCT_ROADMAP.md`'s open item at the time of the 2026-07-30 audit, restated in
`docs/HANDOFF_2026-07-30.md:96`: "`/v1/completions` retirement — never done, compatibility route still
live." The audit compared the cost and risk of removal against the then-current instrumented route.
The implementation decision is now recorded above: choose the typed 410 retirement response and keep
the private worker protocol separate.

## Read this first: two surfaces share one path string, and they are not related

`/v1/completions` names **two independent HTTP endpoints in this codebase**:

| | Public gateway route (the retirement target) | Private worker protocol (NOT in scope) |
|---|---|---|
| Who serves it | `clozn/server/routes/openai.py`, the Python gateway | `engine/core/serve/server_main.cpp:2136-2137`, the C++ worker |
| Who calls it | External OpenAI-compatible clients | The gateway itself (`EngineSubstrate`), the CLI, and the entire mechanistic-interpretability research toolkit |
| Port | The public port a user gives `clozn serve --port` | A private loopback port only the supervisor/gateway knows |
| Journals a run? | Yes (§2) | No — it has no concept of Clozn's run journal at all |
| Retiring it | This ADR's subject | **Explicitly not proposed anywhere in this document** |

The collision is not superficial. `clozn/server/substrates.py`'s `EngineSubstrate.chat_stream()`
(lines 911-994) and `_engine_complete_traced()` (lines 1154-1239) — the function *every* public
generation surface bottoms out in, including `/v1/chat/completions` and the native
`/api/clozn/generate` — POST to `engine.base + "/v1/completions"` themselves. `clozn/cli/commands/
run.py:258` states this plainly in its own comment: `/api/clozn/generate` "is a transparent proxy
straight to the C++ engine's own `/v1/completions`". **The private worker's `/v1/completions` is the
generation primitive underneath the entire product, public-route-agnostic.** Retiring the *public*
route changes nothing about it: no client of the private endpoint (§4) is touched by anything this
document recommends.

Every claim below that says "the route" or "the endpoint" without qualification means the **public**
one, i.e. what an external OpenAI SDK client reaches at `http://<gateway>/v1/completions`. Every
citation into `engine/`, `clozn/server/substrates.py`, `clozn/server/generation_guard.py`,
`clozn/analysis/`, `scripts/tracer/`, `scripts/calibration/`, `scripts/bench/`, or `clozn-client/src/
clozn_client/engine.py` is called out explicitly as the *other* surface, for contrast.

## 1. Callers of the public route

### 1.1 Product code implementing it

- Route dispatch: `clozn/server/routes/openai.py:355-358` — a 4-line delegation (`try_post`), tiny
  compared to `/v1/chat/completions`'s ~340-line handler in the same file (lines 359-701). This
  asymmetry is itself informative: legacy completions never grew prompt-section influence, corrective
  policy beyond the base case, structured output, the guard, `clozn_trust`/`clozn_lens`/calibration
  policy verdicts, or selective-generation actions — none of chat completions' later feature work
  touched it.
- Handler: `clozn/server/generation_gateway.py`'s `openai_completion()` (780-841),
  `_completion_messages()` (580-588), `_completion_sample()` (591-600), `_stream_completion()`
  (603-778) — roughly 280 lines total.
- Request normalization: `clozn/server/openai_compat.py`'s `normalize_completion_request()` (327-365,
  39 lines) plus its own `COMPLETION_SUPPORTED_FIELDS`/`COMPLETION_NEUTRAL_FIELDS` constants
  (84-99ish). These are separate from `normalize_chat_request`'s constants; only small shared helpers
  (`_check_known_fields`, `_positive_int`, `_number_in`, `_fail`, …) are used by both, so removal has a
  small, contained blast radius inside this one file.
- Dispatch-gate membership: `clozn/server/app.py:76-81`, `_GENERATION_POST_PATHS` — the frozenset that
  tells `do_POST` to skip the pre-dispatch gate for this path because `select_for_handler` gates it
  per-worker instead (RT-05). `/v1/completions` is one of five members.
- CLI banner: `clozn/cli/commands/serve.py:201` prints `OpenAI text completions: POST {base}/v1/
  completions` on **every** `clozn serve` boot, managed or not.
- CLI acceptance gate: `clozn/cli/commands/smoke.py:195-213` (`_completion_stream_text`) and
  458-470 (the live SSE check itself, titled "OpenAI completion stream contains only standard
  chunks"). `docs/ARCHITECTURE.md:86-89` documents `clozn smoke MODEL` as checking "both protocol
  surfaces" — this is one of exactly two (the other is chat). This is a **real, currently-shipping,
  user-facing dependency**: every user who runs `clozn smoke` after a hypothetical removal, without a
  matching smoke-battery update, gets a failing acceptance gate on their own machine.
- Schema: `clozn/schemas/defs/clozn.model-routing.v1.json:43` — see §6, this is a hard constraint, not
  just a mention.
- Fixture: `tests/fixtures/schemas/clozn.model-routing.v1/invalid__routed_lifecycle_not_ready.json:3`
  uses `/v1/completions` as one example `route` value in an otherwise-invalid document. It is not
  meaningfully coupled to the route's existence — any valid enum member would do equally well as this
  fixture's example data.

### 1.2 Tests exercising the public route

| Test | What it proves | Cost class |
|---|---|---|
| `tests/test_legacy_completion_instrumented.py` (whole file, 5 tests, ~230 lines) | Real handler dispatch (`cs.make_handler()` + `do_POST()`), model-free fake substrate: strict OpenAI shape + journaling for streaming (`test_stream_completion_is_strict_openai_shape_and_journaled`), think-tag stripping, nonstream/stream failure journaling, client-disconnect handling | Tests the endpoint directly |
| `tests/test_openai_compat.py::test_completion_normalizes_extensions_and_neutral_legacy_fields`, `::test_completion_rejects_unsupported_shapes_and_behavior` (134-150) | Pure `normalize_completion_request` field-table unit tests, no HTTP | Tests the endpoint directly (cheapest to retire) |
| `tests/test_model_routing_gateway.py::test_legacy_openai_completion_is_not_a_default_worker_escape_hatch` (343-363) | A non-default-model request through `/v1/completions` resolves to the *named* worker, never the default, and the persisted routing artifact records `route: "/v1/completions"` | Depends on the endpoint to test something else (ADR-004 routing correctness) — the something-else (routing) is independently proven for chat/Ollama elsewhere, so this is a genuine, narrow migration cost, not a coverage gap |
| `tests/test_product_smoke.py`'s `SmokeGatewayHandler.do_POST` (98-106) | Fakes the **public** gateway's `/v1/completions` SSE response so `clozn smoke`'s client-side battery logic can be tested without a real model | Depends on the endpoint to test something else (the smoke harness itself) |

None of these four files' `/v1/completions` coverage requires a real network or a real model — the
actual migration cost is entirely in rewriting/deleting test code, not in losing acceptance evidence
that can't be reproduced another way.

### 1.3 Docs mentioning the public route (as opposed to describing it accurately — see §5 for the ones that don't)

`docs/CAPABILITIES.md:33`, `docs/ARCHITECTURE.md:47`, `docs/DEVELOPMENT.md:91`, `README.md:98`,
`docs/OPENAI_COMPATIBILITY.md` (the endpoint matrix row, the entire "Legacy Completions request
fields" section at 128-142, and the run-association prose at 150/158-160/209-213),
`docs/design/004-multi-model-routing-contract.md:130,133` (see §7 — this one is load-bearing, not
just a mention), `docs/PRODUCT_ROADMAP.md:252-255` (the "DONE" status line, which is about
instrumentation and would need a follow-up line for whatever M2 decides). `docs/DESIGN.md:233`
mentions it too, but that section is explicitly disclaimed as historical diffusion-serving design,
superseded by `OPENAI_COMPATIBILITY.md` per its own header note at lines 229-231 — no update needed
there regardless of M2's outcome.

## 2. Journaling and the run-id side channel

**Legacy completions journals every request, through the exact same pipeline chat completions uses.**
`openai_completion()` and `_stream_completion()` both call `instrumented_chat()`
(`clozn/server/generation_gateway.py:70-195`), which is the shared seam: "memory assembly, steering,
trace capture, finish-reason capture, and run journaling have already happened by the time a route
turns it into OpenAI- or Ollama-shaped JSON" (its own docstring, line 27-33). It calls
`handler._log_run(...)` (line 174 for non-stream, and inside `_stream_completion` at lines 715, 733,
743), which is `clozn/server/app.py:915`'s `_log_run`, which calls `clozn.runs.store.record` (line
938) — the one journal-write path every generation surface in the product shares. (I did not open
`clozn/runs/store.py` itself for this audit — two agents are concurrently working in `clozn/runs/`,
per instructions — but its call site, arguments, and the resulting run's readable shape are fully
visible from `clozn/server/app.py` and from tests that read runs back, which is what this section relies on.)

What distinguishes a completions-sourced run from a chat-sourced one:

- `meta.compatibility_api == "openai"` and `meta.openai_operation == "completion"`
  (`generation_gateway.py:625,814`; read back and asserted in
  `tests/test_legacy_completion_instrumented.py:168`). Chat completions never sets either key.
- `messages` is a synthetic single-user-turn list built by `_completion_messages()`
  (`generation_gateway.py:580-588`: `[{"role": "user", "content": prompt}]`) rather than the client's
  real conversation history.

**These tags are write-only.** Grepping all of `clozn/` for `compatibility_api`/`openai_operation`/
`text_completion` finds them only at the three write sites (`generation_gateway.py`,
`clozn/server/routes/ollama.py`, `clozn/server/ndjson.py` — the Ollama equivalents) and one unrelated
CLI smoke-battery check (`clozn/cli/commands/smoke.py:206`, checking the *live wire response*, not a
stored run). Nothing in the receipt (`clozn/server/routes/receipts.py`), replay
(`clozn/server/routes/replay.py`), influence-map (`clozn/server/routes/influence_map.py`), exact
execution-fork (`clozn/server/routes/execution_fork.py`), or run-investigation code paths branches on
where a run came from — they all operate on `run["messages"]`/`run["trace"]`/`run["response"]`/
`run["model"]` generically.

**Conclusion: retiring the route does not orphan any historical run.** An old completions-sourced run
reads exactly like any other run through every existing evidence surface. Nothing downstream needs to
change to keep old runs working — only the *write* path (the route itself) would go away. Old-run
READERS require nothing route-specific at all; this was the sharpest open question in the task brief
and the answer, from the code, is unambiguous.

**Run-id side channel, both directions, fully present today:**

- Non-stream: `clozn_run_id` body field + `X-Clozn-Run-Id` header
  (`generation_gateway.py:833-837`) — identical shape to chat completions.
- Stream: `clozn_run_id` **is** injected into the terminal `text_completion` SSE chunk
  (`generation_gateway.py:750`, `write_chunk("", public_finish, run_id=run_id, ...)`; also the
  fallback non-`chat_stream`-substrate branch at line 703). Confirmed by
  `tests/test_legacy_completion_instrumented.py:152`:
  `assert set(frames[-1]) == base_keys | {"clozn_run_id", "clozn_warnings"}`. This matches
  `/v1/chat/completions`'s own SSE behavior (`clozn/server/sse.py:215`, `terminal["clozn_run_id"] =
  rid`) exactly.

**Finding: a stale internal docstring.** `_stream_completion`'s own docstring
(`generation_gateway.py:611-616`) says: "Nor is a proprietary trailing chunk injected into this strict
legacy wire shape. The run is still persisted and can be associated through the server-side
latest-run/session side channel when that Phase-2 facility lands." This is **false** as the function's
own body and the passing test above demonstrate — the trailing chunk is injected, today, not pending
on a future "Phase-2 facility." Trust the code (and the test) over the comment; this is routine
documentation drift, not a functional gap, but it should be corrected as ordinary hygiene whenever
this file is next touched (out of scope for a docs-only audit to fix directly).

## 3. Conformance matrix and pinned-SDK lanes

Both CI lanes that exercise real released SDKs pin the exact versions the task named:
`.github/workflows/ci.yml:53` and `.github/workflows/native-engine-release.yml:51` both install
`openai==2.46.0` and `ollama==0.6.2`.

`docs/CLIENT_CONFORMANCE.md`'s full released-client matrix — OpenAI Python 2.46.0, Ollama Python
0.6.2, Ollama JavaScript 0.6.3, Aider 0.86.2, Open WebUI 0.10.2, the complete list of clients this
project holds itself accountable to — **contains no legacy-text-completions row or cell**. The OpenAI
Python row's "Non-stream text" column says plainly "Chat Completions executable test" (line 26). The
Open WebUI audit states directly (line 62): "its OpenAI-compatible path uses Chat Completions." I
verified this in the test code itself, not just the doc: `tests/test_openai_client_compat.py` and
`tests/test_openai_structured_client_compat.py` — the two files that drive the real `openai` package
against Clozn's real handler — contain **zero** calls to `client.completions.create`; every generation
call in both files is `client.chat.completions.create` (confirmed by grep: `\.completions\.create`
only ever matches as a substring of `.chat.completions.create`). `test_openai_client_compat.py` also
has no raw `urlopen` fallback that could be quietly hitting `/v1/completions` outside the SDK.

**No real, pinned, released client in this project's own compatibility story has ever exercised
`/v1/completions`.** Its only test coverage is: model-free hand-rolled HTTP-handler dispatch
(§1.2's first three rows) and one live SSE check inside Clozn's own `clozn smoke` acceptance gate
(§1.1) — never a third-party SDK.

## 4. The naming collision, exhaustively (the private worker protocol — out of scope)

For completeness, and to make the boundary in the box in the "read this first" section auditable, every
other first-party file matching `/v1/completions` calls the **private** engine endpoint directly, and is
unaffected by anything this document proposes:

- `clozn/server/substrates.py:911-994,1154-1239` — `EngineSubstrate`'s own generation primitive,
  underneath every public surface (§ above).
- `clozn/server/generation_guard.py:202-204,783-785,1019-1127` — the closed-loop guard's generation
  calls.
- `clozn/cli/commands/run.py:258`, `clozn/cli/commands/quant_check.py:196` — comments documenting the
  engine-level dependency, not calls themselves.
- `clozn/analysis/tracer.py:225,233`, `clozn/analysis/provenance.py:160` — the mechanistic-
  interpretability analysis library.
- Ten files under `scripts/tracer/` (`span_selection_validation.py`, `sae_fidelity_vs_concentration.py`,
  `sae_joint_vs_random.py`, `screen_null.py`, `molecules.py`, `provenance_battery.py`,
  `causal_trace_battery.py`, `edge_coalitions.py`, `attn_knockout_controls.py`, `attn_vs_causal.py`,
  plus `quant_regression_mine.py`'s docstring) and `scripts/calibration/guard_signal_calibrate.py`,
  `scripts/calibration/facts_efficacy_engine.py`, `scripts/bench/whitebox_tax.py` — the entire causal-
  tracing/SAE/calibration research toolkit talks to a bare `clozn-server` process's own
  `/v1/completions` directly, bypassing the Python gateway entirely. `docs/RESEARCH_ROADMAP.md:320`
  documents this explicitly: "GPU serialization: Only one agent can use the engine at a time for
  generation (`/v1/completions` is sequential)."
- `clozn-client/src/clozn_client/engine.py` — its own docstrings say what it is without ambiguity:
  "Explicit client for native/private engine research operations" (line 1), and the `EngineClient`
  class: "Direct native engine client, separate from the public Clozn gateway. Pass the native worker
  URL intentionally." (lines 31-35). Its `.complete()`/`.complete_chat()` methods (615-671) call the
  engine's own `/v1/completions`. I separately checked `clozn-client/src/clozn_client/gateway.py` — the
  package's actual *public*-gateway-facing client class — and it has **zero** references to
  completions of either kind.
- `clozn-client/tests/test_clients.py:126` — a fake private engine, for testing `EngineClient`.
- `tests/test_engine_chat_io.py:144-147,272-274`, `tests/test_engine_ctx_overflow.py` (whole file;
  its own module docstring: "regression for the `/v1/completions` decode-500 bugs"),
  `tests/test_product_smoke.py`'s `FakeWorkerHandler` (123-159) — all exercise or fake the private
  worker's own endpoint.
- `engine/core/serve/server_main.cpp:2136-2137` (the C++ route registration) and its header comment
  (lines 7-12), `engine/core/README.md:82-86`, `engine/client/clozn_engine.py` — the actual
  implementation and its documentation.

## 5. Doc-vs-code discrepancies found (trust the code)

1. **`docs/OPENAI_COMPATIBILITY.md:200-203` cites the wrong test file.** It attributes the claim
   "legacy text completions retain their exact raw prompt, decode metadata, token trace, finish
   reason, stable terminal/non-stream run ID, and one coherent journal record" to
   `tests/test_gate0_request_paths.py`. I read that file in full: it contains zero occurrences of the
   word "completion" and tests `clozn.cli.commands.run` (the `clozn run` CLI command's think-tag/journal
   hygiene against the native stream) — nothing to do with `/v1/completions`. The test that actually
   proves the cited claim is `tests/test_legacy_completion_instrumented.py` (§1.2, §2). This reads like
   a stale citation surviving a rename/split, not a real behavioral gap — but it should be corrected
   independent of whatever M2 decides.
2. **A stale internal docstring** — `generation_gateway.py:611-616`, detailed in §2 above.
3. **`docs/PRODUCT_STRATEGY_USER_NEEDS_2026-07-20.md:177,348` is out of date.** It states
   "`/v1/completions` currently journals nothing" (line 177) and lists "Legacy `/v1/completions` |
   Present but bypasses Clozn run instrumentation" (line 348) as a "High"-confidence audit finding.
   This directly contradicts current code and a passing test (§2). The date is the tell:
   `docs/PRODUCT_ROADMAP.md:254` records the instrumentation fix as "Status: DONE (2026-07-20)" — the
   *same day* this strategy document is dated. This document's finding predates (or was overtaken
   same-day by) that fix and was never updated afterward. **Do not treat this document's completions
   claim as current** — it is superseded by the code and by `PRODUCT_ROADMAP.md`'s own status line.

## 6. Schema constraint: the `model-routing.v1` route enum cannot be narrowed

`clozn/schemas/defs/clozn.model-routing.v1.json:43` closes the `route` field, when `surface ==
"openai"`, to exactly `{"enum": ["/v1/chat/completions", "/v1/completions"]}` — this is **not** a
free-form string, contrary to what I initially assumed before reading the schema file directly.

ADR 004's own compatibility rule (`docs/design/004-multi-model-routing-contract.md`, "Compatibility and
versioning") states: "Stored v1 routing receipts are immutable. Optional non-behavioral fields may be
added compatibly. Changing required fields, lifecycle meaning, runtime-key canonicalization, or error
mapping requires `clozn.model-routing.v2`; historical v1 receipts are never rewritten." Narrowing this
enum — removing `/v1/completions` as a valid value for an existing required field — is exactly the kind
of change that rule forbids inside v1: every historical routing receipt that recorded
`route: "/v1/completions"` while the route was live would fail re-validation against a narrowed schema.

**Concrete constraint for the retirement implementation: `clozn.model-routing.v1.json`'s enum must
keep `/v1/completions` as a permanently valid — but, after removal, permanently unproduced — value.**
This is cheap (zero cost: nothing has to change here at all) but easy to get wrong by "cleaning up" the
schema alongside the code, which would be a real, avoidable compatibility break for existing receipts.

## 7. Recommendation: removal vs. a typed 410

ADR 004 already anticipated this decision and partially specified it. `docs/design/
004-multi-model-routing-contract.md:130` hedges the OpenAI route list with "`/v1/completions` while it
exists," and line 133 states outright: "If `/v1/completions` is retired separately, its retirement
response occurs before routing." That sentence already rules out a design where a retired route falls
through to model selection/cold-load before answering — whatever M2 does, the typed response (of
whatever shape) must be the very first thing that happens, before `select_for_handler` runs.

There is also a live, shipped precedent in this exact codebase for "retire a POST route with a
permanent typed 410": `POST /substrate` already does exactly this
(`clozn/server/routes/health.py:129-132`, referenced by `_GATE_EXEMPT_POSTS`'s comment in
`clozn/server/app.py:49-50`: "`/substrate` — POST always 410s (routes/health.py); it never reaches the
substrate at all").

### Option A — outright removal (404)

**Work list:** delete the route dispatch (`openai.py:355-358`), the ~280-line handler
(`generation_gateway.py`), `normalize_completion_request` + its constants (`openai_compat.py`,
~55 lines), the `_GENERATION_POST_PATHS` entry (`app.py:78`), the CLI banner line (`serve.py:201`);
delete `tests/test_legacy_completion_instrumented.py` outright (~230 lines), delete the two
`test_openai_compat.py` tests, delete or repurpose `test_model_routing_gateway.py`'s routing test
(nothing left to route once the path 404s before reaching the router), delete
`_completion_stream_text` and its `clozn smoke` battery entry (`smoke.py`); update `CAPABILITIES.md`,
`ARCHITECTURE.md`, `DEVELOPMENT.md`, `README.md`, `OPENAI_COMPATIBILITY.md` (remove the whole section
and matrix row), `PRODUCT_ROADMAP.md`'s status line. Leave the schema enum untouched (§6).

**Cost:** roughly 330 lines of production code and 300 lines of test code deleted; 6-8 doc files
touched, mostly small deletions. **Risk:** any caller this audit didn't find — hypothetical, since
§1-4 found none among first-party code and §3 found none among released clients — gets a plain 404,
indistinguishable from "this was never implemented," with zero migration guidance.

### Option B — a permanent typed 410 with a migration message

**Shape:** keep the route registered; replace the ~280-line handler with a short (~10-line) responder
that, before touching `select_for_handler`/model routing at all, returns HTTP 410 with an
OpenAI-shaped error envelope (`{"error": {"message": "... use /v1/chat/completions instead ...",
"type": "invalid_request_error", "code": "endpoint_retired"}}`) for both stream and non-stream
requests. No run is journaled (a retired-route hit is not a generation attempt — nothing was
generated).

**Work list:** same deletions as Option A for `normalize_completion_request`+constants and most of
`_stream_completion`/`openai_completion`'s bodies (dead code either way), replaced by the small typed
responder; remove from `_GENERATION_POST_PATHS` (it no longer touches a worker) — likely add to
`_GATE_EXEMPT_POSTS` instead, following `/substrate`'s exact precedent; update the CLI banner to state
the route is retired (or drop the line); update `smoke.py`'s check to assert the 410 shape and message
instead of a working stream — `clozn smoke` keeps a meaningful, real assertion here rather than losing
coverage; **shrink** (not delete) `test_legacy_completion_instrumented.py` to assert the 410 envelope
for both stream/non-stream (roughly 2 tests replacing 5); **keep**
`test_model_routing_gateway.py`'s routing test in spirit, repurposed to assert the retirement response
fires *before* any routing/worker selection — this is now asserting a documented ADR-004 contract
rather than incidental behavior, arguably more valuable than before; same doc updates as Option A, but
`OPENAI_COMPATIBILITY.md` keeps a matrix row marked "retired (410, see migration message)" rather than
deleting it — useful for anyone who finds a stale blog post or bookmark referencing the old endpoint.

**Cost:** similar production-code reduction to Option A (~250 of the ~280 handler lines still go
away); *less* test churn (edits/shrinks three files instead of deleting two and gutting a third);
one extra small responder to write and keep correct. **Risk:** essentially none beyond Option A's,
and strictly better for the one thing that matters if an unmeasured caller exists — they get an
answer that says what happened and what to do next, not silence.

### Recommendation (accepted): Option B

Given (a) ADR 004 already specifies the retirement-response-before-routing behavior a 410 responder
implements almost for free, (b) `/substrate` is a live, working precedent for exactly this pattern in
this codebase, (c) §3 found zero real released clients depending on the route today so the cost
difference between the two options is small, and (d) a 410 costs only marginally more than a 404 to
build and *maintain* less test debt than Option A's wholesale deletions (repurposed tests instead of
new ones written from scratch later if a regression is ever suspected) — **a permanent typed 410 is
the better choice**. It is barely more expensive than outright removal and meaningfully kinder to
anyone this audit's evidence gathering did not — and, per §3, could not have — reached.

## 8. Implementation record (2026-07-31)

- Option B was implemented: the public route returns HTTP 410 with `endpoint_retired` before routing.
- The public validator, stream serializer, direct endpoint tests, smoke check, and compatibility docs
  were removed or updated.
- The v1 routing schema intentionally still accepts `/v1/completions` so historical receipts continue to
  validate; the route is permanently unproduced going forward.
