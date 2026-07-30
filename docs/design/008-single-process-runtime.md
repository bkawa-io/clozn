# ADR 008 — Merge the gateway process into the supervisor process

Status: proposed; not implemented. The merge itself was decided by BK from a summary, not from this
code; this document is the requested design-before-the-change review, including a re-check of the
premise.
Date: 2026-07-30

## Context

`clozn serve` runs three processes today:

- **supervisor** — the `clozn serve` CLI process. Constructs `WorkerRegistry`, blocks in
  `RuntimeStack.wait()` (`clozn/cli/runtime_process.py:389-410`), restarts a dead `ready` worker via
  `WorkerRegistry.maintain()`/`_restart_entry` (`clozn/cli/worker_registry.py:770-814`), and owns process
  cleanup on exit (`clozn/cli/commands/serve.py:233-244`).
- **gateway** — `python -m clozn.server.app`, launched by `subprocess.Popen` from
  `_spawn_managed_runtime`/`spawn_runtime` (`clozn/cli/runtime_process.py:413-693`). Answers HTTP:
  OpenAI/Ollama compatibility, Studio, runs, receipts, steering.
- **worker(s)** — the C++ `clozn-server` binary, spawned by `clozn/cli/engine_process.py:381-382`. Binds
  a random loopback port, reachable only by the gateway. Not a second product server
  (`clozn/cli/runtime_process.py:1-6`'s module docstring).

This ADR is scoped to folding **gateway into supervisor**. The worker stays a separate OS process —
that is explicitly not in question, and nothing below proposes changing it.

**Motivation.** `docs/design/006-cross-process-cold-load-protocol.md` records that cold
load/coalescing/eviction (RT-04) are built and proven in-process
(`tests/test_worker_registry.py:507`, `tests/test_projection_file_router_loader.py`'s 25×10-iteration
flake hunt) but unreachable from `clozn serve`, because the loader needs a live `WorkerRegistry` and the
only live one is owned by the supervisor — a separate OS process from the gateway that would need one
(ADR 006 Context, `clozn/server/app.py:1310-1325`). ADR 006's own recommendation was a control-channel
protocol that keeps both processes and adds a wire contract between them. The owner's read of that
summary was: solve this by removing the boundary, not by building a protocol across it. §7 below checks
that conclusion against the actual code, per this document's brief.

## 1. What actually breaks

### 1.1 `RoutingProjectionTransport` / `ProjectionFileRouter`

`RoutingProjectionTransport` (`clozn/cli/runtime_process.py:140-207`) atomically publishes
`WorkerRegistry.routing_projection()` to a private temp-file (`os.O_CREAT|O_TRUNC` write, `fsync`,
`os.replace`, lines 171-188) whose path crosses to the gateway exactly once, via
`CLOZN_MODEL_ROUTING_FILE`, set before `subprocess.Popen` (line 494). `ProjectionFileRouter._read_projection`
(`clozn/server/model_routing.py:1041-1068`) re-reads that file and SHA-256-fingerprints the raw bytes;
`refresh()` (1070-1101) skips a rebuild when the fingerprint is unchanged. `_current()` (1103-1110) calls
`refresh()` on **every** `select()`/`catalog()`/`runtime_status()`/`control_pair()` call — i.e., on
(almost) every gateway request, this is a `os.path.getsize` + full file read + SHA-256 hash, even when
nothing changed.

In one process this file round-trip has no reason to exist for production traffic: the merged process
can hold one `WorkerRegistry` and one router object referencing it directly, and a state change
(`ensure_loaded`, eviction, `maintain()`'s restart) can call a plain Python method instead of
write-then-poll. **What must stay:** the class and the on-disk format themselves, as a test seam.
`tests/test_projection_file_router_loader.py`'s whole flake-hunted proof
(`test_ten_concurrent_http_posts_through_projection_file_router_cause_exactly_one_load`, ADR 006 line
17-19) constructs `ProjectionFileRouter` directly from a file it writes — it does not require a live
`WorkerRegistry` process at all, by design (that is what makes it possible to run in-process at 250
requests/25 iterations without spawning anything). Deleting the file-based path outright would force that
test (grep confirms `ProjectionFileRouter(` is constructed directly the same way in
`tests/test_managed_model_bootstrap.py` too) to be rewritten around a different construction seam, for no
production benefit. **Recommendation:
keep `RoutingProjectionTransport`/`ProjectionFileRouter` alive as a supported construction path (used by
those unit tests and by any future networked deployment), and add a second, direct in-memory router
construction for the merged production path** — not a replacement, an addition. This avoids relitigating
already-proven, flake-hunted test coverage as part of this migration.

### 1.2 The env-var boot handoff

Three env vars do one-shot, boot-time process-boundary crossing today:

- `CLOZN_ENGINE_PORT` — the worker's port (`runtime_process.py:493,626`; read at `clozn/server/app.py:89`).
- `CLOZN_MODEL_ROUTING_FILE` — the projection path (`runtime_process.py:494`; read at `app.py:1304`).
- `CLOZN_RUNTIME_KIND` — always `"product"` (`runtime_process.py:495,627`; `app.py:16-17`).
- `_ENGINE_IDENTITY_ENV`/`_ENGINE_DISCOVERY_ENV_KEYS` — six more identity facts
  (`runtime_process.py:34-41`, mirrored in `clozn/server/substrates.py:200-207`), whose own comment states
  plainly why they are env vars at all: *"this process (`clozn.server.app`, launched as `python -m
  clozn.server.app`) is a SEPARATE process from the `clozn serve` CLI ... so these facts have no other way
  to cross that boundary"* (`substrates.py:196-199`).

Every one of these exists **only because the two sides don't share memory**. Once they're one process,
every one of these becomes an ordinary function argument or a shared attribute — `_engine_discovery_context()`
(`substrates.py:210-221`) becomes unnecessary, `EngineClient(port=...)` at `app.py:90` can be constructed
directly with the port `WorkerHandle` already returned, and `main()`'s `routing_file = os.environ.get(...)`
branch (`app.py:1304-1330`) becomes "was a router passed in." **What replaces it:** direct call arguments
into whatever function starts the in-process HTTP server (§1.3) — no discovery mechanism is needed at all,
which is the entire point of merging. One caveat worth flagging: `os.environ.setdefault("CLOZN_RUNTIME_KIND",
"product")` at `app.py:16` runs at **import time**, module-level, before `main()`. A merged entry point that
imports `clozn.server.app` inside the same interpreter as the CLI must audit for this kind of import-time
side effect (there is at least one; there may be others in `clozn/server/config.py`, imported at
`app.py:30` with a comment already warning "side effects: sys.path/env/stdout").

### 1.3 `app.py::main()` and `make_handler()`

`make_handler(sub=None, subname=None, runtime_kind=None)` (`app.py:761-774`) already returns a plain
`BaseHTTPRequestHandler` subclass with no global side effects when given explicit `sub`/`subname` — this
is exactly the seam roughly 60 test files that call `cs.make_handler()` already use (grepped
`make_handler\(` across `tests/`). `main()` (`app.py:1293-1357`) is the part that assumes a separate process: it parses its own
`argparse` argv, requires `CLOZN_ENGINE_PORT` to already be in `os.environ` (line 1299-1300, `ap.error`
on missing), constructs `SUB`/`MODEL_ROUTER` as **module globals**, and blocks forever in
`srv.serve_forever()` (line 1353) with no return path except a fatal error.

For an in-process boot, `ThreadingHTTPServer(...).serve_forever()` (the identical class the gateway
already uses, `app.py:36,1351`) needs to run **without blocking the supervisor's own duties** — i.e., on a
background thread, with the supervisor's main thread free to either run `RuntimeStack.wait()`'s
0.25s `maintain()` poll loop as it does today, or (equivalently) put `maintain()` on its own background
thread and let the main thread block in `srv.serve_forever()` directly, mirroring exactly how
`ThreadingHTTPServer` already dispatches one thread per connection while `main()`'s own thread sits idle
in `serve_forever()`. Either arrangement is a **thread-based** merge, not `asyncio` — nothing in this
codebase uses `asyncio` today (`BaseHTTPRequestHandler`/`ThreadingHTTPServer` are synchronous, thread-per-
request), and introducing an event loop alongside `WorkerRegistry`'s condition-variable-based
single-flight coalescing (`worker_registry.py:943-1095`, real `threading.Condition`) would be new
machinery this ADR does not need and should not add. **Ownership of the HTTP server's lifetime**: the
supervisor process — `stack.stop()` (`runtime_process.py:333-341`) is the natural place to add
`srv.shutdown()` alongside the existing `worker_registry.stop_all()`/`routing_transport.close()` calls.

### 1.4 `clozn lab` and other constructors of their own substrate/handler

**This does not exist.** `make_handler`'s own docstring says *"`clozn lab` passes its own [sub/subname],
so it owns its handler + substrate WITHOUT reaching in to mutate this module's globals"* (`app.py:764-766`)
— but there is no `lab` subcommand. `clozn/cli/main.py` hand-wires 17 subcommands directly (grepped:
`run, serve, models, pull, plan, studio, smoke, ps, stop, trace, branch, explain, prove, inspect, test,
version, doctor`) and discovers the rest through `clozn/cli/commands/_autoload.py`'s opt-in scan (any
module setting `CLOZN_AUTOLOAD = True`, registered by `_autoload.register_all(sub)` at `main.py:274`) —
grepped for `CLOZN_AUTOLOAD` across `clozn/cli/commands/`, the autoloaded set adds `setup`, `adapter`,
`adopt`, `compare-runs`, `diff-adapter`, `investigate-experiment`-shaped names, `model-lock`, `snapshot`,
`triage`, `validate-export`, and a few more — **none named `lab`, in either list.**
`clozn/server/substrates.py:6` states the reason: *"The Torch lab adapters were deleted with the memory
program on 2026-07-27."* The docstring
at `app.py:764-766` is stale, describing a feature this repository no longer has. Correcting that comment
is a one-line cleanup independent of this ADR, but it means the brief's concern — "verify `clozn lab`
doesn't need its own accommodation" — resolves to "there is nothing to accommodate." Grepped across the
whole tree, every other caller of `make_handler(` is a test file (`tests/test_*.py`,
`tests/clients/fake_gateway.py`) constructing an in-process fake for its own HTTP-dispatch assertions —
none of them is a second production entry point, and none is affected by where `make_handler`'s *default*
path (no args, reads the module globals) gets called from.

### 1.5 Tests that depend on the subprocess boundary

Grepped for `spawn_runtime|_spawn_managed_runtime|gateway_python` across the whole tree: exactly 12 files
reference these names, and of those, exactly **three are tests** —
`tests/test_runtime_architecture.py`, `tests/test_managed_model_bootstrap.py`,
`tests/test_engine_artifact_identity.py` — matching the brief's claim precisely. The other nine split
into eight production files (`clozn/cli/runtime_process.py`, `clozn/server/substrates.py`,
`clozn/server/app.py`, `clozn/runs/identity_providers/engine_artifact.py`,
`clozn/cli/commands/{serve,run,explain,adopt}.py`) and this ADR's motivating doc
(`docs/design/006-cross-process-cold-load-protocol.md`).

But **none of those three test files actually exercises a real second OS process.** Each one monkeypatches
`runtime_process.subprocess.Popen` with a fake before calling `spawn_runtime`/`_spawn_managed_runtime`
(`test_runtime_architecture.py:307-323`, `test_managed_model_bootstrap.py:185-193`) — they assert on the
**constructed** `env`/`command` dict (e.g. `gateway_call[2]["env"]["CLOZN_ENGINE_PORT"] == "8456"`,
`test_runtime_architecture.py:338`) and on `RuntimeStack`'s bookkeeping, not on a live gateway responding
over a real socket. So these three files test *"does `spawn_runtime` build the right launch spec,"* which
a merged implementation will still need to prove (in a different shape — "does it configure the in-process
server with the right arguments" — but the same spirit), not *"is there a real process boundary."*
Extending the earlier grep to `subprocess.Popen|separate process|CLOZN_ENGINE_PORT|CLOZN_MODEL_ROUTING_FILE`
across `tests/` turns up two more files worth naming for completeness —
`tests/test_steer_concept_routes.py`, `tests/test_runs_store_concurrency.py`, `tests/test_engine_ctx_overflow.py`
— but each of those only *reads* `CLOZN_ENGINE_PORT` to decide whether to skip a live-engine test, unrelated
to the gateway/supervisor split.

**The only place a real, separate gateway OS process is actually exercised** is
`scripts/smoke/managed_runtime_smoke.py` (drives real `clozn serve --models-config`/`clozn ps`/`clozn stop`
against real GGUFs, per its own docstring, lines 1-38) and the `real-runtime-smoke.yml` CI workflow that
runs it (`ubuntu-24.04` only — grepped `runs-on:` across `.github/workflows/*.yml`; the product test suite
in `ci.yml` also runs exclusively on `ubuntu-latest`). **Windows — the primary dev box — and macOS have no
CI coverage of `spawn_runtime`/the process-boundary code at all today**, merged or not; only
`native-engine-release.yml`'s separate matrix (`windows-2025`, `macos-14`, line 78-87) builds the C++
engine binary itself on those platforms, not the Python runtime/supervisor logic. This is a pre-existing
gap this ADR does not need to fix, but the staged plan (§6) should not assume Windows/macOS behavior is
CI-verified anywhere it currently isn't.

## 2. The import direction

`clozn/server` importing `clozn/cli` remains forbidden, and nothing in this design needs it to change.
`clozn/models/__init__.py:4-6` states the rule directly: *"the server must never import clozn.cli."*
Grepped fresh in this tree: `clozn/server` has zero imports of `clozn.cli` (confirmed independently of
ADR 006's own grep, same result). The precedent the brief points to — `clozn/cli` importing `clozn/server`
— is real but narrower than "the boundary is already fully crossed": `clozn/cli/commands/doctor.py:77-78`
imports `clozn.server.config.DEMO` and `clozn.server.static.APP_INDEX`; `clozn/cli/commands/smoke.py:331`
imports only `clozn.server.static.APP_INDEX`. Both are static/config data, not the live route or handler
machinery. `clozn/cli/commands/quant_check.py:116,143` explicitly *avoids* importing `clozn.server.app`
"so this module never has to import clozn.server.app" — i.e., today's precedent is deliberately kept
shallow.

A merged entry point is a **deeper** instance of the same, allowed direction: it needs
`clozn.server.app.make_handler`, `EngineSubstrate`, and (per §1.1) `ProjectionFileRouter` or its in-memory
sibling — the live route/handler graph, not just config constants. This is still `clozn/cli → clozn/server`,
never the reverse, so it does not cross the forbidden wall. It is a real increase in coupling depth
relative to today's `doctor.py`/`smoke.py` precedent, and the merged entry point becomes the first place
in the repo where `clozn/cli` constructs and owns a live `clozn/server` HTTP handler — worth naming
explicitly in the module docstring of wherever this lands, so a future reader doesn't mistake it for
`clozn/server` reaching backward. **No blocking finding here**: the direction stays legal throughout.

## 3. THE ORPHANED-WORKER RISK

### 3.1 The threat model, precisely — and one correction to the brief

The brief frames the risk as *"an unhandled exception in an HTTP request handler can kill the process."*
That specific mechanism does not exist in this codebase, before or after the merge, and stating it that
way would misdirect whoever builds this. `ThreadingHTTPServer`/`BaseHTTPRequestHandler` (stdlib
`socketserver`) already catches and logs an exception raised inside one request thread without killing the
process — that has always been true, including today's separate gateway process. On top of that,
`clozn/server/app.py`'s own dispatch already wraps route bodies in `try/except Exception` returning a 500
(e.g. `app.py:1287-1288`, and the same pattern repeats throughout the route modules) — a second,
independent layer of protection against exactly the failure the brief names. **An ordinary Python
exception in a route handler was never capable of taking down the gateway process, and merging does not
change that.**

The real threat model — the one that *can* kill a process outright — is: a fatal signal (`SIGKILL`,
`SIGTERM` with no handler installed, Windows `TerminateProcess`/Task Manager "End Task"), a native crash
in a C extension on the request path (this process already touches `sqlite3`, `numpy`, `urllib`/sockets to
the worker, and the `clozn_engine` client SDK), an interpreter-fatal error, or an OOM-kill. None of these
are new to the merge — they already exist as the supervisor's own risk today (§3.2) — but merging expands
*which code's faults land in that category*, because far more code (the entire gateway's route/HTTP
surface) now shares the one process whose death is catastrophic. §4 develops this precisely; this section
is about the mitigation.

### 3.2 This risk is pre-existing, not introduced by the merge

Grepped across the whole tree (as ADR 006 already did, independently reconfirmed here): **no
`PR_SET_PDEATHSIG`, no Windows Job Object, no `atexit` child reaper exists anywhere in this codebase
today**, for either child process. `clozn stop`'s fallback — explicitly killing every PID in a runtime's
registry row, including `worker_pid` (`clozn/cli/commands/serve.py:308-333`, especially the comment at
320-322: *"Ask the supervisor to stop first ... Children are still signalled explicitly as a fallback for
a wedged/dead supervisor"*) — is the **only** existing mitigation, and it is opt-in: it only fires when a
human or script runs `clozn stop`, does nothing for a hard crash where the registry row is stale or the
process never gets the chance to run it, and cannot fire at all for the SIGKILL-supervisor case it names
as its own motivation. So **today, `SIGKILL`-ing the supervisor already orphans both the gateway and every
worker**, holding VRAM, invisible except via `clozn ps`'s liveness probe (`serve.py:247-268`, which
prunes a dead row lazily on next read) or manual `nvidia-smi`. Four stranded-GPU-process incidents on this
project (`docs/HANDOFF_2026-07-30.md` §6, last bullet) are the empirical cost of this exact gap, on the
research side (long-running scripts, not `clozn serve`), but the mechanism is the same.

**The merge does not create this risk from nothing.** It changes its shape in one way worth being exact
about: today, an unexpected **gateway** death is cleanly handled — `RuntimeStack.wait()` observes
`gateway.poll()` return a code (line 392-394), the loop returns, `cmd_serve` raises inside its own `try`,
and Python's `finally` (guaranteed to run regardless of the raised exception) reaches `stack.stop()`
(`serve.py:234-235`), which calls `worker_registry.stop_all()` (`runtime_process.py:336-337`) — a live,
still-running supervisor process deliberately tearing down every worker before exiting. **After merging,
this specific case — "the code that used to be a separate, killable gateway process crashes, and a
sibling process notices" — stops being a distinct scenario, because there is no longer a sibling.** A
fault that would previously have only killed the gateway subprocess (leaving a live supervisor to clean
up) now either (a) doesn't kill the process at all (the two-layer catch above, unchanged), or (b) kills
the *entire* merged process, which is exactly the pre-existing supervisor-SIGKILL scenario in the
paragraph above — just reachable from more code than before.

### 3.3 The guard, per platform

A parent-death guard closes the gap in §3.2 for the case that matters most in practice — the worker
processes outliving a dead parent — independent of whether the parent's death was graceful or a hard
crash. It hooks the **worker** spawn, a single call site: `clozn/cli/engine_process.py:381-382`'s
`subprocess.Popen(args, env=..., ...)`. Nothing about it depends on the merge; it is equally applicable
to today's two-process topology (where it would already have prevented some of the four stranded-GPU
incidents) and is explicitly worth landing before, and independent of, this ADR's main change (§6, Stage 0).

- **Linux — protected, pure Python, no engine/core change.** `prctl(PR_SET_PDEATHSIG, SIGKILL)`, called by
  the **child** on itself, tells the kernel to signal it when its parent thread exits. `subprocess.Popen`
  supports a `preexec_fn` callable that runs in the forked child, after `fork()` and before `exec()`
  (POSIX-only) — a `preexec_fn` that does
  `ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)` sets the flag on the
  not-yet-`clozn-server` child before `execve` replaces it, and the setting survives `exec`. No change to
  `engine/core` (the C++ binary) is needed. Caveat: PDEATHSIG is delivered when the *thread* that forked
  exits, not necessarily "the process" in every edge case (e.g. it can misfire if the forking thread exits
  while the process survives) — worth a smoke test, not a blocker.
- **Windows — protected, pure Python, one caveat.** A Job Object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` set via `SetInformationJobObject`, with the worker's process handle
  assigned via `AssignProcessToJobObject`, causes the OS to kill every assigned process the moment the job
  object's last handle closes — which happens automatically when the parent process exits, for **any**
  reason, including `TerminateProcess`/crash/Task Manager. This needs `ctypes` calls to `kernel32.dll`
  only (`CreateJobObjectW`, `SetInformationJobObject`, `AssignProcessToJobObject`) — no new dependency,
  consistent with this repo's stdlib-only ethos (`pyproject.toml:19-24`, discussed in §5). The one honest
  caveat: `AssignProcessToJobObject` needs a real process handle, and the stdlib's `subprocess.Popen`
  exposes one only via the semi-private `Popen._handle` attribute (there is no public API for this in
  `subprocess`) — every stdlib-only Windows job-object recipe in the wild relies on the same private
  attribute, or takes a `pywin32` dependency this project has never taken. Flag it as a known, small,
  accepted risk rather than a real dependency-free path — `_handle` has been stable across CPython's
  `_winapi` implementation for a long time, but it is not a contract.
- **macOS — not protected without an `engine/core` (C++) change; this ADR does not propose one.** There is
  no kernel primitive equivalent to `prctl`/Job Objects on Darwin. The only mechanisms that would work
  (polling `getppid()` — POSIX reparents an orphan to `launchd`/PID 1, exactly like Linux, so this check
  *would* be reliable there; or a heartbeat the worker requires from the supervisor before continuing to
  serve) both have to run **inside the process that needs to act on the loss** — i.e., inside the C++
  worker binary itself, since only the worker can decide to exit itself once orphaned. Nothing on the
  Python supervisor side can retrofit this the way `preexec_fn`/Job Objects do, because both of those work
  by asking the **OS**, not the child's own code, to act — Darwin has no equivalent OS-level hook. Adding
  either mechanism means editing `engine/core`'s C++ source (a `getppid()` poll thread or a
  `/heartbeat`-style protocol addition), which is explicitly out of scope for a design-only ADR with "no
  runtime refactor." **State plainly: after this ADR, macOS's orphan risk is unchanged from today — not
  worse, but not improved either**, and the gap is specifically an `engine/core` gap, not a Python-side one.

Windows (primary dev box) and Linux (nightly CI) — the two platforms this project actually operates and
verifies on — both get a real, OS-enforced fix with no `engine/core` involvement. macOS (owner-reported
only, and per §1.5, not CI-covered for this code path regardless) does not, and closing that gap is future
work this ADR names but does not schedule.

**This guard is worth adding whether or not the merge proceeds.** §3.2 already shows today's two-process
topology has the identical gap for a supervisor `SIGKILL`; the guard's implementation touches only
`clozn/cli/engine_process.py`'s worker-spawn call site and is fully decoupled from whether the gateway is
a separate process or not.

## 4. What is lost

Read `RuntimeStack.wait()` (`runtime_process.py:389-410`) as the literal statement of the supervisor's
watchdog contract today: every 0.25s, it (a) checks `self.gateway.poll()` — if the gateway exited, `wait()`
returns immediately with that code, no restart attempted; (b) otherwise calls
`self.worker_registry.maintain()` (or, on the legacy single-worker path, checks `self.worker.poll()` and
calls `self._restart_worker()`, `runtime_process.py:406-409`), which restarts any `ready` worker whose
process died unexpectedly (`worker_registry.py:805-814`), subject to a **bounded restart budget** — 3
restarts per 60-second window by default (`RuntimeConfig.restart_limit`/`restart_window`,
`runtime_process.py:223-224`; enforced in `WorkerHandle.restart()`, `worker_handle.py:168-179`, raising
`WorkerRestartLimitError` once exceeded, which `cmd_serve` surfaces as a fatal `CloznError`). **The
gateway itself is never restarted by anything in this codebase** — an unexpected gateway death is a fatal
event for the whole runtime today (§3.2), not a self-healing one. So the honest inventory of what a
"watchdog" currently does is narrower than the word suggests: it self-heals **worker** crashes only,
and cleanly (not silently) tears everything down on a **gateway** crash.

After the merge, both halves of that contract have to keep running *inside* the one process that also
serves HTTP, on a thread the HTTP-serving thread(s) cannot block or starve — `maintain()`'s poll loop
moves onto a background thread (§1.3), continuing to restart dead workers exactly as today, with the
identical restart-budget logic (`WorkerHandle`/`WorkerRegistry` are unchanged by this ADR; only their
caller's process boundary changes). **What genuinely disappears, with nothing replacing it in kind:** the
distinct "gateway subprocess crashed, a separate living supervisor process noticed and swept up" case
described in §3.2. There is no smaller-blast-radius sibling process left to notice and react — a fault
severe enough to kill "the gateway's" code now kills the process running the watchdog loop, the worker
registry, and the restart budget all at once. The parent-death guard (§3.3) recovers the **worker
cleanup** piece of that loss on Linux/Windows (dead parent ⇒ OS kills the worker), but recovers nothing
about the **watchdog/restart** role itself — once the merged process is gone, nothing restarts anything;
the operator re-runs `clozn serve`, which was already true for a gateway crash today (§3.2 — a gateway
crash was never self-healing, only cleanly torn down). Net: the practical loss is smaller than "we lose
crash isolation" implies (there was no self-healing to lose on the gateway side), but it is real and
specific — a strictly larger amount of code (every HTTP route, JSON parsing, SQLite, the worker HTTP
client) now shares fate with the one process whose sudden death is expensive to clean up after, on a box
where that cost is 16GB of VRAM. Do not read the parent-death guard as having solved this section; it
solves §3, not this one.

## 5. `gateway_python`

`RuntimeConfig.gateway_python: str = field(default_factory=lambda: sys.executable)`
(`runtime_process.py:220`) is read at exactly two call sites, both inside `runtime_process.py` itself
(lines 498, 643) — the first element of the `command` list handed to `subprocess.Popen`. Grepped across
the **entire repository**: these are the only three references to the name `gateway_python` that exist —
the field definition and its two reads. No test constructs a `RuntimeConfig` with a non-default
`gateway_python`, no doc mentions the name, no script overrides it. It is a real, currently-dormant degree
of freedom: nothing today exercises running the gateway under any interpreter other than the one running
the CLI.

**The brief's stated rationale for this field — "run the stdlib-only gateway in a clean venv while the CLI
has torch/transformers" — does not match the current codebase, and this is worth surfacing plainly.**
`pyproject.toml:19-24` states the actual current design: *"Deliberately stdlib-only: the product
CLI/gateway supervisor (this package) never imports Torch or transformers... `dependencies = []`."* This
is not scoped to the gateway alone — it is `clozn`'s entire top-level dependency declaration, covering
`clozn.cli` too. The `product-minimal` CI lane (`.github/workflows/ci.yml:64-94`) proves this for the
gateway import (`import clozn.server.app` with no torch installed, line 85) **and separately runs
`python -m unittest tests.test_runtime_architecture tests.test_product_smoke` with no torch installed**
(line 94) — and `test_runtime_architecture.py` is the file that exercises `clozn.cli.runtime_process`,
`spawn_runtime`, and `RuntimeConfig` directly (§1.5). Grepped for `import torch` in that test file: zero
matches. So **`clozn.cli.runtime_process` — the supervisor's own module — is already proven torch-free in
CI today**, by the same lane that proves the gateway is. Torch appears in exactly four modules in this
repository (`clozn/behavior/steering/hf_adapter.py`, `clozn/lab/slotmem_qwen/store.py`,
`clozn/readouts/brain.py`, `clozn/readouts/sae7b.py`), none reachable from `clozn serve`'s own import
graph, none imported by `runtime_process.py` or `app.py`. **There is no current CLI/gateway dependency
split for `gateway_python` to protect** — the premise behind the field, as stated in the brief, is false
for this codebase as it exists today.

**What is actually given up, then, is smaller and more speculative than the brief implies:** a documented,
plumbed-through, but never-exercised ability to point the gateway at an entirely different Python
installation — useful in a hypothetical future where the split reappears (e.g., a `clozn.cli` command
gains a hard, non-lazy heavy dependency; or a packaging goal wants the gateway shipped in a smaller,
hermetic interpreter separate from the full CLI's toolchain), or for isolating the gateway's dependency
footprint from a user's system Python for reasons unrelated to torch at all (version pinning, a
constrained deployment target). None of that is real today. Merging forecloses it permanently — once the
gateway is a function call inside the same interpreter as the CLI, "run it under a different Python" stops
being expressible at all, by construction, not by policy. That is a real, permanent loss of an optionality
the code currently holds open at near-zero cost (one dataclass field with a sensible default) — but it is
optionality for a need that does not currently exist, not a live capability being actively relied on.

## 6. Staged implementation plan

A safe staging exists. Five stages, each leaving the tree green and each independently revertible except
the last.

| Stage | What lands | Gate |
|---|---|---|
| **0 — Parent-death guard** (independent of the rest; do this regardless of the merge decision) | `preexec_fn`-based `PR_SET_PDEATHSIG` on Linux and a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` on Windows, wired into the single worker-spawn call site (`clozn/cli/engine_process.py:381-382`); explicitly a no-op stub on macOS with a comment naming §3.3's gap | Existing product suite + `managed_runtime_smoke.py` unaffected; a **new** live test that `SIGKILL`s the supervisor process and asserts the worker process is also gone within a few seconds — runnable in CI on Linux (matches `real-runtime-smoke.yml`'s `ubuntu-24.04` runner); Windows verified locally only, matching today's existing coverage gap (§1.5) |
| **1 — In-process gateway construction, opt-in, additive** | A new function (e.g. `runtime_process.spawn_runtime_inprocess` or a new module) that starts `ThreadingHTTPServer` + `make_handler()` on a background thread inside the supervisor, wires `WorkerRegistry` directly (no env vars, no `RoutingProjectionTransport` file — §1.1/§1.2), and moves `maintain()`'s poll loop onto its own thread. The existing `subprocess.Popen`-based `spawn_runtime`/`_spawn_managed_runtime` stay completely intact and remain the default; the new path is reachable only behind an explicit flag (e.g. `RuntimeConfig.inprocess_gateway: bool = False`, or a CLI-only `--experimental-single-process`) | New tests mirroring `test_runtime_architecture.py`'s shape but against the in-process path, green; **zero** existing tests changed or deleted — the whole point of this stage is that nothing observable moves yet |
| **2 — Direct router wiring, cold load reachable** | The in-process path's `ProjectionFileRouter` construction is replaced with a direct `PreloadedModelRouter`/registry reference (or a thin in-memory equivalent), and a real `ColdLoader` is finally passed at the merged construction site — the actual payoff this ADR exists for. `RoutingProjectionTransport`/`ProjectionFileRouter` remain in the tree, used only by the tests named in §1.1 and by the (still-default) subprocess path | ADR 006's own acceptance bar, adapted: the flake-hunted concurrent-cold-load proof (`test_projection_file_router_loader.py`'s 25×10 pattern) reproduced against the in-process wiring; `docs/MANAGED_MODELS.md`'s Limitations section updated to say cold load is reachable, per ADR 006's own compatibility note |
| **3 — Flip the default** | `clozn serve` defaults to the in-process runtime; the subprocess path becomes the fallback (kept for one deprecation window, or immediately deletable if Stage 2's proof is solid — implementer's call). `clozn stop`'s multi-PID fallback (`serve.py:308-333`) simplifies: one fewer PID class to track (gateway and supervisor share a PID now; only worker PIDs remain distinct). `product-minimal`'s CI assertion (`ci.yml:81-90`) should be extended to import the merged entry point itself, not just `clozn.server.app`, since there is no longer a separate-process boundary reinforcing the torch-free property — a future accidental heavy import in `clozn.cli` would now also break the gateway, where today it wouldn't | Full smoke battery (`managed_runtime_smoke.py`, `real-runtime-smoke.yml`) green against the new default; docs updated (`docs/RUNTIME_SPLIT.md`'s topology diagram, `docs/ARCHITECTURE.md`, `docs/MANAGED_MODELS.md`) |
| **4 — Cleanup (separate, later PR)** | Remove the dead `subprocess.Popen` gateway path, `gateway_python`, the env-var handoff constants, and any test left asserting a two-process boundary that no longer applies. This is the only irreversible stage — do it last, after a soak period, as its own reviewable change | Same product suite; a diff that is almost entirely deletions, easy to review in isolation from the stages that actually changed behavior |

Stage 0 has no dependency on the others and should land first regardless of what happens next. Stages 1-2
can be built and merged with the default behavior completely unchanged (an explicit flag, off by default)
— this is what makes the staging safe: nothing about production `clozn serve` moves until Stage 3, by
which point Stage 2 has already produced the flake-hunted, cross-process-turned-in-process proof this
project's own stated bar requires before calling a capability real (`docs/HANDOFF_2026-07-30.md`'s
"flake-hunt every concurrency test" lesson, §6).

## 7. Recommendation

**Build Stage 0 (the parent-death guard) immediately, independent of the merge decision — the evidence for
it stands on its own regardless of anything else in this document.**

**On the merge itself: the code supports proceeding, but not for the reason the brief's own framing would
suggest, and the margin is smaller than "merge because ADR 006's protocol is unnecessary overhead" implies.**
Here is the honest weighing:

- §3.1 already retracts the brief's stated threat model (ordinary handler exceptions were never fatal,
  before or after). The *actual* residual risk after Stage 0 lands is: a fatal signal/native
  crash/OOM-kill of the merged process orphans workers on macOS only — Linux and Windows, the two
  platforms this project actually runs and tests on, get a real OS-enforced fix with zero `engine/core`
  changes (§3.3).
- §4 shows the crash-isolation loss is real but narrower than it first appears: the gateway was never
  self-healing on its own crash (only cleanly torn down by a living supervisor) — merging trades "a small,
  simple sibling process cleanly notices and tears down" for "the OS-level guard from §3.3 tears down,"
  which is a *comparable*, not obviously worse, outcome on the platforms that matter, with the one honest
  cost being that strictly more code (the whole HTTP surface) now shares fate with the one process whose
  death is expensive.
- §5 removes the strongest-sounding argument against the merge as stated in the brief — `gateway_python`'s
  premise (a real current torch/no-torch split between CLI and gateway) is false today; what's actually
  forfeited is speculative future optionality, not a live need.
- Against all of that: ADR 006's control-channel alternative achieves the **same** unblock (a `ColdLoader`
  reachable from the gateway process) without touching the crash-isolation boundary at all, and without
  needing §3's guard as a prerequisite for the trade to be defensible (§3.3's guard is worth building
  regardless, but the merge's safety case leans on it in a way ADR 006's design never would need to). It
  costs a bespoke wire contract to design, version, and maintain forever (ADR 006 §2-§4) — real, ongoing
  complexity the merge avoids entirely by making the in-process case identical to what
  `test_projection_file_router_loader.py` already proves works.

**Net recommendation: proceed with the staged merge (§6), on the condition that Stage 0 lands and is
verified (the SIGKILL-supervisor test) before Stage 1 begins.** The evidence does not support the brief's
implicit worst case (a request-handler bug taking down GPU-holding workers), and does support that the
merge's real remaining cost — losing the gateway's independent crash-cleanup role — is a **narrower**
loss than advertised once the actual restart semantics (§4) are read precisely, and one Stage 0's guard
covers on both platforms this project actually verifies. **The strongest counter-argument, and the one
that should make the owner pause if VRAM safety is weighted higher than this document weights it:** on
macOS specifically, the merge's residual risk is not mitigated at all without an `engine/core` change this
ADR doesn't propose, and ADR 006's control-channel alternative has no equivalent gap on any platform — it
never removes the sibling-process boundary that makes "the gateway crashed, something else is still alive
to clean up" true everywhere, including macOS, with no OS-specific mechanism required at all. If macOS
support is meant to reach parity with Linux/Windows soon rather than staying owner-reported-only, that
tips the trade back toward ADR 006, or toward treating an `engine/core` heartbeat/`getppid()`-poll addition
as a co-requisite of this merge rather than optional future work.

## How we would know this is working

Mirroring this project's own stated bar (`docs/HANDOFF_2026-07-30.md` §6: *"flake-hunt every concurrency
test... the bar used here was 25+ consecutive runs"*):

1. Stage 0: a live test that starts a real worker under the guard, `SIGKILL`s the parent, and asserts the
   worker process is gone within a bounded window — run repeatedly (not once) on Linux in CI, and manually
   on Windows, before Stage 1 begins.
2. Stage 2: `test_projection_file_router_loader.py`'s exact 25-iteration × 10-concurrent-request pattern,
   reproduced against the in-process cold-load wiring, with the same `call_count == 1` / `9:1`
   coalesced-split assertions — this is the capability the whole ADR exists to unlock, and it should be
   held to the identical bar the in-process router proof already met, not a weaker one just because it's
   now "really" in-process.
3. Stage 3: the full `managed_runtime_smoke.py` battery green against the new default, plus a repeat of
   its existing "kill one worker's OS process, confirm independent supervisor-side recovery" scenario
   (script docstring point 4) to prove `maintain()`'s restart budget survived the move to a background
   thread unchanged.

Until Stage 2's flake-hunted proof passes, this remains a decision record and a partial implementation,
not a shipped capability — per this project's own convention for every prior ADR in this series.
