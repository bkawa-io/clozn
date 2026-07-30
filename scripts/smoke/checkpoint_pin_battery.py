"""checkpoint_pin_battery.py -- the live acceptance instrument for FORK-PIN-01 (durable checkpoint
export / import / truncate): POST /v1/checkpoint/export, /v1/checkpoint/import, /v1/checkpoint/truncate.

WHAT THIS PROVES
-----------------
1. EXPORT/IMPORT ROUND-TRIP EXACTNESS (the headline claim): generate a run with checkpoint_on_finish,
   resume it once to capture a baseline continuation, export the checkpoint, import the envelope back
   in (a NEW checkpoint_id -- the original is never touched again, simulating "the ephemeral one is
   gone"), resume the IMPORTED checkpoint the same way, and assert the continuation is token-for-token
   IDENTICAL to the baseline. A mismatch here is reported as a hard, explicit divergence -- never
   smoothed over.
2. CROSS-RESTART DURABILITY: kill the worker, spawn a fresh one (a genuinely different
   worker_generation_id, same model/build), import the SAME exported envelope into it, resume, and
   assert the SAME token-for-token match against the pre-restart baseline. Also confirms the STALE
   ephemeral checkpoint_id from the killed worker is honestly refused (404, not silently reused) on the
   new one -- the durable pin is what survived; the process-local id did not, correctly.
3. FAIL-CLOSED IMPORT: five deliberately corrupted envelopes (bad payload hash, bad model_sha256,
   bad architecture, mismatched declared kv_bytes, mismatched declared steer_dims) must each be
   REFUSED, never silently accepted or silently reprefilled.
4. TRUNCATE EXACTNESS: /v1/checkpoint/truncate to an EARLIER n_past (once inside the generated tail --
   the live-KV regime, once inside the prompt itself -- the reprefill regime), resumed and compared
   against the ORIGINAL run's own tokens from that same position. Also exports+imports the truncated
   checkpoint, proving "pin an earlier fork point" survives durability too, not only the tip.

HONEST DEGRADE
---------------
Needs a real engine + a real GGUF; neither is guaranteed present. Missing model/engine or a 404 on any
of the three new routes (this worker's build predates FORK-PIN-01) is reported as SKIPPED, exit 0 --
never a silent pass, never a FAIL for something this script cannot exercise.

Usage:
    python scripts/smoke/checkpoint_pin_battery.py
    python scripts/smoke/checkpoint_pin_battery.py --model C:\\...\\x.gguf --port 8098
"""
from __future__ import annotations

import argparse
import base64
import copy
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
ENGINE_CLIENT_DIR = os.path.join(REPO, "engine", "client")
if ENGINE_CLIENT_DIR not in sys.path:
    sys.path.insert(0, ENGINE_CLIENT_DIR)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
DEFAULT_PORT = 8098
DEFAULT_MAX_TOKENS = 40
DEFAULT_PROMPT = ("Write two sentences about a lighthouse keeper who finds something unusual washed "
                  "up on the shore one winter morning.")

_PREFERRED_FRAGMENTS = ["qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct", "qwen2.5-7b-instruct",
                        "meta-llama-3.1-8b-instruct"]


def _find_model(explicit) -> "str | None":
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    root = os.path.expanduser("~/.clozn/models")
    if not os.path.isdir(root):
        return None
    files = {os.path.basename(p).lower(): p for p in glob.glob(os.path.join(root, "*.gguf"))}
    if not files:
        return None
    for frag in _PREFERRED_FRAGMENTS:
        for name, path in files.items():
            if frag in name:
                return path
    from clozn.cli.commands.models import KNOWN
    for frag, _friendly, flags in KNOWN:
        if "mask" in flags or "eos" in flags:
            continue
        for name, path in files.items():
            if frag in name:
                return path
    return None


def _row(section, check, status, note) -> dict:
    return {"section": section, "check": check, "status": status, "note": note}


def _normalize_ids(tokens) -> list:
    out = []
    for t in tokens or []:
        out.append(int(t["id"]) if isinstance(t, dict) else int(t))
    return out


def _first_divergence(expected: list, got: list) -> dict:
    n = min(len(expected), len(got))
    for i in range(n):
        if expected[i] != got[i]:
            return {"offset": i, "expected": expected[i], "got": got[i]}
    if len(expected) != len(got):
        return {"offset": n, "expected": expected[n] if n < len(expected) else None,
                "got": got[n] if n < len(got) else None, "length_mismatch": True,
                "expected_len": len(expected), "got_len": len(got)}
    return {}


def _call(fn, *a, **kw):
    """Run an EngineClient call, returning (response, error_str) instead of raising -- so one bad
    case records a FAIL row instead of aborting the whole battery. Unlike execution_fork_battery.py's
    ForkProbe (which has to guess "route missing" from a reactive first-404, since it has no capability
    flag to check up front), main() below gates the ENTIRE battery on the 'checkpoint_pin' /health
    capability before any section runs -- so a 404 reaching here is always a genuine per-call failure
    (e.g. the deliberately-stale-checkpoint-id check), never "this build predates the route."""
    from clozn_engine import EngineError
    try:
        return fn(*a, **kw), None
    except EngineError as e:
        return None, str(e)


def generate_baseline(eng, prompt, max_tokens, rows) -> "dict | None":
    """Greedy generation with checkpoint_on_finish -- the ground truth every later comparison is
    judged against. Returns {checkpoint_id, worker_generation_id, prompt_tokens, board} or None."""
    try:
        original = eng.complete(prompt, max_tokens=max_tokens, checkpoint_on_finish=True, temperature=0)
    except Exception as e:  # noqa: BLE001
        rows.append(_row("setup", "generate baseline (checkpoint_on_finish=true)", FAIL,
                         f"generation request raised: {e}"))
        return None
    checkpoint_id = original.get("checkpoint_id")
    board = original.get("board") or []
    prompt_tokens = (original.get("usage") or {}).get("prompt_tokens")
    if not checkpoint_id or not isinstance(prompt_tokens, int) or len(board) <= prompt_tokens + 4:
        rows.append(_row("setup", "generate baseline (checkpoint_on_finish=true)", FAIL,
                         f"no usable checkpoint (checkpoint_id={checkpoint_id!r}, "
                         f"prompt_tokens={prompt_tokens!r}, board_len={len(board)})"))
        return None
    health = eng.health()
    rows.append(_row("setup", "generate baseline (checkpoint_on_finish=true)", PASS,
                     f"{len(board) - prompt_tokens} generated tokens, checkpoint_id={checkpoint_id}, "
                     f"worker_generation_id={health.get('worker_generation_id')}"))
    board = [int(t) for t in board]
    n_past = len(board)
    tail_len = n_past - prompt_tokens
    # An INTERIOR point within the already-generated tail (never the tip: resuming AT n_past would
    # generate brand-new tokens with nothing in `board` to compare them against -- there is nothing
    # "expected" past the end of what was already generated). Resuming from here is a live-KV bridge
    # decode + greedy continuation, which is DETERMINISTIC and must reproduce exactly what the
    # original continuous generation already produced at these same later positions -- the same
    # ground truth execution_fork_battery.py's own "interior" checks are judged against.
    interior = prompt_tokens + max(1, tail_len // 2)
    return {
        "checkpoint_id": checkpoint_id,
        "worker_generation_id": health.get("worker_generation_id"),
        "prompt_tokens": prompt_tokens,
        "board": board,
        "n_past": n_past,
        "interior": interior,
        "expected_tail": board[interior:],
        "remaining": n_past - interior,
    }


def resume_tail(eng, checkpoint_id, n_past, max_tokens, *, worker_generation_id=None):
    """execution_fork(intervention=none, truncate_to=n_past) -- the "continue exactly as-is" probe
    every round-trip/restart/truncate check is judged against."""
    return _call(eng.execution_fork, checkpoint_id=checkpoint_id, truncate_to=n_past,
                max_tokens=max_tokens, intervention={"type": "none"},
                worker_generation_id=worker_generation_id)


def judge_match(section, check, expected: list, resp, err, rows) -> bool:
    if err is not None:
        rows.append(_row(section, check, FAIL, f"request failed: {err}"))
        return False
    got = _normalize_ids(resp.get("tokens"))
    if got != expected:
        d = _first_divergence(expected, got)
        if d.get("length_mismatch"):
            rows.append(_row(section, check, FAIL,
                            f"DIVERGED (length): expected {d['expected_len']} got {d['got_len']} "
                            f"tokens (first differing offset {d['offset']}: "
                            f"expected={d['expected']!r} got={d['got']!r})"))
        else:
            rows.append(_row(section, check, FAIL,
                            f"DIVERGED at offset {d['offset']}: expected token_id={d['expected']} "
                            f"got token_id={d['got']}"))
        return False
    rows.append(_row(section, check, PASS, f"{len(got)} tokens reproduced exactly"))
    return True


def section_export_import_roundtrip(eng, baseline, rows):
    """Section 1+2: export/import bit-exactness, then cross-restart durability (needs a fresh
    worker handle from the CALLER once the restart happens -- see main())."""
    section = "export/import round-trip"
    interior = baseline["interior"]
    remaining = baseline["remaining"]
    expected = baseline["expected_tail"]

    baseline_resp, err = resume_tail(eng, baseline["checkpoint_id"], interior, remaining,
                                     worker_generation_id=baseline["worker_generation_id"])
    if not judge_match(section, "resume the ORIGINAL checkpoint at an interior point (ground truth "
                       "= the run's own already-generated tail)", expected, baseline_resp, err, rows):
        return None

    export, err = _call(eng.export_checkpoint, baseline["checkpoint_id"],
                        worker_generation_id=baseline["worker_generation_id"])
    if err is not None or not export or not isinstance(export.get("envelope"), dict):
        rows.append(_row(section, "export the checkpoint", FAIL, f"export failed: {err or export}"))
        return None
    envelope = export["envelope"]
    rows.append(_row(section, "export the checkpoint", PASS,
                     f"size_bytes={export.get('size_bytes')} envelope_bytes={export.get('envelope_bytes')}"))

    imported, err = _call(eng.import_checkpoint, envelope)
    if err is not None or not imported or not imported.get("checkpoint_id"):
        rows.append(_row(section, "import the envelope (never touching the original id again)", FAIL,
                        f"import failed: {err or imported}"))
        return None
    rows.append(_row(section, "import the envelope (never touching the original id again)", PASS,
                     f"new checkpoint_id={imported['checkpoint_id']} "
                     f"worker_generation_id={imported.get('worker_generation_id')}"))

    resp, err = resume_tail(eng, imported["checkpoint_id"], interior, remaining,
                            worker_generation_id=imported.get("worker_generation_id"))
    judge_match(section, "resume the IMPORTED checkpoint -- must match the unexported original",
               expected, resp, err, rows)
    return envelope


def section_fail_closed(eng, envelope, rows):
    section = "fail-closed import"

    def try_bad(label, mutate):
        bad = copy.deepcopy(envelope)
        mutate(bad)
        resp, err = _call(eng.import_checkpoint, bad)
        if err is None:
            rows.append(_row(section, label, FAIL,
                            f"import SUCCEEDED on a corrupted envelope (got checkpoint_id="
                            f"{resp.get('checkpoint_id') if resp else None!r}) -- this must be refused"))
        else:
            rows.append(_row(section, label, PASS, f"correctly refused: {err}"))

    try_bad("bad payload_sha256 (blob hash mismatch)",
           lambda e: e.__setitem__("payload_sha256", "0" * 64))
    try_bad("bad model_sha256 (wrong model identity)",
           lambda e: e["identity"].__setitem__("model_sha256", "f" * 64))
    try_bad("bad architecture string",
           lambda e: e["identity"].__setitem__("architecture", "not-a-real-architecture"))
    try_bad("mismatched declared kv_bytes",
           lambda e: e["state"].__setitem__("kv_bytes", e["state"]["kv_bytes"] + 1))
    try_bad("unsupported envelope_version",
           lambda e: e.__setitem__("envelope_version", "clozn.checkpoint-export.v99"))
    if envelope["state"].get("has_steer"):
        try_bad("mismatched declared steer_dims",
               lambda e: e["state"].__setitem__("steer_dims", e["state"]["steer_dims"] + 1))
    # A single flipped bit deep in the KV blob itself: still base64-decodable, still the declared
    # length, but the payload hash must catch it.
    def _flip_one_byte(e):
        raw = bytearray(base64.b64decode(e["state"]["kv_data_b64"]))
        raw[len(raw) // 2] ^= 0xFF
        e["state"]["kv_data_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    try_bad("one flipped byte deep inside the KV blob (length-preserving)", _flip_one_byte)


def section_truncate(eng, baseline, rows):
    section = "truncate exactness"
    board = baseline["board"]
    prompt_tokens = baseline["prompt_tokens"]
    n_past = len(board)
    tail_len = n_past - prompt_tokens
    if tail_len < 6:
        rows.append(_row(section, "setup", SKIP, f"only {tail_len} generated tokens -- too few"))
        return

    # Case A: truncate to a point INSIDE the generated tail (live-KV regime).
    interior = prompt_tokens + max(1, tail_len // 3)
    resp, err = _call(eng.truncate_checkpoint, baseline["checkpoint_id"], interior,
                      worker_generation_id=baseline["worker_generation_id"])
    if err is not None or not resp or not resp.get("checkpoint_id"):
        rows.append(_row(section, f"truncate to n_past={interior} (interior, expect live_kv_truncated)",
                        FAIL, f"truncate failed: {err or resp}"))
    else:
        restore_mode = resp.get("restore_mode")
        ok_mode = restore_mode == "live_kv_truncated"
        remaining = n_past - interior
        r2, e2 = resume_tail(eng, resp["checkpoint_id"], interior, remaining,
                             worker_generation_id=resp.get("worker_generation_id"))
        matched = judge_match(
            section, f"truncate to n_past={interior} (interior) then resume -- must match "
                    f"the original run's own tail from that point",
            board[interior:], r2, e2, rows)
        if not ok_mode:
            rows.append(_row(section, f"truncate to n_past={interior}: restore_mode", FAIL,
                            f"expected restore_mode='live_kv_truncated', got {restore_mode!r}"))
        elif matched:
            rows.append(_row(section, f"truncate to n_past={interior}: restore_mode", PASS,
                            f"restore_mode={restore_mode!r} as expected"))

        # Export+import the TRUNCATED checkpoint too -- pinning an earlier fork point, not just the tip.
        export, err = _call(eng.export_checkpoint, resp["checkpoint_id"],
                            worker_generation_id=resp.get("worker_generation_id"))
        if err is not None or not export or not isinstance(export.get("envelope"), dict):
            rows.append(_row(section, "export a TRUNCATED checkpoint", FAIL,
                            f"export failed: {err or export}"))
        else:
            rows.append(_row(section, "export a TRUNCATED checkpoint", PASS,
                            f"envelope_bytes={export.get('envelope_bytes')}"))
            imported, err = _call(eng.import_checkpoint, export["envelope"])
            if err is not None or not imported or not imported.get("checkpoint_id"):
                rows.append(_row(section, "import the TRUNCATED checkpoint's envelope", FAIL,
                                f"import failed: {err or imported}"))
            else:
                r3, e3 = resume_tail(eng, imported["checkpoint_id"], interior, remaining,
                                     worker_generation_id=imported.get("worker_generation_id"))
                judge_match(section, "resume the IMPORTED truncated checkpoint -- must still match "
                                    "the original run's tail", board[interior:], r3, e3, rows)

    # Case B: truncate to a point INSIDE the prompt itself (reprefill regime). There is no sense in
    # which resuming from here should reproduce board[boundary:] token-for-token: those are the
    # ORIGINAL PROMPT's own fixed text (never generated by the model at all) followed by a generated
    # tail that was conditioned on the FULL prompt, not a truncated one -- greedy continuation from a
    # SHORTER prompt has no reason to rediscover either. The real exactness claim for reprefill is
    # narrower: truncate_checkpoint's reprefill branch must build the SAME KV state an INDEPENDENT,
    # direct reprefill of board[:boundary] would -- so build that independently via POST /v1/checkpoint
    # (create_checkpoint) and assert the two resume to IDENTICAL continuations (the same
    # cross-construction proof pattern the export/import section uses).
    boundary = max(1, prompt_tokens // 2)
    remaining = n_past - boundary
    resp, err = _call(eng.truncate_checkpoint, baseline["checkpoint_id"], boundary,
                      worker_generation_id=baseline["worker_generation_id"])
    if err is not None or not resp or not resp.get("checkpoint_id"):
        rows.append(_row(section, f"truncate to n_past={boundary} (inside prompt, expect reprefill)",
                        FAIL, f"truncate failed: {err or resp}"))
        return
    restore_mode = resp.get("restore_mode")
    if restore_mode != "reprefill":
        rows.append(_row(section, f"truncate to n_past={boundary}: restore_mode", FAIL,
                        f"expected restore_mode='reprefill', got {restore_mode!r}"))
    else:
        rows.append(_row(section, f"truncate to n_past={boundary}: restore_mode", PASS,
                        f"restore_mode={restore_mode!r} as expected"))

    fresh, err = _call(eng.create_checkpoint, board[:boundary], n_past=boundary, prefill_to=boundary)
    if err is not None or not fresh or not fresh.get("checkpoint_id"):
        rows.append(_row(section, "independently reprefill board[:boundary] via POST /v1/checkpoint "
                                  "(cross-construction ground truth)", FAIL,
                        f"create_checkpoint failed: {err or fresh}"))
        return
    rows.append(_row(section, "independently reprefill board[:boundary] via POST /v1/checkpoint "
                              "(cross-construction ground truth)", PASS,
                    f"checkpoint_id={fresh['checkpoint_id']}"))

    r_trunc, e_trunc = resume_tail(eng, resp["checkpoint_id"], boundary, remaining,
                                   worker_generation_id=resp.get("worker_generation_id"))
    r_fresh, e_fresh = resume_tail(eng, fresh["checkpoint_id"], boundary, remaining,
                                   worker_generation_id=fresh.get("worker_generation_id"))
    if e_fresh is not None:
        rows.append(_row(section, f"resume the cross-construction ground truth at n_past={boundary}",
                        FAIL, f"request failed: {e_fresh}"))
    elif e_trunc is not None:
        rows.append(_row(section, f"truncate-then-resume at n_past={boundary} vs. independent reprefill",
                        FAIL, f"request failed: {e_trunc}"))
    else:
        judge_match(section, f"truncate-then-resume at n_past={boundary} vs. an INDEPENDENT direct "
                             f"reprefill of the same prompt prefix -- must match",
                   _normalize_ids(r_fresh.get("tokens")), r_trunc, None, rows)


def section_restart_durability(model, host, port, cpu, envelope, baseline, rows, *, terminate, spawn):
    section = "cross-restart durability"
    proc2 = None
    try:
        try:
            proc2, health2, gpu2 = spawn(model, port, prefer_gpu=not cpu)
        except Exception as e:  # noqa: BLE001
            rows.append(_row(section, "spawn a fresh worker (new generation)", FAIL,
                            f"could not start a second worker: {e}"))
            return
        new_generation = health2.get("worker_generation_id")
        if not new_generation or new_generation == baseline["worker_generation_id"]:
            rows.append(_row(section, "spawn a fresh worker (new generation)", FAIL,
                            f"expected a NEW worker_generation_id, got {new_generation!r} "
                            f"(was {baseline['worker_generation_id']!r})"))
            return
        rows.append(_row(section, "spawn a fresh worker (new generation)", PASS,
                         f"worker_generation_id={new_generation} (was {baseline['worker_generation_id']})"))

        from clozn_engine import EngineClient
        eng2 = EngineClient(host=host, port=port)

        # The STALE ephemeral id from the killed worker must be honestly refused, never silently
        # reused, against a worker that never issued it.
        resp, err = _call(eng2.execution_fork, checkpoint_id=baseline["checkpoint_id"], truncate_to=1,
                          max_tokens=1, intervention={"type": "none"})
        if err is None:
            rows.append(_row(section, "stale pre-restart checkpoint_id against the NEW worker", FAIL,
                            "the new worker accepted a checkpoint_id it never issued -- must 404"))
        else:
            rows.append(_row(section, "stale pre-restart checkpoint_id against the NEW worker", PASS,
                             f"correctly refused: {err}"))

        # The DURABLE pin resolves correctly on the new generation.
        imported, err = _call(eng2.import_checkpoint, envelope)
        if err is not None or not imported or not imported.get("checkpoint_id"):
            rows.append(_row(section, "import the pre-restart envelope into the NEW worker", FAIL,
                            f"import failed: {err or imported}"))
            return
        rows.append(_row(section, "import the pre-restart envelope into the NEW worker", PASS,
                         f"resolved as checkpoint_id={imported['checkpoint_id']} under the NEW "
                         f"generation {imported.get('worker_generation_id')}"))

        resp, err = resume_tail(eng2, imported["checkpoint_id"], baseline["interior"],
                                baseline["remaining"],
                                worker_generation_id=imported.get("worker_generation_id"))
        judge_match(section, "resume on the NEW worker -- must match the PRE-RESTART baseline "
                            "(the run's own already-generated tail)",
                   baseline["expected_tail"], resp, err, rows)
    finally:
        if proc2 is not None:
            terminate(proc2)


def _report(rows: list, tag: str) -> bool:
    print("\n" + "=" * 118)
    print(f"{'SECTION':<26}{'CHECK':<68}STATUS")
    print("-" * 118)
    for r in rows:
        print(f"{r['section']:<26}{r['check']:<68}{r['status']}")
        print(f"    -> {r['note']}")
    n_pass = sum(1 for r in rows if r["status"] == PASS)
    n_fail = sum(1 for r in rows if r["status"] == FAIL)
    n_skip = sum(1 for r in rows if r["status"] == SKIP)
    print("-" * 118)
    print(f"{n_pass}/{len(rows)} PASS, {n_fail} FAIL, {n_skip} SKIP")
    if n_fail:
        print("\n*** FAILURES (never smoothed over) ***")
        for r in rows:
            if r["status"] == FAIL:
                print(f"  [{r['section']}] {r['check']}\n    {r['note']}")

    out_dir = os.path.join(REPO, "runs", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"checkpoint_pin_battery_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"pass": n_pass, "fail": n_fail, "skip": n_skip, "total": len(rows), "rows": rows},
                  f, indent=2)
    print(f"\nJSON summary -> {out_path}")
    return n_fail == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Live acceptance battery for FORK-PIN-01 (checkpoint export/import/truncate). "
                    "Needs a real engine + GGUF; degrades to SKIP (exit 0) when unavailable.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tag", default="battery")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    model = _find_model(args.model)
    if model is None:
        print("SKIPPED: no autoregressive GGUF model found. Pass --model, or place one under "
             "~/.clozn/models/*.gguf.")
        return 0

    from clozn.cli import main as ctx
    from clozn.cli.engine_process import spawn_engine, _terminate_process
    from clozn.cli.commands.models import _flags_for

    def _spawn(model_path, port, *, prefer_gpu):
        return spawn_engine(model_path, port, _flags_for(model_path), prefer_gpu=prefer_gpu)

    try:
        proc, health, gpu = _spawn(model, args.port, prefer_gpu=not args.cpu)
    except ctx.CloznError as e:
        print(f"SKIPPED: no engine available to run the live battery -- {e}")
        return 0

    if not (health.get("capabilities") or {}).get("checkpoint_pin"):
        print("SKIPPED: this worker build does not advertise the 'checkpoint_pin' capability -- "
             "FORK-PIN-01's C++ side has not been merged into the engine build this script found yet.")
        _terminate_process(proc)
        return 0

    print(f"engine up: {os.path.basename(model)} (gpu={gpu}) port={args.port} "
         f"worker_generation_id={health.get('worker_generation_id')}")

    rows: list = []
    try:
        from clozn_engine import EngineClient
        eng = EngineClient(host=args.host, port=args.port)

        baseline = generate_baseline(eng, args.prompt, args.max_tokens, rows)
        if baseline is None:
            ok = _report(rows, args.tag)
            return 0 if ok else 1

        envelope = section_export_import_roundtrip(eng, baseline, rows)
        if envelope is not None:
            section_fail_closed(eng, envelope, rows)
        section_truncate(eng, baseline, rows)

        # Restart durability needs the FIRST worker gone before the second one starts (same port).
        _terminate_process(proc)
        proc = None
        if envelope is not None:
            section_restart_durability(model, args.host, args.port, args.cpu, envelope, baseline, rows,
                                       terminate=_terminate_process, spawn=_spawn)

        ok = _report(rows, args.tag)
        return 0 if ok else 1
    finally:
        if proc is not None:
            _terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
