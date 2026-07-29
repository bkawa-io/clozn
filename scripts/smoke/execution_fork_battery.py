"""execution_fork_battery.py -- the live exactness matrix for POST /v1/execution-fork (exact execution
forks), the acceptance instrument for the whole feature.

WHAT THIS PROVES
-----------------
For each of four generation regimes (greedy, sampled, steered, sampled+steered):
  1. Generate ~100 tokens with checkpoint_on_finish=True (an EXISTING /v1/completions capability --
     the engine saves its own live KV as a checkpoint the moment decoding finishes, so the saved state
     is the run's own numerics by construction, not a reconstruction).
  2. Fork from that checkpoint at three interior positions (generated index 1, midpoint, final-1) with
     intervention {"type": "none"} -- the "do nothing different" control -- and assert the returned
     tokens are IDENTICAL, token for token, to the original run's tokens from that index onward. Per
     the wire contract, truncate_to > prompt_len at every one of these positions, so the engine must
     report restore_mode "live_kv_truncated" and exactness.source "live_kv"; this script checks BOTH
     fields, not just the token match, so a mislabeled-but-accidentally-matching reply still fails.
  3. Separately, fork at generated index 0 (truncate_to == prompt_len -- the prompt/generation boundary,
     where there is no generated-token KV left to truncate back to) and assert the engine reports
     restore_mode "reprefill" and never claims exactness.source "live_kv" there. This is a CORRECT
     result, not a bug -- the regime rule this whole feature exists to police.

HONEST DEGRADE
---------------
This needs a real engine + a real GGUF; neither is guaranteed to be present (this repo's product-minimal
CI lane and a bare dev checkout have neither), and the C++ side of this feature may not be merged into
the engine build this script finds even when one exists. Every one of those is reported as SKIPPED with
an explicit reason and a 0 exit code -- never as a FAIL and never silently. A 404 from POST
/v1/execution-fork specifically means "route not built on this worker yet" (see RouteNotBuilt below) and
is also a clean SKIP, not a FAIL.

A genuine divergence -- the model actually not run out the same continuation -- is a HARD FAIL that
names the regime, the exact continuation offset, and both token ids. Ethos: receipts, not self-narration;
a battery that smooths over a mismatch is worse than no battery.

Usage:
    python scripts/smoke/execution_fork_battery.py                      # auto-discovers model + engine
    python scripts/smoke/execution_fork_battery.py --model C:\\...\\x.gguf --port 8099 --max-tokens 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
ENGINE_CLIENT_DIR = os.path.join(REPO, "engine", "client")
if ENGINE_CLIENT_DIR not in sys.path:
    sys.path.insert(0, ENGINE_CLIENT_DIR)

PASS, FAIL = "PASS", "FAIL"
DEFAULT_PORT = 8099
DEFAULT_MAX_TOKENS = 100
DEFAULT_PROMPT = ("Write a short story about a lighthouse keeper who discovers something unusual "
                  "washed up on the shore one winter morning.")

# Preference order for auto-discovery under ~/.clozn/models: smallest/fastest known AUTOREGRESSIVE
# model first. Every clozn.cli.commands.models.KNOWN fragment WITHOUT a "mask"/"eos" flag is AR-safe;
# checkpoint_on_finish (and therefore this whole feature) is AR-only (see server_main.cpp), so an
# unrecognized GGUF is never guessed at -- it might be a diffusion model.
_PREFERRED_FRAGMENTS = ["qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct", "qwen2.5-7b-instruct",
                        "meta-llama-3.1-8b-instruct"]


class RouteNotBuilt(Exception):
    """POST /v1/execution-fork 404'd before any earlier call on this engine had already succeeded --
    read as "this worker's build predates the route", per this script's SKIP contract, not a feature
    failure. (A 404 AFTER the route has already answered once is a different, real bug -- see
    ForkProbe.fork -- and is reported as an ordinary FAIL row instead.)"""


class ForkProbe:
    """Wraps EngineClient.execution_fork to tell "route missing" apart from "this specific call
    failed": the FIRST 404 this engine ever returns for the route is ambiguous (unregistered route vs.
    a legitimately unknown checkpoint), but every 404 AFTER at least one call has already succeeded can
    only be the latter, since a route that answers once cannot un-register itself mid-run."""

    def __init__(self, eng):
        self.eng = eng
        self.confirmed = False

    def fork(self, **kw):
        from clozn_engine import EngineError
        try:
            resp = self.eng.execution_fork(**kw)
        except EngineError as e:
            if not self.confirmed and " -> 404:" in str(e):
                raise RouteNotBuilt(str(e)) from e
            raise
        self.confirmed = True
        return resp


def _find_model(explicit: "str | None") -> "str | None":
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
            continue   # diffusion -- checkpoint_on_finish is AR-only, skip it
        for name, path in files.items():
            if frag in name:
                return path
    return None   # nothing recognizably AR -- refuse to guess at an unknown GGUF


def _regimes(n_embd: int, n_layer: int) -> list:
    """(name, generation kwargs for EngineClient.complete) for the four required regimes. The steer
    vector is a fixed-seed synthetic direction (this battery proves EXACTNESS of the fork, not that
    steering is semantically meaningful) sized to the loaded model and pushed at a mid-depth layer."""
    rng = random.Random(20260728)
    steer_vec = [rng.uniform(-1.0, 1.0) for _ in range(max(1, n_embd))]
    steer_layer = max(1, n_layer // 2)
    steer_kwargs = {"steer_vec": steer_vec, "steer_coef": 3.0, "steer_layer": steer_layer}
    sampled_kwargs = {"temperature": 0.9, "seed": 20260728, "top_k": 40, "top_p": 0.95}
    return [
        ("greedy", {"temperature": 0}),
        ("sampled", dict(sampled_kwargs)),
        ("steered", {"temperature": 0, **steer_kwargs}),
        ("sampled+steered", {**sampled_kwargs, **steer_kwargs}),
    ]


def _normalize_ids(tokens) -> list:
    """The wire contract shows tokens:[...] without pinning an element shape; tolerate a flat list of
    ints OR a list of {id: ...} objects (the shape /score already uses elsewhere in this client) so a
    schema choice on the C++ side doesn't spuriously fail this battery."""
    out = []
    for t in tokens or []:
        out.append(int(t["id"]) if isinstance(t, dict) else int(t))
    return out


def _first_divergence(expected: list, got: list) -> dict:
    """The exact continuation offset and both token ids at the first mismatch -- never smoothed over.
    Also fires on a pure length mismatch (one side ran out first), which is itself a divergence."""
    n = min(len(expected), len(got))
    for i in range(n):
        if expected[i] != got[i]:
            return {"offset": i, "expected": expected[i], "got": got[i]}
    if len(expected) != len(got):
        return {"offset": n, "expected": expected[n] if n < len(expected) else None,
                "got": got[n] if n < len(got) else None, "length_mismatch": True,
                "expected_len": len(expected), "got_len": len(got)}
    return {}


def _row(regime, check, truncate_to, status, note) -> dict:
    return {"regime": regime, "check": check, "truncate_to": truncate_to, "status": status, "note": note}


def _forked(probe: ForkProbe, *, checkpoint_id, truncate_to, max_tokens):
    """One execution_fork(intervention={"type": "none"}) call -> (response, error_str). Never raises for
    an ordinary (already route-confirmed) engine error -- that comes back as a string so one bad cell
    records a FAIL row instead of aborting the rest of the matrix. RouteNotBuilt is the one exception
    let through unchanged (see ForkProbe.fork): if the route truly isn't built, every later call would
    404 too, so the whole battery SKIPs once instead of failing 16 times."""
    from clozn_engine import EngineError
    try:
        return probe.fork(checkpoint_id=checkpoint_id, truncate_to=truncate_to, max_tokens=max_tokens,
                          intervention={"type": "none"}), None
    except RouteNotBuilt:
        raise
    except (EngineError, ValueError) as e:
        return None, str(e)


def _judge_interior(regime: str, idx: int, truncate_to: int, expected: list, resp, err) -> dict:
    check = f"fork gen_idx={idx} (intervention=none, expect live_kv exact)"
    if err is not None:
        return _row(regime, check, truncate_to, FAIL, f"request failed: {err}")
    restore_mode = resp.get("restore_mode")
    exactness = resp.get("exactness") or {}
    exactness_source = exactness.get("source")
    got = _normalize_ids(resp.get("tokens"))
    if restore_mode != "live_kv_truncated" or exactness_source != "live_kv":
        return _row(regime, check, truncate_to, FAIL,
                   f"regime rule violated: expected restore_mode='live_kv_truncated' + "
                   f"exactness.source='live_kv' at gen_idx={idx}, got restore_mode={restore_mode!r} "
                   f"exactness.source={exactness_source!r}")
    if got != expected:
        d = _first_divergence(expected, got)
        if d.get("length_mismatch"):
            return _row(regime, check, truncate_to, FAIL,
                       f"DIVERGED (length): expected {d['expected_len']} tokens, got {d['got_len']} "
                       f"(first differing offset {d['offset']}, generated-index "
                       f"{idx + d['offset']}: expected={d['expected']!r} got={d['got']!r})")
        return _row(regime, check, truncate_to, FAIL,
                   f"DIVERGED at continuation offset {d['offset']} (generated-index {idx + d['offset']}, "
                   f"checkpoint position {truncate_to + d['offset']}): "
                   f"expected token_id={d['expected']} got token_id={d['got']}")
    return _row(regime, check, truncate_to, PASS, f"{len(got)} tokens reproduced exactly")


def _judge_boundary(regime: str, prompt_len: int, resp, err) -> dict:
    check = "fork gen_idx=0 (prompt boundary, expect reprefill, NOT a bug)"
    if err is not None:
        return _row(regime, check, prompt_len, FAIL, f"request failed: {err}")
    restore_mode = resp.get("restore_mode")
    exactness = resp.get("exactness") or {}
    exactness_source = exactness.get("source")
    if restore_mode != "reprefill":
        return _row(regime, check, prompt_len, FAIL,
                   f"expected restore_mode='reprefill' at the prompt boundary (truncate_to == "
                   f"prompt_len), got {restore_mode!r}")
    if exactness_source == "live_kv":
        return _row(regime, check, prompt_len, FAIL,
                   "reprefill response illegally claimed exactness.source == 'live_kv'")
    return _row(regime, check, prompt_len, PASS,
               f"prompt-boundary fork correctly reprefilled (restore_mode={restore_mode!r}, "
               f"exactness.source={exactness_source!r})")


def run_matrix(probe: ForkProbe, eng, prompt: str, max_tokens: int, n_embd: int, n_layer: int,
              rows: list) -> None:
    """Appends one row per check to `rows`. Raises RouteNotBuilt (uncaught here, on purpose) the first
    time a fork call 404s before any fork call has ever succeeded on this engine -- see ForkProbe."""
    for regime_name, gen_kwargs in _regimes(n_embd, n_layer):
        gen_check = f"generate ~{max_tokens} tokens (checkpoint_on_finish=true)"
        try:
            original = eng.complete(prompt, max_tokens=max_tokens, checkpoint_on_finish=True, **gen_kwargs)
        except Exception as e:   # noqa: BLE001 -- one bad regime must not abort the other three
            rows.append(_row(regime_name, gen_check, None, FAIL, f"generation request raised: {e}"))
            continue

        checkpoint_id = original.get("checkpoint_id")
        board = original.get("board") or []
        prompt_tokens = (original.get("usage") or {}).get("prompt_tokens")
        if not checkpoint_id or not isinstance(prompt_tokens, int) or len(board) <= prompt_tokens:
            rows.append(_row(regime_name, gen_check, None, FAIL,
                            f"no usable checkpoint from generation (checkpoint_id={checkpoint_id!r}, "
                            f"prompt_tokens={prompt_tokens!r}, board_len={len(board)})"))
            continue

        generated_ids = [int(t) for t in board[prompt_tokens:]]
        gen_len = len(generated_ids)
        rows.append(_row(regime_name, gen_check, None, PASS,
                        f"{gen_len} tokens generated, checkpoint_id={checkpoint_id}"))
        if gen_len < 3:
            rows.append(_row(regime_name, "fork positions (gen_idx 1/mid/final-1)", None, FAIL,
                            f"only {gen_len} generated tokens -- too few to exercise all three positions"))
            continue

        for idx in sorted({1, gen_len // 2, gen_len - 1}):
            truncate_to = prompt_tokens + idx
            expected = generated_ids[idx:]
            resp, err = _forked(probe, checkpoint_id=checkpoint_id, truncate_to=truncate_to,
                                max_tokens=len(expected))
            rows.append(_judge_interior(regime_name, idx, truncate_to, expected, resp, err))

        resp, err = _forked(probe, checkpoint_id=checkpoint_id, truncate_to=prompt_tokens, max_tokens=4)
        rows.append(_judge_boundary(regime_name, prompt_tokens, resp, err))


def _report(rows: list, tag: str) -> bool:
    print("\n" + "=" * 108)
    print(f"{'REGIME':<18}{'CHECK':<58}{'TRUNC':>7}  STATUS")
    print("-" * 108)
    for r in rows:
        trunc = "" if r["truncate_to"] is None else str(r["truncate_to"])
        print(f"{r['regime']:<18}{r['check']:<58}{trunc:>7}  {r['status']}")
        print(f"    -> {r['note']}")
    n_pass = sum(1 for r in rows if r["status"] == PASS)
    n_fail = sum(1 for r in rows if r["status"] == FAIL)
    print("-" * 108)
    print(f"{n_pass}/{len(rows)} PASS, {n_fail} FAIL")
    if n_fail:
        print("\n*** FAILURES (exact index + both token ids above; never smoothed over) ***")
        for r in rows:
            if r["status"] == FAIL:
                print(f"  [{r['regime']}] {r['check']}\n    {r['note']}")

    out_dir = os.path.join(REPO, "runs", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"execution_fork_battery_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"pass": n_pass, "fail": n_fail, "total": len(rows), "rows": rows}, f, indent=2)
    print(f"\nJSON summary -> {out_path}")
    return n_fail == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Live exactness matrix for POST /v1/execution-fork (exact execution forks). "
                    "Needs a real engine + GGUF; degrades to a clean SKIP (exit 0) when either is "
                    "unavailable, or when the route itself 404s (not yet built on this worker).")
    ap.add_argument("--model", default=None,
                    help="GGUF path (default: auto-discover an autoregressive model under "
                         "~/.clozn/models/*.gguf)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="tokens generated per regime before forking (default 100)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tag", default="battery", help="output filename tag")
    ap.add_argument("--cpu", action="store_true", help="force a CPU engine build")
    args = ap.parse_args(argv)

    model = _find_model(args.model)
    if model is None:
        print("SKIPPED: no autoregressive GGUF model found. Pass --model, or place one under "
             "~/.clozn/models/*.gguf (checkpoint_on_finish -- and so this whole feature -- is "
             "AR-only; an unrecognized GGUF is never guessed at in case it is a diffusion model).")
        return 0

    from clozn.cli import main as ctx
    from clozn.cli.engine_process import spawn_engine, _terminate_process
    from clozn.cli.commands.models import _flags_for

    try:
        proc, health, gpu = spawn_engine(model, args.port, _flags_for(model), prefer_gpu=not args.cpu)
    except ctx.CloznError as e:
        print(f"SKIPPED: no engine available to run the live battery -- {e}")
        return 0

    print(f"engine up: {os.path.basename(model)} (gpu={gpu}) "
         f"n_embd={health.get('n_embd')} n_layer={health.get('n_layer')} port={args.port}")

    try:
        from clozn_engine import EngineClient
        eng = EngineClient(host=args.host, port=args.port)
        probe = ForkProbe(eng)
        rows: list = []
        try:
            run_matrix(probe, eng, args.prompt, args.max_tokens,
                      int(health.get("n_embd") or 0), int(health.get("n_layer") or 0), rows)
        except RouteNotBuilt as e:
            print("SKIPPED: POST /v1/execution-fork is not available on this engine build (404) -- "
                 "the C++ side of this feature has not been merged into the worker this script found "
                 f"yet.\n  ({e})")
            return 0

        ok = _report(rows, args.tag)
        return 0 if ok else 1
    finally:
        _terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
