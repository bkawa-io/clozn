"""quant_regression_mine.py -- Phase 1 of the quantization-regression POPULATION study
(research/quant-regression-population): mine top-1 disagreements between Qwen2.5-7B-Instruct-Q8_0
(reference) and each of Q2_K / Q4_K_M (candidates) at scale, cheaply, across the full
scripts/tracer/quant_regression_prompts.json corpus (several hundred prompts spanning arithmetic,
factual_recall, code_completion, structured_json, multi_step_reasoning, instruction_following,
multilingual).

WHY THIS EXISTS
------------------
The instrument (clozn.analysis.causal_bisect) has passing acceptance batteries but its first real
application (scripts/tracer/transplant_localize.py) was thin: 15 prompts, 4 disagreements, ONE bisected.
That is not enough evidence to characterize a DISTRIBUTION of verdicts. Phase 1 mines disagreements at
scale so Phase 2 (scripts/tracer/quant_regression_bisect.py) can bisect a properly stratified sample
instead of whatever happened to be first.

METHOD -- EXACT GREEDY DECODE VIA TEACHER-FORCED SELF-FEEDING, THEN TEACHER-FORCED CANDIDATE READOUT
-----------------------------------------------------------------------------------------------------
There is no `/generate`-style endpoint on this engine surface that returns exact token ids for a
free-running completion -- `/v1/completions` returns text/board/layout, not a token-id trace suitable for
exact re-teacher-forcing (retokenizing text back would risk the documented BPE boundary-approximate drift;
see clozn_engine.EngineClient.score's own docstring). So the reference's greedy continuation is
reconstructed exactly, one token at a time, via `/score`'s own topk read-back: a `/score` call with any
non-empty throwaway `continuation` text returns `tokens[0].topk[0]` for the position right after the
supplied `prompt_ids` -- that entry depends ONLY on `prompt_ids` (see routes_whitebox.cpp: `logits_for[0]
= n_p - 1`), never on the throwaway continuation's own identity. Feeding that token back into `prompt_ids`
and repeating N_STEPS times reconstructs exact greedy decoding without ever needing a generation endpoint.
This is the same trick already used (single-step) by scripts/tracer/transplant_localize.py and
scripts/calibration/quant_vs_reference.py; this script is the first to iterate it into a multi-token
continuation.

Once the reference's continuation_ids are known, each candidate is teacher-forced on that EXACT sequence
in ONE `/score` call (`continuation_ids=...`, `topk>=1`): at continuation position k, `tokens[k].topk[0]`
is the candidate's own top-1 prediction for what should follow prompt+continuation[0:k] -- compared
against continuation_ids[k] (the reference's actual, greedily-chosen token there). A mismatch is a
genuine quantization regression: given the IDENTICAL preceding context the reference itself produced, the
candidate would not have made the same next-token choice.

SEQUENTIAL VRAM DISCIPLINE
-----------------------------
One engine process at a time (this box has 16GB VRAM; Q8_0 alone is ~7.5GB). Reference (Q8_0) is booted
once, walks the WHOLE corpus, then is torn down completely before either candidate is booted. Each
candidate is booted once, walks the whole corpus in ONE `/score` call per prompt (teacher-forced, no
generation loop needed), then is torn down.

CHECKPOINTING
----------------
Reference decode and each candidate's teacher-forced pass are checkpointed to
runs/experiments/_quant_regression_mine_checkpoint.json every CHECKPOINT_EVERY prompts (atomic
write-then-replace) and on completion of each pass. Re-running this script resumes from whatever the
checkpoint already has for the currently active phase/model, rather than repeating finished work.

OUTPUT
--------
runs/experiments/quant_regression_mine.json -- `clozn.quant_regression_mine.v1`: per-candidate
disagreement rates overall AND broken down by prompt category (the useful result Phase 1 promises on its
own, independent of whatever Phase 2 later bisects), plus every individual prompt's full per-position
record (needed as Phase 2's input pool to stratify from).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "engine", "client"))

from clozn.cli.engine_process import spawn_engine, _terminate_process  # noqa: E402
from clozn.cli.commands.models import _flags_for  # noqa: E402
from clozn_engine import EngineClient  # noqa: E402

MODELS_DIR = os.path.expanduser("~/.clozn/models")
REFERENCE_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q8_0.gguf")
CANDIDATES = {
    "Q2_K": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q2_K.gguf"),
    "Q4_K_M": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
}
CORPUS_PATH = os.path.join(REPO, "scripts", "tracer", "quant_regression_prompts.json")
OUT_PATH = os.path.join(REPO, "runs", "experiments", "quant_regression_mine.json")
CHECKPOINT_PATH = os.path.join(REPO, "runs", "experiments", "_quant_regression_mine_checkpoint.json")

N_STEPS_DEFAULT = 8       # greedy continuation length, in tokens
TOPK_DEFAULT = 5
CHECKPOINT_EVERY = 20
# Pieces that signal the reference has run off the end of anything meaningful to continue (raw
# completion prompts, no chat wrapping -- see corpus docstring); generation for THAT prompt stops early
# rather than forcing garbage tokens past a natural end. Best-effort: /health carries no eos_token_id
# (checked -- routes_state.cpp/server_main.cpp expose n_embd/n_layer/vocab_size but no eos id), so this is
# a piece-string heuristic, not an engine-confirmed stop id.
_STOP_PIECE_MARKERS = ("<|endoftext|>", "<|im_end|>", "<|im_start|>")

_PORT = [8801]


def _next_port() -> int:
    _PORT[0] += 1
    return _PORT[0]


@contextmanager
def _boot(path: str, port: int):
    proc, health, gpu = spawn_engine(path, port, _flags_for(path), prefer_gpu=True)
    try:
        yield EngineClient(host="127.0.0.1", port=port), health
    finally:
        _terminate_process(proc)


def _atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_corpus() -> tuple:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    flat = []
    for cat, prompts in doc["categories"].items():
        for p in prompts:
            flat.append({"category": cat, "prompt": p})
    return doc, flat


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ================================================================================= reference: greedy decode

def greedy_decode_all(engine, flat: list, n_steps: int, topk: int, done: dict) -> list:
    """`done` maps prompt-index (str) -> already-decoded record (resume support). Returns the full list in
    corpus order."""
    out = [None] * len(flat)
    n_resumed = 0
    for i, item in enumerate(flat):
        key = str(i)
        if key in done:
            out[i] = done[key]
            n_resumed += 1
            continue
        r0 = engine.score(prompt=item["prompt"], continuation=".", topk=topk)
        prompt_ids = list(r0["prompt_ids"])
        n_prompt = int(r0["n_prompt"])
        cur_ids = list(prompt_ids)
        steps = []
        stopped_early = False
        for _ in range(n_steps):
            r = engine.score(prompt_ids=cur_ids, continuation=".", topk=topk)
            top = r["tokens"][0]["topk"][0]
            piece = top.get("piece") or ""
            steps.append({"id": top["id"], "piece": piece, "logprob": top.get("logprob")})
            cur_ids.append(top["id"])
            if any(m in piece for m in _STOP_PIECE_MARKERS):
                stopped_early = True
                break
        out[i] = {"index": i, "category": item["category"], "prompt": item["prompt"],
                  "prompt_ids": prompt_ids, "n_prompt": n_prompt, "continuation": steps,
                  "stopped_early": stopped_early}
        done[key] = out[i]
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(flat):
            _atomic_write(CHECKPOINT_PATH, {"phase": "reference_decode", "done": done})
            print(f"  [reference decode] {i + 1}/{len(flat)} ({n_resumed} resumed from checkpoint)",
                  flush=True)
    return out


# ============================================================================ candidates: teacher-forcing

def teacher_force_candidate(engine, reference_rows: list, topk: int, done: dict) -> list:
    out = [None] * len(reference_rows)
    n_resumed = 0
    for i, row in enumerate(reference_rows):
        key = str(i)
        if key in done:
            out[i] = done[key]
            n_resumed += 1
            continue
        cont_ids = [s["id"] for s in row["continuation"]]
        if not cont_ids:
            out[i] = {"index": i, "category": row["category"], "prompt": row["prompt"],
                      "n_prompt": row["n_prompt"], "positions": [], "n_disagree": 0,
                      "disagree_positions": [], "has_disagreement": False,
                      "note": "reference continuation was empty (stopped immediately) -- nothing to force"}
            done[key] = out[i]
            continue
        r = engine.score(prompt_ids=row["prompt_ids"], continuation_ids=cont_ids, topk=max(topk, 1))
        positions = []
        disagree_positions = []
        for k, tok in enumerate(r.get("tokens") or []):
            topk_list = tok.get("topk") or []
            top1 = topk_list[0] if topk_list else None
            ref_id = cont_ids[k]
            cand_top1_id = top1.get("id") if top1 else None
            agree = (cand_top1_id == ref_id) if top1 is not None else None
            positions.append({
                "position_index": k, "reference_token_id": ref_id,
                "reference_piece": row["continuation"][k].get("piece"),
                "reference_logprob": row["continuation"][k].get("logprob"),
                "candidate_top1_id": cand_top1_id,
                "candidate_top1_piece": top1.get("piece") if top1 else None,
                "candidate_logprob_of_reference_token": tok.get("logprob"),
                "agree": agree,
            })
            if agree is False:
                disagree_positions.append(k)
        out[i] = {"index": i, "category": row["category"], "prompt": row["prompt"],
                  "prompt_ids": row["prompt_ids"], "n_prompt": row["n_prompt"], "positions": positions,
                  "n_disagree": len(disagree_positions), "disagree_positions": disagree_positions,
                  "has_disagreement": len(disagree_positions) > 0}
        done[key] = out[i]
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(reference_rows):
            print(f"    {i + 1}/{len(reference_rows)} ({n_resumed} resumed from checkpoint)", flush=True)
    return out


def _category_breakdown(rows: list, n_steps: int) -> dict:
    by_cat: dict = {}
    for r in rows:
        c = r["category"]
        b = by_cat.setdefault(c, {"n_prompts": 0, "n_with_disagreement": 0, "n_positions_scored": 0,
                                  "n_disagree_positions": 0})
        b["n_prompts"] += 1
        b["n_with_disagreement"] += int(r["has_disagreement"])
        b["n_positions_scored"] += len(r["positions"])
        b["n_disagree_positions"] += r["n_disagree"]
    for c, b in by_cat.items():
        b["prompt_disagreement_rate"] = round(b["n_with_disagreement"] / b["n_prompts"], 4) if b["n_prompts"] else None
        b["position_disagreement_rate"] = (round(b["n_disagree_positions"] / b["n_positions_scored"], 4)
                                           if b["n_positions_scored"] else None)
    return by_cat


def main() -> int:
    # Console codepage safety only (Windows cp1252 chokes on some decoded token pieces) -- never affects
    # what is written to the JSON report, which is always UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-steps", type=int, default=N_STEPS_DEFAULT)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test knob: only mine the first N prompts of the corpus (in corpus order)")
    args = ap.parse_args()

    corpus_doc, flat = load_corpus()
    if args.limit is not None:
        flat = flat[:args.limit]
    print(f"corpus: {len(flat)} prompts across {len(corpus_doc['counts'])} categories "
          f"{corpus_doc['counts']}", flush=True)

    checkpoint = _load_checkpoint()

    # ---- reference: Q8_0 greedy decode of the whole corpus
    if checkpoint.get("phase") == "reference_decode" and checkpoint.get("done"):
        ref_done = checkpoint["done"]
        print(f"[reference] resuming from checkpoint ({len(ref_done)}/{len(flat)} already decoded)",
              flush=True)
    else:
        ref_done = {}
    if len(ref_done) < len(flat) or checkpoint.get("phase") != "reference_decode_complete":
        t0 = time.monotonic()
        with _boot(REFERENCE_MODEL, _next_port()) as (eng, _h):
            reference_rows = greedy_decode_all(eng, flat, args.n_steps, args.topk, ref_done)
        _atomic_write(CHECKPOINT_PATH, {"phase": "reference_decode_complete", "done": ref_done,
                                        "reference_rows": reference_rows})
        print(f"[reference] decoded {len(reference_rows)} prompts in {time.monotonic() - t0:.1f}s", flush=True)
    else:
        reference_rows = checkpoint["reference_rows"]
        print(f"[reference] loaded {len(reference_rows)} decoded prompts from checkpoint", flush=True)

    n_stopped_early = sum(1 for r in reference_rows if r.get("stopped_early"))
    print(f"[reference] {n_stopped_early}/{len(reference_rows)} continuations stopped early "
          f"(stop-piece marker hit before n_steps={args.n_steps})", flush=True)

    # ---- candidates: teacher-force each on the reference's exact continuation
    candidate_results = {}
    for name, path in CANDIDATES.items():
        if not os.path.isfile(path):
            print(f"[{name}] model not found: {path} -- skipping", flush=True)
            candidate_results[name] = {"skipped": True, "reason": f"model not found: {path}"}
            continue
        ckpt_key = f"candidate_{name}"
        prior = checkpoint if checkpoint.get("phase") == ckpt_key else {}
        cand_done = dict(prior.get("done") or {})
        if prior.get("done"):
            print(f"[{name}] resuming from checkpoint ({len(cand_done)}/{len(reference_rows)} already scored)",
                  flush=True)
        t0 = time.monotonic()
        print(f"[{name}] booting candidate, teacher-forcing {len(reference_rows)} prompts ...", flush=True)
        with _boot(path, _next_port()) as (eng, _h):
            rows = teacher_force_candidate(eng, reference_rows, args.topk, cand_done)
            _atomic_write(CHECKPOINT_PATH, {"phase": ckpt_key, "done": cand_done})
        load_time = time.monotonic() - t0
        n_with = sum(1 for r in rows if r["has_disagreement"])
        n_pos_scored = sum(len(r["positions"]) for r in rows)
        n_pos_disagree = sum(r["n_disagree"] for r in rows)
        by_cat = _category_breakdown(rows, args.n_steps)
        candidate_results[name] = {
            "model": os.path.basename(path),
            "n_prompts": len(rows),
            "n_prompts_with_disagreement": n_with,
            "prompt_disagreement_rate": round(n_with / len(rows), 4) if rows else None,
            "n_positions_scored": n_pos_scored,
            "n_disagree_positions": n_pos_disagree,
            "position_disagreement_rate": round(n_pos_disagree / n_pos_scored, 4) if n_pos_scored else None,
            "by_category": by_cat,
            "load_time_s": round(load_time, 1),
            "prompts": rows,
        }
        print(f"[{name}] {n_with}/{len(rows)} prompts disagreed ({candidate_results[name]['prompt_disagreement_rate']}); "
              f"{n_pos_disagree}/{n_pos_scored} positions disagreed "
              f"({candidate_results[name]['position_disagreement_rate']}) in {load_time:.1f}s", flush=True)

    report = {
        "schema": "clozn.quant_regression_mine.v1",
        "generated_at": _now_iso(),
        "reference_model": os.path.basename(REFERENCE_MODEL),
        "n_steps_requested": args.n_steps,
        "topk": args.topk,
        "corpus": {"path": os.path.relpath(CORPUS_PATH, REPO), "total_prompts": len(flat),
                   "counts_by_category": corpus_doc["counts"]},
        "n_reference_stopped_early": n_stopped_early,
        "method": ("reference (Q8_0) greedy-decoded via iterated /score topk read-back (see module "
                  "docstring); each candidate teacher-forced on the EXACT reference continuation in one "
                  "/score call per prompt; a position disagrees when the candidate's own top-1 differs "
                  "from the reference token that was actually forced there."),
        "candidates": candidate_results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _atomic_write(args.out, report)
    print(f"\nwrote {args.out}")

    print("\n=== disagreement rates by category ===")
    for name, cr in candidate_results.items():
        if cr.get("skipped"):
            continue
        print(f"\n-- {name} ({cr['model']}) -- overall {cr['n_prompts_with_disagreement']}/{cr['n_prompts']} "
              f"prompts ({cr['prompt_disagreement_rate']}), {cr['n_disagree_positions']}/{cr['n_positions_scored']} "
              f"positions ({cr['position_disagreement_rate']})")
        for cat, b in sorted(cr["by_category"].items()):
            print(f"   {cat:<24} {b['n_with_disagreement']:>3}/{b['n_prompts']:<3} prompts "
                  f"({b['prompt_disagreement_rate']})  |  {b['n_disagree_positions']:>3}/{b['n_positions_scored']:<4} "
                  f"positions ({b['position_disagreement_rate']})")

    # Checkpoint no longer needed once the final report is safely on disk.
    try:
        os.remove(CHECKPOINT_PATH)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
