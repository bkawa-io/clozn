"""quant_regression_bisect.py -- Phase 2 + 3 of the quantization-regression POPULATION study
(research/quant-regression-population): take a bounded, CATEGORY-STRATIFIED sample of the disagreements
scripts/tracer/quant_regression_mine.py found, run clozn.analysis.causal_bisect.run_bisect() on each --
over BOTH the `ffn` hook range and a `head` grid, as two independent searches -- and characterize the
resulting verdict distribution.

WHY TWO SEPARATE run_bisect() CALLS PER DISAGREEMENT (NOT ONE search_kinds=["ffn","head"] CALL)
----------------------------------------------------------------------------------------------------
run_bisect()'s own `_derive_verdict` produces ONE combined label from every window/site tested, whichever
hook kind found it (see causal_bisect.py's docstring: each composable kind gets "its OWN independent
tile-and-bisect search... populating the SAME window_tests array"). A single mixed-hook call would still
let a reader recover a per-hook read from `window_tests[].hook`/`single_site_tests[].hook`, but computing
"what would the verdict be from THIS hook alone" would mean re-deriving `_derive_verdict`'s own logic
outside `clozn.analysis` -- exactly the kind of reimplementation this project's own instructions warn
against (drift risk, and this repo's policy is report-a-bug-don't-patch-around-it for that module). Two
separate calls -- `search_kinds=["ffn"]` and `search_kinds=["head"]` -- give two genuinely independent,
directly-comparable `clozn.causal-bisect.v1` documents, each with its OWN verdict, at the cost of one
extra candidate-model residency per disagreement (modest: leaf single-site confirmations, the expensive
part, only run for RETAINED windows -- see below).

BOUNDS -- EVERY ONE OF THESE IS RECORDED IN THE OUTPUT REPORT, NOT JUST HERE
----------------------------------------------------------------------------------
  * SAMPLE: `--sample-size` disagreements (default 26), category-STRATIFIED (round-robin across the 7
    corpus categories, each category's own queue interleaving its Q2_K and Q4_K_M disagreements) -- never
    "the first N found". See `select_stratified_sample`.
  * Per disagreement, the write/readout is the prompt's FIRST disagreement position only (matches the
    prior transplant study's method: one flip, teacher-forced context up to it).
  * ffn: the hook's full writable range is always searched (never bounded beyond that -- see
    causal_bisect.py), tiled at `--ffn-window-size` (default max(4, n_layer//4)) and capped at
    `--ffn-max-windows` (default 4) coarse windows.
  * head: the FULL `head_layers x head_indices` grid is captured (`head_layers=range(n_layer)`,
    `head_indices=range(n_head)`) but narrowed to the top `--max-head-sites` (default 16) sites by
    observational reference-vs-candidate divergence BEFORE tiling/bisection (causal_bisect's own
    `max_head_sites` mechanism) -- tiled at `--head-window-size` (default 4), capped at
    `--head-max-windows` (default 4) coarse windows.
  * `store_tensors=False` throughout (verdicts/metrics only, no persisted tensor blobs -- matches
    scripts/smoke/bisect_acceptance.py's own choice and rationale).
  * SEED VARIES PER DISAGREEMENT (`--seed` is a BASE, `seed = base + sample_index`), deliberately NOT
    fixed -- see "A REAL METHODOLOGICAL BUG THIS SCRIPT FOUND IN ITSELF" below. Never varied to chase a
    particular outcome: assigned once, before any bisect runs, purely by each disagreement's fixed
    position in the deterministic stratified sample.

A REAL METHODOLOGICAL BUG THIS SCRIPT FOUND IN ITSELF (disclosed, not hidden, and now fixed here)
-------------------------------------------------------------------------------------------------------
`transplant.py`'s `_random_equal_norm_vector(reference_row, rng)` draws its raw direction from whatever
`rng` it is handed; `transplant.run_site()` constructs that `rng` as `random.Random(seed)` FRESH on every
call (transplant.py line ~594), using exactly the `seed` its caller passed in. `causal_bisect.run_bisect()`
threads ONE `seed` value through to EVERY leaf's `run_site()` confirmation unconditionally (its own
leaf-processing loop: `transplant.run_site(..., seed=seed, ...)`, same `seed` variable every time -- this
is the module working exactly as documented, not a defect in it).

The first version of this script called `run_bisect(seed=1, ...)` for all 30 disagreements uniformly (the
same pattern scripts/smoke/bisect_acceptance.py's own battery_4 already uses). Since Python's
`random.Random(1).gauss(...)` reproduces the IDENTICAL sequence of floats every time it is constructed
(verified live), and since every ffn/residual site shares the same vector width (n_embd) and every head
site shares the same width (d_head), that meant EVERY single-site leaf confirmation across the ENTIRE
30-disagreement sample -- different prompts, different layers, different candidate models -- was
comparing the reference transplant against the exact SAME frozen raw random direction, only rescaled to
each site's own reference-vector norm. If that one frozen direction happened to be an unusually weak
perturbation in this model (plausible in a ~3584-dim space -- nothing guarantees any single draw is
"typically" effective), it would systematically UNDER-perform as a control across the whole population,
inflating `reference_specific`/`localized_site` counts for a reason having nothing to do with genuine
quantization-damage localization. The first run's raw verdict histogram (`ffn`: 12 localized_site / 30 --
much higher than this project's own DISTRIBUTED_FUNCTION.md prior of ~3/12 reference-specific) was the
signal that triggered checking this.

THE FIX, AND ITS REMAINING LIMIT: varying `seed` per disagreement (this file) gives each of the 30
disagreements' searches an independently-drawn control direction, closing the cross-population confound.
It does NOT close a smaller, second-order version of the same gap: `causal_bisect.run_bisect()` still
passes ONE `seed` to every LEAF within a SINGLE disagreement's own bisection (when a search produces more
than one leaf site to confirm), so leaves WITHIN one disagreement's search still share a frozen raw
direction (rescaled per leaf's own norm). This is a real, disclosed limitation of `run_bisect()`'s current
seeding contract -- see this script's final report for how often that situation actually arose in the
sample -- and is reported here rather than patched, per this experiment's instruction not to edit anything
under `clozn/analysis/`.

NOTHING HERE IS TUNED TO PRODUCE A LOCALIZED VERDICT. Every bound above is fixed BEFORE the sample is
selected and applied uniformly to every disagreement in the sample, hook, and category. If a bound turns
out to matter (e.g. `max_head_sites` hides a genuine effect on a site it dropped), that is disclosed
honestly in `coverage.bounds_applied` on every individual document AND summarized in this report -- it is
never quietly widened only for a case that looks like it might localize.

CHECKPOINTING
----------------
runs/experiments/_quant_regression_bisect_checkpoint.json accumulates one completed disagreement record
(both hooks' full documents) at a time, atomic write-then-replace after each. Re-running this script skips
any selected disagreement id already present in the checkpoint.

OUTPUT
--------
runs/experiments/quant_regression_population_report.json -- `clozn.quant_regression_population.v1`: the
full per-disagreement bisect documents (the receipts) PLUS the characterization: verdict histogram overall,
by hook, by category, by hook x category, an instrument-sanity health check across every raw observation,
and a layer-depth breakdown of where (if anywhere) movement/beat_control occurred.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "engine", "client"))

from clozn.analysis import causal_bisect, pair_compatibility  # noqa: E402
from clozn.cli.engine_process import spawn_engine, _terminate_process  # noqa: E402
from clozn.cli.commands.models import _flags_for  # noqa: E402
from clozn_engine import EngineClient  # noqa: E402

MODELS_DIR = os.path.expanduser("~/.clozn/models")
REFERENCE_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q8_0.gguf")
CANDIDATES = {
    "Q2_K": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q2_K.gguf"),
    "Q4_K_M": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
}
MINE_REPORT_PATH = os.path.join(REPO, "runs", "experiments", "quant_regression_mine.json")
OUT_PATH = os.path.join(REPO, "runs", "experiments", "quant_regression_population_report.json")
CHECKPOINT_PATH = os.path.join(REPO, "runs", "experiments", "_quant_regression_bisect_checkpoint.json")

CATEGORY_ORDER = ("arithmetic", "factual_recall", "code_completion", "structured_json",
                  "multi_step_reasoning", "instruction_following", "multilingual")

_PORT = [8901]


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


def clean_loader(path: str):
    """A plain, unproxied engine loader -- the shape causal_bisect.run_bisect()'s reference_loader /
    candidate_loader expect (a zero-arg callable returning a context manager yielding something with
    .score()). Duplicated from scripts/smoke/bisect_acceptance.py's identical helper rather than imported
    -- that script is a standalone CLI battery, not a library module."""
    port = _next_port()

    @contextmanager
    def _cm():
        with _boot(path, port) as (eng, _h):
            yield eng
    return _cm()


def _atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =========================================================================== Phase 2a: build + select pool

def build_pool(mine_report: dict) -> dict:
    """category -> [disagreement entries], each candidate's own disagreements interleaved within a
    category so a later round-robin selection can't accidentally exhaust one candidate model before ever
    touching the other."""
    per_cat_per_cand: dict = {}
    for cand_key, cr in mine_report["candidates"].items():
        if cr.get("skipped"):
            continue
        for row in cr["prompts"]:
            if not row["has_disagreement"]:
                continue
            idx0 = row["disagree_positions"][0]
            pos = row["positions"][idx0]
            cont_ids = [p["reference_token_id"] for p in row["positions"]]
            entry = {
                "id": f"{cand_key}__{row['index']}",
                "candidate": cand_key,
                "category": row["category"],
                "prompt": row["prompt"],
                "prompt_ids": row["prompt_ids"],
                "continuation_ids": cont_ids,
                "first_disagree_index": idx0,
                "n_disagree_positions_this_prompt": row["n_disagree"],
                "target_token_id": pos["reference_token_id"],
                "target_piece": pos["reference_piece"],
                "candidate_top1_id": pos["candidate_top1_id"],
                "candidate_top1_piece": pos["candidate_top1_piece"],
                "reference_target_logprob": pos.get("reference_logprob"),
            }
            per_cat_per_cand.setdefault(row["category"], {}).setdefault(cand_key, []).append(entry)

    pool: dict = {}
    for cat, by_cand in per_cat_per_cand.items():
        cand_order = sorted(by_cand.keys())
        queues = {k: list(v) for k, v in by_cand.items()}
        merged = []
        while any(queues[k] for k in cand_order):
            for k in cand_order:
                if queues[k]:
                    merged.append(queues[k].pop(0))
        pool[cat] = merged
    return pool


def select_stratified_sample(pool: dict, target_n: int) -> list:
    """Deterministic round-robin across CATEGORY_ORDER (a category absent from `pool` -- no disagreements
    mined for it at all -- is simply skipped, never padded). No RNG: re-running with the same mine report
    and the same target_n reproduces the identical sample."""
    order = [c for c in CATEGORY_ORDER if pool.get(c)]
    queues = {c: list(pool[c]) for c in order}
    selected = []
    i = 0
    while len(selected) < target_n and any(queues[c] for c in order):
        c = order[i % len(order)]
        i += 1
        if queues[c]:
            selected.append(queues[c].pop(0))
    return selected


# ================================================================================ Phase 2b: run the bisects

def _pair_and_dims(candidate_path: str) -> dict:
    pc = pair_compatibility.assess_gguf_pair(REFERENCE_MODEL, candidate_path, label_a="reference",
                                              label_b="candidate")
    n_layer = (pc.get("layer_count") or {}).get("value_b")
    n_head = (pc.get("head_count") or {}).get("value_b")
    return {"pair_compat": pc, "n_layer": n_layer, "n_head": n_head}


def run_one_disagreement(entry: dict, candidate_path: str, dims: dict, *, ffn_window_size: "int | None",
                         ffn_max_windows: int, head_window_size: int, head_max_windows: int,
                         max_head_sites: int, seed: int, topk: int) -> dict:
    pc = dims["pair_compat"]
    n_layer, n_head = dims["n_layer"], dims["n_head"]
    if pc["verdict"]["overall"] == "incompatible":
        err = {"ok": False, "error": f"pair_compatibility refused: {pc['verdict']['reasons']}"}
        return {"ffn": err, "head": err}

    k = entry["first_disagree_index"]
    prompt_ids_ext = list(entry["prompt_ids"]) + list(entry["continuation_ids"][:k])
    continuation_ids_ext = [entry["continuation_ids"][k]]
    write_position = len(prompt_ids_ext) - 1
    readout_position = len(prompt_ids_ext)
    target_token_id = entry["target_token_id"]
    ref_logprob = entry.get("reference_target_logprob")

    common = dict(pair_compat=pc, reference_loader=lambda: clean_loader(REFERENCE_MODEL),
                 candidate_loader=lambda: clean_loader(candidate_path), prompt_ids=prompt_ids_ext,
                 continuation_ids=continuation_ids_ext, write_positions=[write_position],
                 readout_position=readout_position, target_token_id=target_token_id,
                 primary_metric="reference_token_logprob_recovery", topk=topk, seed=seed,
                 store_tensors=False, validate=True)
    if ref_logprob is not None:
        common["reference_target_logprob"] = float(ref_logprob)

    win_ffn = ffn_window_size or max(4, (n_layer or 28) // 4)
    ffn_out = causal_bisect.run_bisect(search_kinds=["ffn"], window_size=win_ffn,
                                       max_windows=ffn_max_windows, **common)

    if n_head:
        head_out = causal_bisect.run_bisect(
            search_kinds=["head"], head_layers=list(range(n_layer)), head_indices=list(range(n_head)),
            max_head_sites=max_head_sites, window_size=head_window_size, max_windows=head_max_windows,
            **common)
    else:
        head_out = {"ok": False, "error": "candidate pair's head_count is unknown -- head search skipped "
                                          "(never silently treated as 0 disagreement)"}

    return {"ffn": ffn_out, "head": head_out, "write_position": write_position,
           "readout_position": readout_position, "target_token_id": target_token_id,
           "prompt_ids_ext_len": len(prompt_ids_ext), "n_layer": n_layer, "n_head": n_head}


# =================================================================================== Phase 3: characterize

_LOCALIZING_LABELS = ("localized_site", "localized_window", "distributed_restoration")


def _depth_bucket(layer: int, n_layer: int) -> str:
    if not n_layer:
        return "unknown"
    frac = layer / max(1, n_layer - 1)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def characterize(records: list) -> dict:
    hist_overall = Counter()
    hist_by_hook = {"ffn": Counter(), "head": Counter()}
    hist_by_category = defaultdict(Counter)
    hist_by_hook_category = {"ffn": defaultdict(Counter), "head": defaultdict(Counter)}
    instrument_sane_true = 0
    instrument_sane_false = 0
    instrument_sane_unknown = 0
    localized_hits = []
    errors = []
    depth_counts = {"ffn": defaultdict(lambda: Counter()), "head": defaultdict(lambda: Counter())}
    multi_leaf_runs = []   # documents where >1 bisection_leaf shared one seed's random-control draw

    for rec in records:
        cat = rec["category"]
        n_layer = rec.get("n_layer")
        for hook in ("ffn", "head"):
            res = rec.get(hook)
            if not res:
                continue
            if not res.get("ok"):
                hist_overall["unavailable_or_error"] += 1
                hist_by_hook[hook]["unavailable_or_error"] += 1
                hist_by_category[cat]["unavailable_or_error"] += 1
                hist_by_hook_category[hook][cat]["unavailable_or_error"] += 1
                errors.append({"id": rec["id"], "hook": hook, "error": res.get("error")})
                continue
            doc = res["document"]
            label = doc["verdict"]["label"]
            hist_overall[label] += 1
            hist_by_hook[hook][label] += 1
            hist_by_category[cat][label] += 1
            hist_by_hook_category[hook][cat][label] += 1

            for w in doc.get("window_tests", []):
                sane = w.get("instrument_sane")
                if sane is True:
                    instrument_sane_true += 1
                elif sane is False:
                    instrument_sane_false += 1
                else:
                    instrument_sane_unknown += 1
                for layer in w.get("layers", []):
                    bucket = _depth_bucket(layer, n_layer)
                    depth_counts[hook][bucket]["tested"] += 1
                    if w.get("moved"):
                        depth_counts[hook][bucket]["moved"] += 1
                    if w.get("beat_control"):
                        depth_counts[hook][bucket]["beat_control"] += 1
            for s in doc.get("single_site_tests", []):
                if not s.get("ok"):
                    continue
                a = (s.get("transplant") or {}).get("analysis", {})
                sane = a.get("instrument_sane")
                if sane is True:
                    instrument_sane_true += 1
                elif sane is False:
                    instrument_sane_false += 1
                else:
                    instrument_sane_unknown += 1
                layer = s.get("layer")
                if layer is not None:
                    bucket = _depth_bucket(layer, n_layer)
                    depth_counts[hook][bucket]["tested"] += 1
                    if a.get("reference_moved_toward_reference"):
                        depth_counts[hook][bucket]["moved"] += 1
                    if a.get("reference_specific"):
                        depth_counts[hook][bucket]["beat_control"] += 1

            if label in _LOCALIZING_LABELS:
                localized_hits.append({
                    "id": rec["id"], "candidate": rec["candidate"], "category": cat, "hook": hook,
                    "label": label, "prompt": rec["prompt"], "evidence": doc["verdict"]["evidence"],
                })

            n_leaves = sum(1 for s in doc.get("single_site_tests", [])
                          if s.get("source") == "bisection_leaf")
            if n_leaves > 1:
                multi_leaf_runs.append({"id": rec["id"], "hook": hook, "n_leaves": n_leaves,
                                        "note": "these leaves' random_equal_norm controls share one "
                                                "seed's raw draw (run_bisect threads a single seed to "
                                                "every leaf's run_site() call) -- rescaled per leaf's own "
                                                "reference norm, but not independently redrawn"})

    total_sane_evaluable = instrument_sane_true + instrument_sane_false
    sane_rate = round(instrument_sane_true / total_sane_evaluable, 4) if total_sane_evaluable else None

    def _serialize_counter_dict(d):
        return {k: dict(v) for k, v in d.items()}

    return {
        "n_records": len(records),
        "verdict_histogram_overall": dict(hist_overall),
        "verdict_histogram_by_hook": {h: dict(c) for h, c in hist_by_hook.items()},
        "verdict_histogram_by_category": _serialize_counter_dict(hist_by_category),
        "verdict_histogram_by_hook_and_category": {
            h: _serialize_counter_dict(c) for h, c in hist_by_hook_category.items()
        },
        "instrument_sanity": {
            "sane_true": instrument_sane_true, "sane_false": instrument_sane_false,
            "sane_unknown_or_not_evaluable": instrument_sane_unknown, "sane_rate": sane_rate,
        },
        "localized_or_distributed_hits": localized_hits,
        "layer_depth_breakdown": {
            h: {bucket: dict(counts) for bucket, counts in buckets.items()}
            for h, buckets in depth_counts.items()
        },
        "multi_leaf_runs_sharing_one_seed_draw": multi_leaf_runs,
        "errors": errors,
    }


# ======================================================================================================= cli

def main() -> int:
    # Console codepage safety only (Windows cp1252 chokes on some decoded token pieces, e.g. box-drawing
    # replacement glyphs) -- never affects what is written to the JSON report, which is always UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mine-report", default=MINE_REPORT_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--sample-size", type=int, default=26)
    ap.add_argument("--ffn-window-size", type=int, default=None, help="default: max(4, n_layer//4)")
    ap.add_argument("--ffn-max-windows", type=int, default=4)
    ap.add_argument("--head-window-size", type=int, default=4)
    ap.add_argument("--head-max-windows", type=int, default=4)
    ap.add_argument("--max-head-sites", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test knob: only bisect the first N selected disagreements")
    ap.add_argument("--select-only", action="store_true",
                    help="print the stratified sample and exit -- no engine calls")
    args = ap.parse_args()

    with open(args.mine_report, encoding="utf-8") as f:
        mine_report = json.load(f)

    pool = build_pool(mine_report)
    print("pool sizes by category:", {c: len(v) for c, v in pool.items()}, flush=True)
    sample = select_stratified_sample(pool, args.sample_size)
    if args.limit is not None:
        sample = sample[:args.limit]
    print(f"selected {len(sample)} disagreements (target {args.sample_size}):", flush=True)
    cat_counts = Counter(e["category"] for e in sample)
    cand_counts = Counter(e["candidate"] for e in sample)
    print(f"  by category: {dict(cat_counts)}", flush=True)
    print(f"  by candidate: {dict(cand_counts)}", flush=True)
    for e in sample:
        print(f"    [{e['id']}] {e['category']:<22} {e['candidate']:<7} pos={e['first_disagree_index']} "
              f"{e['prompt'][:50]!r} ref={e['target_piece']!r} cand={e['candidate_top1_piece']!r}",
              flush=True)
    if args.select_only:
        return 0

    checkpoint = {}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:
            checkpoint = {}
    completed = checkpoint.get("completed", {})
    if completed:
        print(f"resuming: {len(completed)} disagreements already bisected in checkpoint", flush=True)

    bounds = {
        "sample_size_target": args.sample_size, "sample_size_actual": len(sample),
        "ffn_window_size": args.ffn_window_size or "max(4, n_layer // 4), per candidate",
        "ffn_max_windows": args.ffn_max_windows, "head_window_size": args.head_window_size,
        "head_max_windows": args.head_max_windows, "max_head_sites": args.max_head_sites,
        "seed_base": args.seed,
        "seed_scheme": ("seed = seed_base + sample_index -- VARIED per disagreement, not fixed. See "
                       "module docstring's 'A REAL METHODOLOGICAL BUG THIS SCRIPT FOUND IN ITSELF': a "
                       "single fixed seed across many disagreements reuses transplant.py's "
                       "_random_equal_norm_vector's raw draw (random.Random(seed) reproduces identically), "
                       "which would silently correlate the 'random' control across the whole sample."),
        "topk": args.topk, "store_tensors": False,
        "write_target": "first disagreement position only, per prompt",
    }

    dims_by_candidate: dict = {}
    records = []
    for sample_index, entry in enumerate(sample):
        eid = entry["id"]
        if eid in completed:
            records.append(completed[eid])
            continue
        cand_path = CANDIDATES[entry["candidate"]]
        if entry["candidate"] not in dims_by_candidate:
            dims_by_candidate[entry["candidate"]] = _pair_and_dims(cand_path)
        dims = dims_by_candidate[entry["candidate"]]

        entry_seed = args.seed + sample_index
        t0 = time.monotonic()
        print(f"\n[bisect {eid}] category={entry['category']} candidate={entry['candidate']} "
              f"seed={entry_seed} prompt={entry['prompt'][:60]!r}", flush=True)
        out = run_one_disagreement(
            entry, cand_path, dims, ffn_window_size=args.ffn_window_size,
            ffn_max_windows=args.ffn_max_windows, head_window_size=args.head_window_size,
            head_max_windows=args.head_max_windows, max_head_sites=args.max_head_sites,
            seed=entry_seed, topk=args.topk)
        out["seed"] = entry_seed
        elapsed = time.monotonic() - t0

        ffn_label = (out["ffn"]["document"]["verdict"]["label"] if out["ffn"].get("ok")
                    else f"ERROR: {out['ffn'].get('error')}")
        head_label = (out["head"]["document"]["verdict"]["label"] if out["head"].get("ok")
                     else f"ERROR: {out['head'].get('error')}")
        print(f"[bisect {eid}] ffn={ffn_label!r} head={head_label!r} ({elapsed:.1f}s)", flush=True)

        record = dict(entry)
        record.update(out)
        records.append(record)
        completed[eid] = record
        _atomic_write(CHECKPOINT_PATH, {"completed": completed})

    characterization = characterize(records)

    report = {
        "schema": "clozn.quant_regression_population.v1",
        "generated_at": _now_iso(),
        "reference_model": os.path.basename(REFERENCE_MODEL),
        "candidates": {k: os.path.basename(v) for k, v in CANDIDATES.items()},
        "mine_report_used": os.path.relpath(args.mine_report, REPO),
        "pool_sizes_by_category": {c: len(v) for c, v in pool.items()},
        "bounds": bounds,
        "sample": [{"id": e["id"], "candidate": e["candidate"], "category": e["category"],
                    "prompt": e["prompt"], "first_disagree_index": e["first_disagree_index"],
                    "target_piece": e["target_piece"], "candidate_top1_piece": e["candidate_top1_piece"]}
                   for e in sample],
        "characterization": characterization,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _atomic_write(args.out, report)
    print(f"\nwrote {args.out}", flush=True)

    print("\n=== verdict histogram (overall) ===")
    for label, n in sorted(characterization["verdict_histogram_overall"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:<28} {n}")
    print("\n=== verdict histogram by hook ===")
    for hook, hist in characterization["verdict_histogram_by_hook"].items():
        print(f"  {hook}: {dict(hist)}")
    print("\n=== instrument sanity ===")
    print(f"  {characterization['instrument_sanity']}")
    print(f"\n=== localized/distributed hits: {len(characterization['localized_or_distributed_hits'])} ===")
    for h in characterization["localized_or_distributed_hits"]:
        print(f"  [{h['id']}] hook={h['hook']} label={h['label']} evidence={h['evidence']}")

    try:
        os.remove(CHECKPOINT_PATH)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
