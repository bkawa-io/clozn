"""tracer.py -- the intervention-validated circuit tracer (S0-S2 of notes/CIRCUIT_TRACER_DESIGN.md).

The claim this module manufactures: "token y at position t was caused by THESE internal components
-- proven by changing only them and watching the prediction move." Every node in the output graph
carries a MEASURED patch effect (delta logprob under an ablation we actually ran against the live
engine), never a formula score. Correlational tools only NOMINATE candidates (the S0 screen);
nothing enters the graph without surviving its own patch test (S1) against matched controls.

Method (all logit-tier arms are ONE teacher-forced /score forward each, ~100-250 ms):
  S0 screen  -- capture every position's residual at the J-lens layers off the baseline forward
                (/score + capture, chunked); rank sites by |h . dir(c)| where dir(c) =
                normalize(J_l^T @ W_U[c]) (the validated injection direction, built client-side
                from the J sidecar + the engine's /jlens/unembed_row route -- EXACT position
                alignment, no retokenization drift). The screen is correlational and only
                nominates; a missed site is a completeness limitation (reported via
                screened_sites), a false site dies in S1.
  S1 solo    -- per candidate node: full-ablate (h <- the run's own mean residual at that layer;
                mean, never zero -- zero is off-manifold and overstates effects) and
                directional-ablate (h <- h - (h.d)d, the named component only). Controls:
                random-equal-norm direction at the same sites + the real direction at random
                non-candidate sites. The noise floor is 3x the median |control delta|; survivors
                must beat it.
  S2 joint   -- ALL surviving nodes full-ablated in ONE forward (the engine's multi-write) ->
                delta_total, and the interaction gap = delta_total - sum(solo) reported as-is
                (nodes interact; solo effects don't sum -- hiding that in a normalization would
                be dishonest).

Effects are LOG-PROB deltas (what /score returns): delta_i = logprob_y(baseline) -
logprob_y(ablated). Positive = the node was pushing TOWARD y. `legibility` (delta_dir/delta_full)
is NOT clamped to [0,1]: >1 means the named direction's ablation removes MORE than mean-row
replacement (the mean row itself carries concept mass), negative means the node is suppressive
along that direction. Both are meaningful and reported raw -- read the two deltas, not just the
ratio. First live trace (9B, "capital of France" -> " Paris") produced exactly these cases. The margin-flip PREDICTION
(delta_full > baseline margin => the generated token should actually change) is recorded per node;
the generation arms that OBSERVE it are S4 (not in this slice -- `margin_flip_observed` is null).

Verdicts (stamped on the receipt, never softened):
  PASS                -- >= 1 survivor and the controls sit well below the real effects.
  NO_CAUSAL_NODES     -- nothing beat the noise floor. An honest outcome (possibly a genuinely
                         distributed circuit), distinct from FAILED.
  FAILED_CONTROLS     -- random interventions moved the target as much as the "real" ones: the
                         screen is not finding anything real on this model/prompt. Per the design's
                         STOP check, do not trust (or ship) traces in this state.

House honesty style: trace() never raises for engine/data problems -- it returns a labeled
{"ok": False, "blocked": ...} dict (mirrors concept_dir.ConceptSteer). The pure math helpers DO
raise on malformed inputs; they are the fixture-tested seam.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from clozn.analysis.concept_dir import dir_c_from_row, load_jlens_jacobians

NOISE_FLOOR_MULT = 3.0   # survivors must beat this multiple of the median |control| delta
DEFAULT_ENGINE = "http://127.0.0.1:8080"


@dataclass
class TraceBudget:
    """Arm-count knobs. Defaults give ~80-150 forwards (a few seconds on the 9B)."""
    max_candidates: int = 24     # sites the screen may nominate (cap, reported)
    capture_chunk: int = 48      # positions per /score capture call (payload control)
    n_random_dir: int = 8        # random-equal-norm-direction control arms
    n_random_site: int = 8       # real-direction-at-random-site control arms
    topk: int = 5                # baseline top-k (for the margin)
    layers: Optional[list] = None  # lens layers to trace at (default: every fitted sidecar layer)
    extra_concepts: list = field(default_factory=list)  # words beyond the target token itself
    max_edges: int = 8           # S3 path-patching pairs (4 arms each; 0 = skip S3)
    run_s4: bool = True          # S4 generation arms (patch + greedy decode + divergence check)
    ablate_screen_arms: int = 64  # grid budget for screen_mode="ablate" (any-GGUF, no sidecar)


# ====================================================================== pure math (fixture-tested)

def directional_ablate(h: np.ndarray, d: np.ndarray) -> np.ndarray:
    """h with its component along d removed: h - (h . d_hat) d_hat. Raises on shape mismatch or a
    ~zero direction (an ablation along nothing is a bug, not a no-op)."""
    h = np.asarray(h, dtype=np.float32).reshape(-1)
    d = np.asarray(d, dtype=np.float32).reshape(-1)
    if h.shape != d.shape:
        raise ValueError(f"shape mismatch: h {h.shape} vs d {d.shape}")
    nd = float(np.linalg.norm(d))
    if nd < 1e-8:
        raise ValueError("degenerate (~zero) ablation direction")
    d_hat = d / nd
    return h - float(h @ d_hat) * d_hat


def screen_candidates(H_by_layer: dict, dirs_by_layer: dict, max_candidates: int,
                      force_sites: list) -> list:
    """Rank sites (layer, pos) by max-over-concepts |h(layer,pos) . dir(c)| (unit dirs).
    H_by_layer: {layer: [n_pos, d] float32}; dirs_by_layer: {layer: {concept_label: unit dir}}.
    Returns up to max_candidates dicts {layer, pos, screen_score, concept}, force_sites first
    (deduped) -- forced sites carry the screen score they earned, or 0.0 if unscored."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    scored = {}
    for layer, H in H_by_layer.items():
        H = np.asarray(H, dtype=np.float32)
        for label, d in (dirs_by_layer.get(layer) or {}).items():
            proj = H @ np.asarray(d, dtype=np.float32)          # [n_pos]
            for p in range(H.shape[0]):
                s = abs(float(proj[p]))
                key = (int(layer), int(p))
                if key not in scored or s > scored[key][0]:
                    scored[key] = (s, label)
    out, seen = [], set()
    for layer, pos in force_sites:
        key = (int(layer), int(pos))
        if key in seen:
            continue
        seen.add(key)
        s, label = scored.get(key, (0.0, None))
        out.append({"layer": key[0], "pos": key[1], "screen_score": s, "concept": label})
    for key, (s, label) in sorted(scored.items(), key=lambda kv: -kv[1][0]):
        if len(out) >= max_candidates:
            break
        if key in seen:
            continue
        seen.add(key)
        out.append({"layer": key[0], "pos": key[1], "screen_score": s, "concept": label})
    return out[:max_candidates]


def noise_floor(control_deltas: list, mult: float = NOISE_FLOOR_MULT) -> float:
    """The survive-me bar: mult x median(|control delta|). Raises on an empty control set --
    a trace with no controls has no floor and must not silently pass anything."""
    if not control_deltas:
        raise ValueError("no control deltas: cannot establish a noise floor")
    return mult * float(np.median(np.abs(np.asarray(control_deltas, dtype=np.float64))))


def accounting(solo_deltas: list, delta_total: float) -> dict:
    """The explicit unexplained-mass split: sum of solo effects vs the measured joint effect.
    interaction_gap = delta_total - sum_solo (negative => sub-additive / self-repair;
    positive => super-additive). Reported raw, never normalized away."""
    sum_solo = float(np.sum(solo_deltas)) if solo_deltas else 0.0
    return {"delta_total": float(delta_total), "sum_solo": sum_solo,
            "interaction_gap": float(delta_total) - sum_solo}


def controls_verdict(survivor_full_deltas: list, control_deltas: list) -> str:
    """PASS / NO_CAUSAL_NODES / FAILED_CONTROLS (see module docstring). FAILED_CONTROLS wins over
    everything: if the strongest control matches the strongest 'real' effect, nothing here is
    trustworthy regardless of how many nominal survivors there are."""
    if not control_deltas:
        raise ValueError("no control deltas: cannot issue a verdict")
    ctl_max = float(np.max(np.abs(control_deltas)))
    real_max = float(np.max(np.abs(survivor_full_deltas))) if survivor_full_deltas else 0.0
    if survivor_full_deltas and ctl_max >= real_max:
        return "FAILED_CONTROLS"
    if not survivor_full_deltas:
        return "NO_CAUSAL_NODES"
    return "PASS"


def edge_candidates(nodes: list, max_edges: int) -> list:
    """S3 path-patching pairs (A, B): layer_A < layer_B AND pos_A <= pos_B (under causal attention
    a later position never feeds an earlier one, and within a position information only flows
    upward through layers). Ordered by |delta_A * delta_B| (strongest joint mass first), capped."""
    if max_edges < 0:
        raise ValueError("max_edges must be >= 0")
    pairs = [(A, B) for A in nodes for B in nodes
             if A["layer"] < B["layer"] and A["pos"] <= B["pos"]]
    pairs.sort(key=lambda ab: -abs(ab[0]["delta_full"] * ab[1]["delta_full"]))
    return pairs[:max_edges]


def group_joint_writes(nodes: list, mean_rows: dict) -> list:
    """The S2 joint-arm write specs: one {layer, positions, values} per layer, every surviving
    node's position mean-ablated simultaneously (values = that layer's mean row per position)."""
    by_layer = {}
    for n in nodes:
        by_layer.setdefault(int(n["layer"]), []).append(int(n["pos"]))
    specs = []
    for layer, positions in sorted(by_layer.items()):
        row = np.asarray(mean_rows[layer], dtype=np.float32)
        values = np.concatenate([row] * len(positions))
        specs.append({"layer": layer, "positions": positions, "values": values.tolist()})
    return specs


# ================================================================================ engine plumbing

def _post(engine_url: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(engine_url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _score(engine_url: str, prompt: str, cont, *, topk: int = 0, write=None, capture=None,
           arms=None) -> dict:
    """One teacher-forced /score arm. `cont` is token ids (exact, preferred) or text (the engine
    flags boundary_approximate). `write` is a spec dict or list of them; `capture` is
    {layers, positions}. `arms` is the batched multi-arm form (screening-only regime -- the
    response's own numerical_regime label says why)."""
    body = {"prompt": prompt, "topk": topk}
    if isinstance(cont, (list, tuple)):
        body["continuation_ids"] = list(cont)
    else:
        body["continuation"] = cont
    if write is not None:
        body["write"] = write
    if capture is not None:
        body["capture"] = capture
    if arms is not None:
        body["arms"] = arms
    return _post(engine_url, "/score", body)


def _target_logprob(resp: dict, target_idx: int) -> float:
    return float(resp["tokens"][target_idx]["logprob"])


def _complete(engine_url: str, prompt: str, max_tokens: int, *, reference=None, write=None) -> dict:
    """One GREEDY generation arm via /v1/completions — the S4 observation: patch armed (`write`),
    decode, stop at the first token that diverges from `reference` (the 2.1 early-stop; the
    response carries diverged/diverged_at)."""
    body = {"prompt": prompt, "max_tokens": int(max_tokens), "temperature": 0}
    if reference is not None:
        body["reference_tokens"] = list(reference)
    if write is not None:
        body["write"] = write
    return _post(engine_url, "/v1/completions", body)


def _resolve_concept(engine_url: str, word: str):
    """Concept word -> single leading-space vocab token id via /score's own tokenizer (the
    concept_dir resolve trick). Multi-token words are reported unresolvable, never truncated."""
    try:
        r = _score(engine_url, "Consider the word:", " " + word.strip(), topk=0)
    except Exception as e:
        return None, f"tokenization round-trip failed: {e}"
    toks = r.get("tokens") or []
    if len(toks) != 1:
        return None, f"'{word}' is {len(toks)} tokens (need exactly 1)"
    return int(toks[0]["id"]), None


def _unembed_row(engine_url: str, token_id: int) -> np.ndarray:
    r = _post(engine_url, "/jlens/unembed_row", {"token_id": int(token_id)})
    vec = r.get("vector")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError(f"/jlens/unembed_row returned no vector for token {token_id}: {r!r}")
    return np.asarray(vec, dtype=np.float32)


# ==================================================================================== the tracer

def trace(prompt: str, continuation, target_idx: int, *,
          engine_url: str = DEFAULT_ENGINE, jlens_dir: Optional[str] = None,
          budget: Optional[TraceBudget] = None, seed: int = 0,
          screen_mode: str = "auto", contrast=None) -> dict:
    """Trace the circuit behind continuation token `target_idx`. Returns the receipt dict
    (see notes/CIRCUIT_TRACER_DESIGN.md section 4); {"ok": False, "blocked": ...} on any
    engine/data failure. `continuation` is token ids (exact, from a stored trace) or text.

    `screen_mode`: how S0 nominates candidate sites.
      * "jlens" -- dir(c) projections through the fitted J-lens sidecar (the original screen;
        richer nominations + per-node legibility). Refuses if no sidecar qualifies.
      * "ablate" -- a coarse mean-ablation sweep over a (layer x position) grid: sites are
        nominated by their own measured causal delta. Works on ANY GGUF, no sidecar -- this is
        what makes a second-model-family trace possible -- and is causal from the first step
        (the jlens screen is correlational until S1 confirms). Cost: ~grid-size extra /score
        arms. Per-node `legibility` is None in this mode (there is no named direction to
        project onto) and `delta_dir` is None.
      * "auto" -- jlens when a sidecar loads, ablate otherwise (the loud downgrade is recorded
        in the receipt's config.screen_mode + config.screen_note).

    `contrast`: an optional FOIL token (text or a single id, or the literal "auto" = the
    baseline runner-up). When set, every delta in the trace becomes CONTRASTIVE -- the change in
    the logit GAP between y and the foil, not y's absolute logprob:
        delta = [lp_y(base) - lp_foil(base)] - [lp_y(ablated) - lp_foil(ablated)]
    This is the answer given by the screen-null result (runs/experiments/screen_null_*.json,
    notes/CIRCUIT_TRACER_DESIGN.md): the ABSOLUTE screen nominates sites causal for the answer
    CATEGORY ("we're asking for a capital") -- which move a wrong-but-plausible foil almost as
    much as the true answer, so an absolute PASS does NOT certify answer-SELECTIVITY. Contrastive
    scoring keeps the same positions but makes the verdict select the sites that prefer y OVER the
    foil. Cost: each arm scores the foil too (sequential screen forced; the batched-arms screen is
    single-continuation and is skipped in contrast mode)."""
    budget = budget or TraceBudget()
    rng = np.random.default_rng(seed)
    try:
        # ---- baseline: target logprob + margin + the fixed sequence bookkeeping -------------
        base = _score(engine_url, prompt, continuation, topk=max(2, budget.topk))
        n_p, n_a = int(base["n_prompt"]), int(base["n_cont"])
        if not (0 <= target_idx < n_a):
            return {"ok": False, "blocked": f"target_idx {target_idx} out of range (n_cont {n_a})"}
        tgt = base["tokens"][target_idx]
        y_id, y_piece = int(tgt["id"]), tgt["piece"]
        base_lp = float(tgt["logprob"])
        # margin: y's logprob vs the best OTHER token at that row (positive => y is the argmax)
        others = [t for t in (tgt.get("topk") or []) if int(t["id"]) != y_id]
        margin = base_lp - float(others[0]["logprob"]) if others else float("nan")
        t_abs = n_p + target_idx          # y's absolute position; the row predicting it is t_abs-1
        sites_end = t_abs                 # patchable sites: absolute positions [0, t_abs-1]
        # Trim the continuation to the target (causal: later tokens can't affect y's row). Exact
        # ids from the baseline response, so every later arm scores the identical sequence.
        cont_ids = [int(t["id"]) for t in base["tokens"][: target_idx + 1]]

        # ---- contrastive scoring setup (optional foil) ---------------------------------------
        # y and the foil are BOTH predicted from the same row (t_abs-1, conditioned on the shared
        # prefix cont_ids[:target_idx]); the foil arm just swaps the last id. base_lp_foil is the
        # foil's baseline logprob at that row. arm() below subtracts the foil's ablated delta so
        # the whole pipeline scores the y-vs-foil logit gap.
        foil_id = foil_piece = None
        base_lp_foil = 0.0
        cont_ids_foil = None
        if contrast is not None:
            if isinstance(contrast, str) and contrast == "auto":
                if not others:
                    return {"ok": False, "blocked": "contrast='auto' but no runner-up token in topk"}
                foil_id = int(others[0]["id"])
            elif isinstance(contrast, (int,)) or (isinstance(contrast, str) is False and
                                                  isinstance(contrast, (list, tuple))):
                foil_id = int(contrast[0] if isinstance(contrast, (list, tuple)) else contrast)
            else:  # text: tokenize via a throwaway /score, take its first continuation id
                fb = _score(engine_url, prompt, contrast, topk=0)
                if not fb.get("tokens"):
                    return {"ok": False, "blocked": f"contrast text {contrast!r} produced no tokens"}
                foil_id = int(fb["tokens"][0]["id"])
            if foil_id == y_id:
                return {"ok": False, "blocked": "contrast foil equals the target token"}
            cont_ids_foil = cont_ids[:target_idx] + [foil_id]
            bf = _score(engine_url, prompt, cont_ids_foil, topk=0)
            base_lp_foil = _target_logprob(bf, target_idx)
            foil_piece = bf["tokens"][target_idx].get("piece")

        # ---- screen selection: dir(c) through the sidecar, or the any-GGUF ablation grid -----
        if screen_mode not in ("auto", "jlens", "ablate"):
            return {"ok": False, "blocked": f"unknown screen_mode {screen_mode!r}"}
        J_by_layer = None
        screen_note = None
        if screen_mode in ("auto", "jlens"):
            try:
                J_by_layer = load_jlens_jacobians(jlens_dir, layers=budget.layers)
                # QUALIFY the sidecar against THIS engine, not just load it: with jlens_dir=None
                # the loader falls back to the default dir, which may hold a DIFFERENT model's J
                # (measured: the Qwen 3584-dim J loaded cleanly against a Llama 4096-dim engine
                # and exploded 16/16 cases downstream). A J whose d_model differs from the
                # engine's own hidden size does not qualify -- same rule as the product
                # artifact-contract path, applied at this research seam.
                health = json.loads(urllib.request.urlopen(
                    engine_url.rstrip("/") + "/health", timeout=10).read())
                n_embd = int(health.get("n_embd") or 0)
                j_dim = next(iter(J_by_layer.values())).shape[0] if J_by_layer else 0
                if n_embd and j_dim and j_dim != n_embd:
                    msg = (f"sidecar J is {j_dim}-dim but the engine's model is {n_embd}-dim "
                           "(a different model's artifact)")
                    if screen_mode == "jlens":
                        return {"ok": False, "blocked": f"jlens screen requested but the sidecar "
                                                        f"does not qualify: {msg}"}
                    J_by_layer = None
                    screen_note = f"{msg} -- downgraded to the mean-ablation screen"
            except (FileNotFoundError, ValueError, OSError) as e:
                if screen_mode == "jlens":
                    return {"ok": False, "blocked": f"jlens screen requested but no sidecar "
                                                    f"qualifies: {e}"}
                screen_note = (f"no J-lens sidecar ({type(e).__name__}) -- downgraded to the "
                               "mean-ablation screen; legibility unavailable")
        concept_ids = {y_piece.strip() or repr(y_piece): y_id}
        concept_notes = []
        dirs_by_layer = None
        if J_by_layer:
            # The sidecar loaded and its d_model qualified, but the jlens screen still needs the
            # engine to serve unembed rows (/jlens). A build that doesn't offer that surface (e.g.
            # the --no-flash-attn knockout engine) 400s HERE, after qualification. In "auto" mode
            # that must be the documented LOUD DOWNGRADE to ablate, not a hard failure -- otherwise
            # causal-trace breaks out-of-the-box for anyone with a stale ~/.clozn/jlens.
            try:
                layers = sorted(J_by_layer.keys())
                for w in budget.extra_concepts:
                    cid, err = _resolve_concept(engine_url, w)
                    if cid is None:
                        concept_notes.append({"concept": w, "skipped": err})
                    else:
                        concept_ids[w] = cid
                dirs_by_layer = {}
                for L in layers:
                    dirs_by_layer[L] = {}
                    for label, cid in concept_ids.items():
                        w_row = _unembed_row(engine_url, cid)
                        dirs_by_layer[L][label] = dir_c_from_row(w_row, L, J_by_layer)
            except Exception as e:  # engine could not serve the jlens screen after qualification
                if screen_mode == "jlens":
                    return {"ok": False, "blocked": f"jlens screen requested but the engine could "
                                                    f"not serve it ({type(e).__name__}): {e}"}
                J_by_layer = None
                dirs_by_layer = None
                concept_ids = {y_piece.strip() or repr(y_piece): y_id}
                screen_note = ((screen_note + "; " if screen_note else "") +
                               f"jlens screen failed on the engine ({type(e).__name__}) -- "
                               "downgraded to the mean-ablation screen")
        if not J_by_layer:
            # No sidecar: a depth spread over the model's own layers. Capture (l_out-<il>) is
            # valid on layers [1, n_layer-2] -- the last layer only materializes logit rows
            # (inp_out_ids) and would silently capture nothing (the engine 400s on it).
            health = json.loads(urllib.request.urlopen(
                engine_url.rstrip("/") + "/health", timeout=10).read())
            n_layer = int(health["n_layer"])
            layers = sorted({max(1, min(n_layer - 2, int(round(n_layer * f))))
                             for f in (0.15, 0.4, 0.65, 0.9)})
            if budget.extra_concepts:
                concept_notes.append({"concept": ",".join(budget.extra_concepts),
                                      "skipped": "ablation screen has no directions; extra "
                                                 "concepts are unused in this mode"})
        screen_used = "jlens_dirs" if J_by_layer else "mean_ablation"  # after any auto-downgrade

        # ---- S0: capture every site's residual (chunked) + mean rows -------------------------
        H_by_layer = {L: None for L in layers}
        for start in range(0, sites_end, budget.capture_chunk):
            chunk = list(range(start, min(start + budget.capture_chunk, sites_end)))
            r = _score(engine_url, prompt, cont_ids,
                       capture={"layers": layers, "positions": chunk})
            cap = r.get("captured") or {}
            for L in layers:
                rows = cap.get(str(L)) or {}
                for p in chunk:
                    if str(p) in rows:
                        row = np.asarray(rows[str(p)], dtype=np.float32)
                        if H_by_layer[L] is None:
                            H_by_layer[L] = np.zeros((sites_end, len(row)), dtype=np.float32)
                        H_by_layer[L][p] = row
        missing = [L for L in layers if H_by_layer[L] is None]
        if missing:
            return {"ok": False, "blocked": f"capture returned no rows for layer(s) {missing}"}
        mean_rows = {L: H_by_layer[L].mean(axis=0) for L in layers}
        force = [(L, sites_end - 1) for L in layers] + [(L, n_p - 1) for L in layers]

        def arm(write_specs) -> float:
            r = _score(engine_url, prompt, cont_ids, write=write_specs)
            d_y = base_lp - _target_logprob(r, target_idx)    # positive = pushed TOWARD y
            if cont_ids_foil is None:
                return d_y
            # contrastive: subtract the foil's own delta under the SAME write, so we measure the
            # change in the y-vs-foil gap. A site that moves y and the foil equally -> ~0 (shared
            # answer-category scaffolding, correctly rejected). A site that prefers y -> positive.
            rf = _score(engine_url, prompt, cont_ids_foil, write=write_specs)
            d_foil = base_lp_foil - _target_logprob(rf, target_idx)
            return d_y - d_foil

        if dirs_by_layer:
            candidates = screen_candidates(H_by_layer, dirs_by_layer, budget.max_candidates, force)
        else:
            # Mean-ablation grid screen: nominate sites by their own measured delta. Grid strided
            # so the whole sweep stays within ~budget.ablate_screen_arms arms; forced sites always
            # included. screen_score here IS a causal delta (unlike the jlens projection score).
            per_layer = max(1, budget.ablate_screen_arms // max(1, len(layers)))
            stride = max(1, -(-sites_end // per_layer))          # ceil division
            grid = sorted({p for p in range(0, sites_end, stride)} |
                          {p for _, p in force if 0 <= p < sites_end})
            scored = []
            # Screen arms batch through /score `arms` when the engine offers it -- the
            # APPROXIMATE regime (measured: deltas differ from sequential by up to ~0.19 nats,
            # deterministic within-regime). Legitimate HERE and only here: the screen merely
            # NOMINATES sites; S1 re-measures every survivor sequentially, so a screen-regime
            # error can only shuffle nominations near ties, never touch a claimed number.
            # Chunk size 15 = 14 write arms + 1 in-batch baseline, under the engine's 16-seq cap.
            all_sites = [(L, p) for L in layers for p in grid]
            use_arms = bool(json.loads(urllib.request.urlopen(
                engine_url.rstrip("/") + "/health", timeout=10).read())
                .get("capabilities", {}).get("score_arms")) if all_sites else False
            if cont_ids_foil is not None:
                use_arms = False   # batched screen scores one continuation; contrast needs the foil too
            if use_arms:
                CHUNK = 14
                for i in range(0, len(all_sites), CHUNK):
                    chunk = all_sites[i:i + CHUNK]
                    arms = [{}] + [{"write": {"layer": L, "positions": [p],
                                              "values": mean_rows[L].tolist()}}
                                   for L, p in chunk]
                    r = _score(engine_url, prompt, cont_ids, arms=arms)
                    lps = [a["tokens"][target_idx]["logprob"] for a in r["arms"]]
                    for (L, p), lp in zip(chunk, lps[1:]):
                        scored.append({"layer": L, "pos": p,
                                       "screen_score": abs(lps[0] - lp), "concept": None})
            else:
                for L, p in all_sites:
                    d = arm({"layer": L, "positions": [p], "values": mean_rows[L].tolist()})
                    scored.append({"layer": L, "pos": p, "screen_score": abs(d),
                                   "concept": None})
            scored.sort(key=lambda c: -c["screen_score"])
            seen, candidates = set(), []
            for c in scored:
                key = (c["layer"], c["pos"])
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(c)
                if len(candidates) >= budget.max_candidates:
                    break
            for L, p in force:                                   # forced sites ride along, deduped
                if (L, p) not in seen and 0 <= p < sites_end:
                    seen.add((L, p))
                    candidates.append({"layer": L, "pos": p, "screen_score": 0.0,
                                       "concept": None})

        # ---- S1: solo arms + controls --------------------------------------------------------
        for c in candidates:
            L, p = c["layer"], c["pos"]
            h = H_by_layer[L][p]
            c["delta_full"] = arm({"layer": L, "positions": [p],
                                   "values": mean_rows[L].tolist()})
            if dirs_by_layer:
                label = c["concept"] or next(iter(dirs_by_layer[L]))
                d = dirs_by_layer[L][label]
                c["delta_dir"] = arm({"layer": L, "positions": [p],
                                      "values": directional_ablate(h, d).tolist()})
                c["name"] = label
                # legibility: how much of the site's full-ablation mass the NAMED direction carries
                c["legibility"] = (c["delta_dir"] / c["delta_full"]) if abs(c["delta_full"]) > 1e-9 else None
            else:
                c["delta_dir"] = None
                c["name"] = y_piece.strip() or repr(y_piece)
                c["legibility"] = None      # no named direction exists in ablation-screen mode

        controls = []
        d_model = len(mean_rows[layers[0]])
        ctl_sites = candidates[: budget.n_random_dir]
        for c in ctl_sites:                       # random-equal-norm direction, SAME sites
            L, p = c["layer"], c["pos"]
            d_r = rng.standard_normal(d_model).astype(np.float32)
            d_r /= np.linalg.norm(d_r)
            controls.append({"kind": "random_direction", "layer": L, "pos": p,
                             "delta": arm({"layer": L, "positions": [p],
                                           "values": directional_ablate(H_by_layer[L][p], d_r).tolist()})})
        cand_keys = {(c["layer"], c["pos"]) for c in candidates}
        y_label = next(iter(concept_ids))
        for _ in range(budget.n_random_site):     # the CLAIMED intervention, random non-candidate sites
            for _try in range(20):
                L = layers[int(rng.integers(len(layers)))]
                p = int(rng.integers(sites_end))
                if (L, p) not in cand_keys:
                    break
            if dirs_by_layer:
                ctl_values = directional_ablate(H_by_layer[L][p], dirs_by_layer[L][y_label]).tolist()
            else:
                # ablation-screen mode: the claimed intervention is FULL mean-ablation, so the
                # matched site-control is full mean-ablation at a random non-candidate site --
                # controls must mirror the intervention actually being claimed.
                ctl_values = mean_rows[L].tolist()
            controls.append({"kind": "random_site", "layer": L, "pos": p,
                             "delta": arm({"layer": L, "positions": [p], "values": ctl_values})})
        ctl_deltas = [c["delta"] for c in controls]
        floor = noise_floor(ctl_deltas)
        ctl_max = float(np.max(np.abs(ctl_deltas)))
        # Survival rule. Absolute mode: a site is real if |delta| clears the noise floor.
        # Contrastive mode: DIRECTIONAL -- a site is part of y's circuit only if it positively
        # SUPPORTS y over the foil (delta_full > floor). This is what makes the screen-null
        # discriminate: a token the model isn't selecting has its supporting sites carry NEGATIVE
        # contrastive delta (they support the foil), so it collapses to NO_CAUSAL_NODES instead of
        # inheriting the shared answer-category scaffolding. Sites with delta_full < -floor are
        # recorded as foil-supporting (opposes_target), never counted as y's.
        def _supports(d):
            return (d > floor) if cont_ids_foil is not None else (abs(d) > floor)
        survivors = [c for c in candidates if _supports(c["delta_full"])]
        for c in candidates:
            c["survived"] = _supports(c["delta_full"])
            if cont_ids_foil is not None:
                c["opposes_target"] = c["delta_full"] < -floor
            # Per-node separation from the STRONGEST control arm — the number that says how much
            # to trust this node, published so nobody has to take "survived" on faith. A 16-prompt
            # battery found real traces spanning 1.3x to 218x on the SAME trace: the strong nodes
            # are overwhelming, the tail is barely distinguishable from a random intervention.
            # Tiers (documented, not silently applied): strong >= 3x, weak 1-3x, marginal <= 1x.
            c["control_ratio"] = (abs(c["delta_full"]) / ctl_max) if ctl_max > 1e-12 else None
            r_ = c["control_ratio"]
            c["strength"] = ("strong" if (r_ is not None and r_ >= 3.0)
                             else "weak" if (r_ is not None and r_ > 1.0) else "marginal")
            c["marginal"] = c["survived"] and c["strength"] == "marginal"
            ok_margin = not np.isnan(margin) and margin > 0
            c["margin_flip_predicted"] = bool(ok_margin and c["delta_full"] > margin)
            c["margin_flip_observed"] = None      # S4 (generation arms) fills this in

        # ---- S2: joint arm + accounting ------------------------------------------------------
        acct = {"delta_total": None, "sum_solo": None, "interaction_gap": None}
        if survivors:
            joint = group_joint_writes(survivors, mean_rows)
            acct = accounting([c["delta_full"] for c in survivors], arm(joint))
        verdict = controls_verdict([c["delta_full"] for c in survivors], ctl_deltas)

        # ---- S3: path patching (edges among the SOLID survivors; marginal parents give
        # meaningless routed fractions) --------------------------------------------------------
        solid = [c for c in survivors if not c.get("marginal")]
        edges = []
        for A, B in edge_candidates(solid, budget.max_edges):
            lA, pA, lB, pB = A["layer"], A["pos"], B["layer"], B["pos"]
            # arm 1: ablate A, capture B's state under do(A)
            r = _score(engine_url, prompt, cont_ids,
                       write={"layer": lA, "positions": [pA], "values": mean_rows[lA].tolist()},
                       capture={"layers": [lB], "positions": [pB]})
            h_doA = ((r.get("captured") or {}).get(str(lB)) or {}).get(str(pB))
            if h_doA is None:
                continue
            # arm 2: patch ONLY B to that state — A's effect routed exclusively through B
            delta_edge = arm({"layer": lB, "positions": [pB], "values": h_doA})
            # shuffled-edge control: B's state under an UNRELATED same-layer ablation A' (a random
            # site that can still reach B, i.e. pos <= pB). The routed fraction should collapse.
            pS = pA
            for _try in range(20):
                cand_p = int(rng.integers(pB + 1))
                if cand_p != pA and (lA, cand_p) not in {(c["layer"], c["pos"]) for c in solid}:
                    pS = cand_p
                    break
            rs = _score(engine_url, prompt, cont_ids,
                        write={"layer": lA, "positions": [pS], "values": mean_rows[lA].tolist()},
                        capture={"layers": [lB], "positions": [pB]})
            h_doS = ((rs.get("captured") or {}).get(str(lB)) or {}).get(str(pB))
            delta_shuf = arm({"layer": lB, "positions": [pB], "values": h_doS}) if h_doS else None
            frac = delta_edge / A["delta_full"] if abs(A["delta_full"]) > 1e-9 else None
            # A cross-position routed_fraction is a LOWER BOUND, not a measurement (see
            # notes/CIRCUIT_TRACER_DESIGN.md §5f): patching ONE destination site leaves the source
            # position un-ablated at later layers, so the network re-imports the information
            # downstream and the effect collapses. Holding the whole destination column does not
            # rescue it either, because llama.cpp materializes only the logit rows at the last
            # layer (inp_out_ids), leaving one unpatchable layer to re-supply the effect. Measured:
            # 0.0% routed at every held depth for a late-layer cross-position source, which is
            # physically impossible as a true effect size. SAME-COLUMN edges are unaffected (and
            # close to structural — patching a position at a later layer does hold everything
            # flowing through it), which is why every edge claimed so far is same-column.
            same_column = (pA == pB)
            edges.append({
                "from": [lA, pA], "to": [lB, pB],
                "delta_edge": delta_edge, "routed_fraction": frac,
                "same_column": same_column,
                "fraction_is_lower_bound": not same_column,
                "shuffled_site": [lA, pS], "delta_shuffled": delta_shuf,
                # claimed: the routed effect beats the noise floor AND at least doubles the
                # shuffled control — otherwise it is reported but NOT claimed as a real edge.
                "claimed": bool(abs(delta_edge) > floor and delta_shuf is not None
                                and abs(delta_edge) >= 2.0 * abs(delta_shuf)),
            })

        # ---- S4: margin-test generation arms (predicted-vs-OBSERVED) -------------------------
        # The graph PREDICTED (from two published numbers: delta_full and the baseline margin)
        # which node ablations flip the actually-generated token. Now we run the generations and
        # score the prediction. Baseline must itself be greedy-reproducible or the behavioral
        # tier is inapplicable (reported, not fudged).
        gen_tier = {"ran": False, "baseline_greedy": None}
        if budget.run_s4 and solid:
            b = _complete(engine_url, prompt, target_idx + 1, reference=cont_ids)
            base_ok = not bool(b.get("diverged"))
            gen_tier = {"ran": True, "baseline_greedy": base_ok}
            if base_ok:
                for c in solid:
                    g = _complete(engine_url, prompt, target_idx + 1, reference=cont_ids,
                                  write={"layer": c["layer"], "positions": [c["pos"]],
                                         "values": mean_rows[c["layer"]].tolist()})
                    if g.get("diverged") and int(g.get("diverged_at", -1)) == target_idx:
                        c["margin_flip_observed"] = True          # flipped exactly at the target
                    elif g.get("diverged"):
                        c["margin_flip_observed"] = "diverged_early"  # upstream token changed first
                    else:
                        c["margin_flip_observed"] = False

        # the 2x2 the graph gets judged on: predictions were published BEFORE the generations ran
        observed = [c for c in solid if isinstance(c.get("margin_flip_observed"), bool)]
        scorecard = {
            "predicted_flips": sum(1 for c in candidates if c["margin_flip_predicted"]),
            "observed_flips": sum(1 for c in observed if c["margin_flip_observed"]) if gen_tier["ran"] else None,
            "correct_predictions": sum(1 for c in observed
                                       if c["margin_flip_predicted"] == c["margin_flip_observed"]),
            "wrong_predictions": sum(1 for c in observed
                                     if c["margin_flip_predicted"] != c["margin_flip_observed"]),
            "diverged_early": sum(1 for c in solid
                                  if c.get("margin_flip_observed") == "diverged_early"),
            "generation_tier": gen_tier,
        }

        return {
            "ok": True,
            "target": {"pos": target_idx, "abs_pos": t_abs, "id": y_id, "piece": y_piece,
                       "baseline_logprob": base_lp, "margin": None if np.isnan(margin) else margin},
            "nodes": [c for c in candidates if c["survived"]],
            "edges": edges,
            "all_candidates": candidates,          # dead ones too: the screen's over-nomination, published
            "accounting": {**acct,
                           "screened_sites": sites_end * len(layers),
                           "candidates": len(candidates), "survivors": len(survivors)},
            "controls": {"arms": controls, "noise_floor": floor,
                         "median_abs": float(np.median(np.abs(ctl_deltas))),
                         "max_abs": float(np.max(np.abs(ctl_deltas))), "verdict": verdict},
            "prediction_scorecard": scorecard,
            "config": {"layers": layers, "concepts": list(concept_ids),
                       "concept_notes": concept_notes, "seed": seed,
                       "screen_mode": screen_used,
                       "screen_note": screen_note,
                       "budget": {"max_candidates": budget.max_candidates,
                                  "n_random_dir": budget.n_random_dir,
                                  "n_random_site": budget.n_random_site,
                                  "max_edges": budget.max_edges, "run_s4": budget.run_s4,
                                  "ablate_screen_arms": budget.ablate_screen_arms},
                       "units": ("delta = [lp_y - lp_foil](baseline) - [lp_y - lp_foil](ablated), "
                                 "teacher-forced CONTRASTIVE" if cont_ids_foil is not None
                                 else "delta = logprob_y(baseline) - logprob_y(ablated), teacher-forced"),
                       "scoring": "contrastive" if cont_ids_foil is not None else "absolute",
                       "contrast": ({"id": foil_id, "piece": foil_piece,
                                     "baseline_logprob": base_lp_foil}
                                    if cont_ids_foil is not None else None),
                       "boundary_approximate": bool(base.get("boundary_approximate", False))},
        }
    except Exception as e:  # engine down, sidecar missing, contract drift -- labeled, never raised
        return {"ok": False, "blocked": f"{type(e).__name__}: {e}"}


def save_receipt(receipt: dict, path: str) -> str:
    """Atomic write (tmp + rename -- the house rule for user-data writers)."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
