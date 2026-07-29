"""analysis/pair_compatibility.py -- the SHARED, versioned model-pair compatibility contract
(`clozn.pair-compatibility.v1`). Several features need to answer "can I do X across these two model
files": `diff-model` (per-token teacher-forced diffing), Experiments, mechanistic diff, Studio Compare,
and causal bisect (residual transplant). Before this module each reimplemented its own preflight; this
is the one place that logic lives.

WHY A CONTRACT AND NOT JUST A BOOLEAN
--------------------------------------
"Are these two models compatible" is not one question -- it is several, with DIFFERENT requirements:

  * Per-token teacher-forced comparison (diff-model's whole reason for existing -- see that module's
    docstring's "THE TRAP") is meaningless unless the two models tokenize IDENTICALLY: a token id (or
    even a matching id with a different piece string) means something different in each vocabulary.
  * A residual transplant (writing an activation vector captured from model A's residual stream into
    model B's forward pass, e.g. causal bisect) is a DIFFERENT question with a DIFFERENT hard
    requirement: `hidden_size` (the residual width) must match EXACTLY. The engine's write validation
    (`GgmlAdapter::add_write_state`, engine/core/src/model_ggml.cpp:452-464) rejects any write where
    `values.size() != positions.size() * n_embd` of the TARGET engine, and there is no projection layer
    anywhere in the codebase to bridge two different widths. This is a mechanical fact about the C++
    write path, not a policy choice -- so "same residual width" is a hard gate for transplant operations,
    never a nicety. That same function also fixes which layers are writable at all: `il <= 0 || il >=
    n_layer_` is rejected, so writable layers are `[1, n_layer)` -- layer 0 (embeddings) and the final
    layer are never writable. See `writable_layer_range` / the `writable_layers` document field.

A pair can therefore be fine for one operation and refused for the other in the same breath -- a
tokenizer mismatch blocks per-token comparison outright while leaving residual transplant untouched (it
only cares about `hidden_size`), and a `hidden_size` mismatch blocks transplant while leaving per-token
comparison untouched (it only cares about the tokenizer). Collapsing that into one bare boolean would
either over-refuse or under-refuse depending which operation the caller actually wanted. So every
dimension below reports an explicit STATE (never a bare bool), and the document's `verdict.operations`
answers "may I do per-token comparison?" / "may I do residual transplant?" / "may I do a head
transplant?" as separate, reasoned yes/no answers a caller can query independently.

A per-head (`kqv_out-<il>` / `head_write`) transplant is a THIRD such question, with its OWN hard
requirement, gated independently of the other two: `head_count` (the query-head count -- kqv_out rows
are per-Q-head even under GQA, per `clozn.receipts.hook_vocabulary`'s GQA note) must match exactly,
because a head INDEX only refers to the same conceptual slice on both models when they agree on how many
slices `kqv_out` is divided into. A `head_count` mismatch blocks `head_transplant` and nothing else --
it does not touch `per_token_comparison` (tokenizer-only) or `residual_transplant` (hidden_size-only),
mirroring exactly how a `hidden_size` mismatch today blocks only `residual_transplant`. `head_count_kv`
(the key/value-head count, which differs from `head_count` under GQA) is carried on `gguf_identity` for
completeness but is deliberately NOT part of this gate: `kqv_out`'s row structure is dimensioned by
query heads, not KV heads, so KV-head count is not mechanically relevant to whether a `head_write` index
means the same thing on both sides.

TWO WAYS TO KNOW A DIMENSION
-----------------------------
Some dimensions only have one honest source. `architecture` / `layer_count` / `hidden_size` /
`vocab_size` come from the GGUF header alone (`clozn.artifacts.contracts.gguf_identity`) -- there is no
"probe" for them, so `assess()` always compares them structurally.

`tokenizer` and `template` have TWO possible sources, and probe truth wins when both are available:

  * STATIC (method "hash") -- `gguf_identity` already digests every `tokenizer.*` GGUF metadata key into
    `tokenizer_sha256`, and the chat template into `chat_template_sha256`. Comparing those digests needs
    no engine, no GPU, no boot -- this is what Experiments / Studio Compare / causal bisect use when they
    only have two file paths, not two live engines.
  * BEHAVIORAL (method "probe") -- `check_tokenizer_compat` / `check_template_match` (moved here
    unmodified from `clozn.cli.commands.diff_model`, which now delegates to them) actually round-trip
    fixed probe strings / a canonical conversation through two live engines. This is strictly stronger
    ground truth (it catches a runtime tokenizer bug a metadata hash would miss) and is what `diff-model`
    supplies once its two engines are already booted for scoring anyway.

`assess()` takes the probe result as an optional override; omit it for a pure static, no-engine
assessment, or pass `check_tokenizer_compat(...)`'s / `check_template_match(...)`'s own return value
straight through when engines are available.

ROADMAP RULE 2 ("omit, never null-pad")
----------------------------------------
A dimension whose value could not be honestly measured on one or both sides gets `"state": "unknown"`
and simply OMITS `value_a`/`value_b` for the missing side(s) -- never `null`. This module is stdlib-only
throughout (no lazy imports needed: every import here, including `clozn.artifacts.contracts`, is either
stdlib or another stdlib-only clozn submodule); see `clozn/artifacts/contracts.py`'s own docstring.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from clozn import schemas
from clozn.artifacts.contracts import gguf_identity

SCHEMA_VERSION = "clozn.pair-compatibility.v1"

# =========================================================================== tokenizer preflight (probe)
# Moved here verbatim from clozn/cli/commands/diff_model.py -- that module now imports these names rather
# than keeping its own copies (see this module's docstring, "TWO WAYS TO KNOW A DIMENSION").

# A short, fixed prefix every probe is scored after -- keeps the /score call shape identical to a real
# forced-scoring call (prompt + continuation) without needing chat messages/apply_template at all: the
# preflight's whole point is to compare TOKENIZATION, which is orthogonal to chat templating.
TOKENIZER_PROBE_PREFIX = "Consider the following: "

# ~4 diverse probes (per the design): a plain English sentence, digits/arithmetic, a code snippet with
# symbols, and a unicode/multilingual string -- chosen to stress different tokenizer code paths (BPE
# merges on common words, digit-splitting behavior, punctuation/whitespace-sensitive code tokens, and
# multi-byte/non-Latin scripts + emoji), not for statistical coverage.
TOKENIZER_PROBES = [
    ("plain_english", "The quick brown fox jumps over the lazy dog while the sun sets slowly."),
    ("digits_arithmetic", "12345 + 67890 = 80235, and roughly 3.14159 times 2 is 6.28318."),
    ("code_snippet", "def add(a, b):\n    return a + b  # sum two numbers; edge case: a is None"),
    ("unicode_multilingual", "Café naïve résumé — 日本語 中文"
                             "测试 \U0001F680"),
]


def check_tokenizer_compat(sub_a, sub_b) -> dict:
    """The mandatory preflight (see diff-model's module docstring's THE TRAP): for each of
    `TOKENIZER_PROBES`, call EACH engine's own `.score(prompt=TOKENIZER_PROBE_PREFIX, continuation=probe,
    topk=0)` -- letting that engine's OWN tokenizer segment the probe text into ids, exactly as it would
    any real continuation -- and compare the returned token id sequence AND piece-string sequence
    position by position (plain list equality already does this: different order, different length, or
    any single differing element all read as a mismatch). `sub_a`/`sub_b` are
    `quant_check._EngineScoreSub`-shaped (or anything exposing `.engine.score(...)`) -- production wraps
    two real `EngineClient`s, tests wrap fakes.

    Returns {"compatible": bool, "probes": [{"probe", "text", "ids_match", "pieces_match", "n_a", "n_b"},
    ...]} -- "compatible" is True only if EVERY probe's ids AND pieces matched. Never raises: a probe
    whose scoring blew up on either arm is recorded as a hard mismatch (ids_match/pieces_match False),
    never silently skipped, since an engine that can't even score a plain probe string is itself grounds
    for refusing to diff."""
    probes_out = []
    all_compatible = True
    for name, text in TOKENIZER_PROBES:
        try:
            resp_a = sub_a.engine.score(prompt=TOKENIZER_PROBE_PREFIX, continuation=text, topk=0)
            resp_b = sub_b.engine.score(prompt=TOKENIZER_PROBE_PREFIX, continuation=text, topk=0)
            toks_a = resp_a.get("tokens", []) if isinstance(resp_a, dict) else []
            toks_b = resp_b.get("tokens", []) if isinstance(resp_b, dict) else []
            ids_a = [t.get("id") for t in toks_a if isinstance(t, dict)]
            ids_b = [t.get("id") for t in toks_b if isinstance(t, dict)]
            pieces_a = [t.get("piece") for t in toks_a if isinstance(t, dict)]
            pieces_b = [t.get("piece") for t in toks_b if isinstance(t, dict)]
            ids_match = bool(ids_a) and ids_a == ids_b
            pieces_match = bool(pieces_a) and pieces_a == pieces_b
        except Exception:
            ids_a, ids_b, ids_match, pieces_match = [], [], False, False
        if not (ids_match and pieces_match):
            all_compatible = False
        probes_out.append({"probe": name, "text": text, "ids_match": ids_match,
                           "pieces_match": pieces_match, "n_a": len(ids_a), "n_b": len(ids_b)})
    return {"compatible": all_compatible, "probes": probes_out}


def tokenizer_refusal_message(compat: dict) -> str:
    """The refusal text for an incompatible tokenizer preflight -- states plainly why per-token diffing is
    meaningless here (not just THAT it failed), and suggests the fix (a same-family pair)."""
    bad = [p["probe"] for p in (compat.get("probes") or []) if not (p.get("ids_match") and p.get("pieces_match"))]
    return (
        "diff-model refuses: the reference and candidate do not tokenize identically (failed probe(s): "
        f"{', '.join(bad) or 'unknown'}). Per-token teacher-forced diffing is meaningless across different "
        "tokenizers -- a token id (or even a matching id with a different piece string) means something "
        "different in each vocabulary, so a per-position 'preserved' or 'flipped' verdict would be comparing "
        "unrelated units, not the same model's behavior under two conditions. This usually means the two "
        "files are not close enough in lineage to diff this way (e.g. a merge stapled together checkpoints "
        "from different tokenizer vintages). Compare same-tokenizer-family pairs instead -- a base model "
        "and its own fine-tune/LoRA, or two checkpoints of the same run."
    )


# ============================================================================= template policy (probe)

CANONICAL_TEMPLATE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

TEMPLATE_DIFFER_REFERENCE_CAVEAT = (
    "chat templates differ between the reference and the candidate; both arms were scored on the "
    "REFERENCE model's rendering (the default policy), so this diff isolates WEIGHTS -- the candidate is "
    "being evaluated slightly off its own deployed chat format. Pass --own-templates to measure the "
    "candidate's actual deployed behavior instead (weights AND template both in play)."
)

TEMPLATE_OWN_CAVEAT = (
    "--own-templates: each model rendered its OWN chat template. This measures the candidate's DEPLOYED "
    "behavior including template differences, not an isolated weights diff -- a divergence below could "
    "come from the template change, the weights change, or both, and this run cannot separate them."
)

# The generic caveat attached when a caller (not diff-model, which always supplies its own wording above)
# asks for an assessment of a pair whose templates differ but has no opinion on rendering policy.
_GENERIC_TEMPLATE_CAVEAT = (
    "chat templates differ between the two models; behavior compared under one model's rendering may not "
    "reflect the other model's own deployed chat format."
)


def check_template_match(sub_a, sub_b) -> dict:
    """Compares `apply_template` output on a canonical 2-message conversation ([system: "You are a helpful
    assistant.", user: "What is the capital of France?"]) under both engines. Returns {"match": bool,
    "rendering_a", "rendering_b"} -- "match" is True only when both renders succeeded and are byte-
    identical. Never raises: a template render that fails on either side (e.g. no embedded chat template)
    counts as a mismatch, since the two arms plainly are not rendering the same way in that case either."""
    try:
        rendering_a = sub_a.engine.apply_template(list(CANONICAL_TEMPLATE_MESSAGES))
    except Exception:
        rendering_a = None
    try:
        rendering_b = sub_b.engine.apply_template(list(CANONICAL_TEMPLATE_MESSAGES))
    except Exception:
        rendering_b = None
    match = rendering_a is not None and rendering_a == rendering_b
    return {"match": match, "rendering_a": rendering_a, "rendering_b": rendering_b}


# =================================================================================== structural facts

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_ref(identity: Mapping[str, object], label: str | None) -> dict:
    """A `model_a`/`model_b` block -- identifying facts only, each key present only when known (roadmap
    rule 2: omit, never null-pad)."""
    out: dict = {}
    if label:
        out["label"] = label
    filename = identity.get("filename")
    if filename:
        out["filename"] = filename
    sha256 = identity.get("sha256")
    if sha256:
        out["sha256"] = sha256
    return out


def _dimension(value_a, value_b) -> dict:
    """A same/differs/unknown structural dimension (architecture, layer_count, hidden_size, vocab_size):
    "unknown" whenever either side could not be measured, else a direct equality check. `value_a`/
    `value_b` are included only for the side(s) actually known."""
    if value_a is None or value_b is None:
        state = "unknown"
    elif value_a == value_b:
        state = "same"
    else:
        state = "differs"
    out = {"state": state}
    if value_a is not None:
        out["value_a"] = value_a
    if value_b is not None:
        out["value_b"] = value_b
    return out


def _tokenizer_dimension(*, identity_a: Mapping[str, object], identity_b: Mapping[str, object],
                          probe: dict | None) -> dict:
    """tokenizer uses "exact" rather than "same" (see module docstring) -- a stronger word for the one
    dimension with a hard downstream refusal. `probe` (an optional `check_tokenizer_compat(...)` result)
    is BEHAVIORAL ground truth and wins when supplied; otherwise falls back to comparing `gguf_identity`'s
    `tokenizer_sha256` digest, which needs no live engine."""
    if probe is not None:
        state = "exact" if probe.get("compatible") else "differs"
        out = {"state": state, "method": "probe"}
        failed = [p["probe"] for p in (probe.get("probes") or [])
                 if not (p.get("ids_match") and p.get("pieces_match"))]
        if failed:
            out["failed_probes"] = failed
        return out

    hash_a = identity_a.get("tokenizer_sha256")
    hash_b = identity_b.get("tokenizer_sha256")
    if not hash_a or not hash_b:
        return {"state": "unknown", "method": "unknown"}
    return {"state": "exact" if hash_a == hash_b else "differs", "method": "hash"}


def _template_dimension(*, identity_a: Mapping[str, object], identity_b: Mapping[str, object],
                         match: bool | None, policy_applied: str | None, caveat: str | None) -> dict:
    """`match` (an optional `check_template_match(...)["match"]` value) is BEHAVIORAL ground truth and
    wins when supplied; otherwise falls back to comparing `gguf_identity`'s `chat_template_sha256`
    digest. `policy_applied`/`caveat` are the caller's own rendering-policy decision (e.g. diff-model's
    --own-templates) -- passed through untouched; a generic caveat is supplied when the templates differ
    and the caller has no policy opinion of its own."""
    if match is not None:
        state = "same" if match else "differs"
        method = "probe"
    else:
        hash_a = identity_a.get("chat_template_sha256")
        hash_b = identity_b.get("chat_template_sha256")
        if not hash_a or not hash_b:
            state, method = "unknown", "unknown"
        else:
            state = "same" if hash_a == hash_b else "differs"
            method = "hash"

    out = {"state": state, "method": method}
    if policy_applied is not None:
        out["policy_applied"] = policy_applied
    if caveat is not None:
        out["caveat"] = caveat
    elif state == "differs":
        out["caveat"] = _GENERIC_TEMPLATE_CAVEAT
    return out


def writable_layer_range(layer_count) -> dict | None:
    """The writable layer range for ONE model, mirroring `GgmlAdapter::add_write_state`'s own gate
    (`il <= 0 || il >= n_layer_` is rejected -- engine/core/src/model_ggml.cpp:452-464): layer 0 (raw
    embeddings, no `l_out` name) and the final layer are never writable, so the writable range is
    `[1, layer_count)`. Returns None when `layer_count` is not a positive int (unknown -- never a fake
    empty range)."""
    if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count <= 0:
        return None
    return {"min": 1, "max_exclusive": layer_count}


# =================================================================================== verdict + operations

def _build_verdict(*, tokenizer: dict, template: dict, architecture: dict, layer_count: dict,
                    hidden_size: dict, vocab_size: dict, head_count: dict) -> dict:
    reasons: list[str] = []

    if tokenizer["state"] == "differs":
        reasons.append("tokenizers differ: a token id (or a matching id with a different piece string) "
                       "means something different in each vocabulary.")
    elif tokenizer["state"] == "unknown":
        reasons.append("tokenizer compatibility could not be determined (no probe result and no "
                       "tokenizer_sha256 available on one or both sides).")

    if template["state"] == "differs":
        reasons.append("chat templates differ" + (f": {template['caveat']}" if template.get("caveat")
                                                   else " between the two models."))
    elif template["state"] == "unknown":
        reasons.append("chat template compatibility could not be determined.")

    if architecture["state"] == "differs":
        reasons.append(f"architectures differ ({architecture.get('value_a')!r} vs "
                       f"{architecture.get('value_b')!r}).")
    elif architecture["state"] == "unknown":
        reasons.append("architecture is unknown for at least one model.")

    if hidden_size["state"] == "differs":
        reasons.append(f"hidden_size differs ({hidden_size.get('value_a')} vs {hidden_size.get('value_b')}); "
                       "a residual transplant is mechanically impossible without a projection layer, which "
                       "clozn does not implement.")
    elif hidden_size["state"] == "unknown":
        reasons.append("hidden_size is unknown for at least one model; residual transplant cannot be "
                       "confirmed mechanically possible.")

    if layer_count["state"] == "differs":
        reasons.append(f"layer_count differs ({layer_count.get('value_a')} vs {layer_count.get('value_b')}).")
    elif layer_count["state"] == "unknown":
        reasons.append("layer_count is unknown for at least one model.")

    if vocab_size["state"] == "differs":
        reasons.append(f"vocab_size differs ({vocab_size.get('value_a')} vs {vocab_size.get('value_b')}).")
    elif vocab_size["state"] == "unknown":
        reasons.append("vocab_size is unknown for at least one model.")

    if head_count["state"] == "differs":
        reasons.append(f"head_count differs ({head_count.get('value_a')} vs {head_count.get('value_b')}); "
                       "a per-head kqv_out transplant is mechanically impossible -- a head index does not "
                       "refer to the same slice on models with different query-head counts.")
    elif head_count["state"] == "unknown":
        reasons.append("head_count is unknown for at least one model; head transplant cannot be confirmed "
                       "mechanically possible.")

    if tokenizer["state"] == "differs":
        overall = "incompatible"
    elif reasons:
        overall = "compatible_with_caveats"
    else:
        overall = "compatible"

    per_token_permitted = tokenizer["state"] == "exact"
    if per_token_permitted:
        per_token_reason = "tokenizers match exactly; per-token comparison is meaningful."
    elif tokenizer["state"] == "differs":
        per_token_reason = ("tokenizers differ -- a token id means something different in each vocabulary, "
                            "so a per-position comparison would be comparing unrelated units, not the same "
                            "model's behavior under two conditions.")
    else:
        per_token_reason = ("tokenizer compatibility is unknown; refusing per-token comparison until it is "
                            "confirmed (no silent degrade -- see docs/SEAMS.md rule 3).")

    transplant_permitted = hidden_size["state"] == "same"
    if transplant_permitted:
        transplant_reason = (f"hidden_size matches exactly ({hidden_size.get('value_a')}); the target "
                             "engine's write validation (values.size() == positions.size() * hidden_size) "
                             "can be satisfied.")
    elif hidden_size["state"] == "differs":
        transplant_reason = (f"hidden_size differs ({hidden_size.get('value_a')} vs "
                             f"{hidden_size.get('value_b')}); the target engine's write validation requires "
                             "an exact match and clozn has no projection layer to bridge different residual "
                             "widths.")
    else:
        transplant_reason = ("hidden_size is unknown for at least one model; residual transplant cannot be "
                             "confirmed mechanically possible.")

    head_transplant_permitted = head_count["state"] == "same"
    if head_transplant_permitted:
        head_transplant_reason = (f"head_count matches exactly ({head_count.get('value_a')}); a kqv_out "
                                  "head index refers to the same conceptual slice on both models.")
    elif head_count["state"] == "differs":
        head_transplant_reason = (f"head_count differs ({head_count.get('value_a')} vs "
                                  f"{head_count.get('value_b')}); a head index would not refer to the same "
                                  "slice on both models.")
    else:
        head_transplant_reason = ("head_count is unknown for at least one model; head transplant cannot be "
                                  "confirmed mechanically possible.")

    return {
        "overall": overall,
        "reasons": reasons,
        "operations": {
            "per_token_comparison": {"permitted": per_token_permitted, "reason": per_token_reason},
            "residual_transplant": {"permitted": transplant_permitted, "reason": transplant_reason},
            "head_transplant": {"permitted": head_transplant_permitted, "reason": head_transplant_reason},
        },
    }


# =========================================================================================== public API

def assess(identity_a: Mapping[str, object], identity_b: Mapping[str, object], *,
           label_a: str | None = None, label_b: str | None = None,
           tokenizer_compat: dict | None = None, template_match: bool | None = None,
           template_policy: str | None = None, template_caveat: str | None = None,
           generated_at: str | None = None, validate: bool = True) -> dict:
    """Build one `clozn.pair-compatibility.v1` document from two `gguf_identity(...)`-shaped mappings.

    Pure and model-free: `identity_a`/`identity_b` are plain dicts (real callers get them from
    `clozn.artifacts.contracts.gguf_identity`, but tests can hand-build synthetic ones -- see
    `assess_gguf_pair` for the path-based convenience wrapper that does the real file I/O).

    `tokenizer_compat`/`template_match` are optional BEHAVIORAL overrides -- the return values of
    `check_tokenizer_compat(...)` / `check_template_match(...)["match"]` respectively -- for callers who
    already have two live engines (diff-model). Omit both for a pure static, no-engine assessment driven
    entirely by GGUF header digests. `template_policy`/`template_caveat` are the caller's own rendering-
    policy decision (e.g. diff-model's "reference" vs "own"); a generic caveat is supplied automatically
    when templates differ and the caller has no policy opinion.

    `validate=True` (default) round-trips the built document through `clozn.schemas.validate` before
    returning -- this function IS the one producer of this schema, so a shape bug here should fail loudly
    at the call site, not ship silently to a consumer."""
    tokenizer = _tokenizer_dimension(identity_a=identity_a, identity_b=identity_b, probe=tokenizer_compat)
    template = _template_dimension(identity_a=identity_a, identity_b=identity_b, match=template_match,
                                   policy_applied=template_policy, caveat=template_caveat)
    architecture = _dimension(identity_a.get("architecture"), identity_b.get("architecture"))
    layer_count = _dimension(identity_a.get("layer_count"), identity_b.get("layer_count"))
    hidden_size = _dimension(identity_a.get("hidden_size"), identity_b.get("hidden_size"))
    vocab_size = _dimension(identity_a.get("vocab_size"), identity_b.get("vocab_size"))
    head_count = _dimension(identity_a.get("head_count"), identity_b.get("head_count"))

    writable_layers: dict = {}
    range_a = writable_layer_range(identity_a.get("layer_count"))
    if range_a is not None:
        writable_layers["model_a"] = range_a
    range_b = writable_layer_range(identity_b.get("layer_count"))
    if range_b is not None:
        writable_layers["model_b"] = range_b

    verdict = _build_verdict(tokenizer=tokenizer, template=template, architecture=architecture,
                             layer_count=layer_count, hidden_size=hidden_size, vocab_size=vocab_size,
                             head_count=head_count)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "model_a": _model_ref(identity_a, label_a),
        "model_b": _model_ref(identity_b, label_b),
        "tokenizer": tokenizer,
        "template": template,
        "architecture": architecture,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "head_count": head_count,
        "writable_layers": writable_layers,
        "verdict": verdict,
    }
    if validate:
        schemas.validate(doc)
    return doc


def assess_gguf_pair(path_a: str, path_b: str, *, label_a: str | None = None, label_b: str | None = None,
                     include_file_hash: bool = True, tokenizer_compat: dict | None = None,
                     template_match: bool | None = None, template_policy: str | None = None,
                     template_caveat: str | None = None) -> dict:
    """Convenience wrapper: read both GGUFs via `gguf_identity` and `assess()` them. This is the entry
    point for callers who only have two file paths (Experiments, Studio Compare, causal bisect) -- no
    engine boot required. `include_file_hash=False` skips the whole-file SHA-256 (fast inventory only;
    `model_a`/`model_b`.`sha256` will simply be omitted from the document)."""
    identity_a = gguf_identity(path_a, include_file_hash=include_file_hash)
    identity_b = gguf_identity(path_b, include_file_hash=include_file_hash)
    return assess(identity_a, identity_b, label_a=label_a, label_b=label_b,
                 tokenizer_compat=tokenizer_compat, template_match=template_match,
                 template_policy=template_policy, template_caveat=template_caveat)


def may_per_token_compare(report: dict) -> bool:
    """True iff `report` (an `assess(...)` document) permits per-token teacher-forced comparison."""
    return bool(report.get("verdict", {}).get("operations", {})
               .get("per_token_comparison", {}).get("permitted"))


def may_residual_transplant(report: dict) -> bool:
    """True iff `report` (an `assess(...)` document) permits a residual transplant."""
    return bool(report.get("verdict", {}).get("operations", {})
               .get("residual_transplant", {}).get("permitted"))


def may_head_transplant(report: dict) -> bool:
    """True iff `report` (an `assess(...)` document) permits a per-head (`kqv_out`/`head_write`)
    transplant -- gated on `head_count` matching exactly, independent of `residual_transplant`'s own
    `hidden_size` gate (see this module's docstring)."""
    return bool(report.get("verdict", {}).get("operations", {})
               .get("head_transplant", {}).get("permitted"))
