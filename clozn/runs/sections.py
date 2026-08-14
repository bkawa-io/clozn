"""sections.py -- the ablatable prompt-section manifest (foundation for per-section influence).

WHY THIS EXISTS. A run's prompt is rarely one undifferentiated blob: it's a system prompt, maybe a RAG
context dump with several retrieved documents, maybe an applied memory card, then the user's actual
question. "Which PART of the prompt actually shaped the reply" is a question the (separately-owned)
receipts system wants to answer by ablation -- but ablation needs something concrete to ablate: a set of
named, char-addressable spans over the run's own recorded text. That's what a "section" is here. This
module's only job is producing that manifest; it never scores influence itself, and it never touches the
receipts/replay code that will consume it.

THE SCHEMA (a fixed contract with the receipts system -- do not change the shape):
    {"id": "sec_rag_context", "name": "rag_context", "source": "api" | "auto",
     "parts": [{"message_index": 1, "start": 0, "end": 438}], "char_count": 438, "preview": "..."}
`parts` are char spans: `message_index` indexes the run's `messages` list (int) or is `null` for a
raw-prompt run, whose offsets are into the run's `final_prompt` string instead. A part spanning a whole
message's content means "the entire message". Multiple parts under one section name means "this one
logical section rode in more than one message" (the same RAG block quoted twice, a card that repeats).
THE FINAL user message is never a section -- it's the question being answered, not context to ablate.

TWO WAYS A SECTION IS BORN, in priority order when more than one could apply to the same run:
  1. EXPLICIT (`source: "api"`) -- the caller tagged messages with `clozn_section` (chat shape,
     `sections_from_messages`) or gave exact char ranges (native shape, `sections_from_native`). Explicit
     always wins over auto for the SAME run: a caller who bothered to tag their own structure knows it
     better than any heuristic.
  2. AUTO (`source: "auto"`) -- no explicit tags anywhere -> the deterministic structural chunker
     (`auto_chunk_messages` for chat requests, `auto_chunk_prompt` for the raw-prompt/native shape) splits
     the non-final, non-user content on the boundaries real prompts actually have: markdown headers, doc
     dividers, RAG chunk markers, fenced code, XML-ish wrappers, paragraph breaks. MUST be a pure function
     of its input -- same messages in, byte-identical manifest out, every time, forever (no randomness, no
     model calls, no wall-clock/uuid anywhere in this file) -- that's what makes a manifest replayable and
     what makes THIS module's own tests meaningful.

     TWO-TIER CHUNKING POLICY (`_chunk_text`, the auto-chunker's core). A short system/RAG message with
     three "## " headers used to get chunked into ONE whole-message blob, because the old code only looked
     for structure at all once a message was already over LONG_MESSAGE_CHARS -- an author who bothered to
     mark three sections in 176 characters got exactly the same non-answer as an author who marked none.
     The fix distinguishes two tiers of "boundary", by how much the split reflects something the author
     actually did versus something this module is guessing at:
       * STRONG boundaries -- markdown headers, horizontal rules, "Document N"/"[N]" markers, and the
         edges of protected spans (fenced code / XML wrappers), i.e. everything `_boundary_offsets` finds.
         These are structure the author explicitly marked, so they split the message at ANY length (no
         LONG_MESSAGE_CHARS gate), and the chunks they produce are NEVER folded away by merge-small: a
         two-line "## Reference" section staying tiny is the intended outcome, not noise to clean up.
       * WEAK boundaries -- blank-line paragraph breaks (`_split_paragraphs`), the lowest-priority,
         purely-heuristic fallback. These only fire to further split one already-strong segment that is
         itself still a long unstructured blob (trimmed length > LONG_MESSAGE_CHARS), and only the
         resulting paragraph fragments are subject to merge-small (`_merge_small_spans`) -- folding a
         stray tiny paragraph into its neighbor is fine because nothing there was author-marked.
     Net behavior: a message with strong markers gets one chunk per strong segment regardless of length
     (long segments additionally paragraph-split internally); a message with none, under LONG_MESSAGE_CHARS,
     is an honest single "nothing to split on" chunk; a message with none, over LONG_MESSAGE_CHARS, is the
     old paragraph-split-and-merge behavior. Cap-16 (`_cap`) still applies globally, last, across whatever
     chunks either tier produced.

A THIRD source used to exist: `memory_card_sections` (`source: "memory_card"`), which located each applied
memory-card's text inside the assembled prompt by plain substring search. Memory cards were cut from the
product on 2026-07-27 along with the rest of memory, and this producer was removed with them -- nothing in
this codebase emits `source: "memory_card"` anymore. A run recorded before that cut may still carry a
`"memory_card"`-sourced entry in its stored `sections` field; downstream consumers (clozn.receipts.deltas/
forced, clozn.server.routes.section_influence/section_drill) treat that as a legacy value to skip or decline
honestly, never something to re-derive.

ID UNIQUENESS. Each of the functions below guarantees unique `id`s WITHIN its own returned list (the
"sec_" + slug(name), suffixed _2/_3/... on collision rule the schema specifies). `dedupe_ids` exists for a
caller that ever needs to combine more than one of these lists into a single run's `sections` field -- two
independently-deduped lists can still collide with each other (a `clozn_section` literally named "auto_1",
say) -- though today no caller combines lists (explicit and auto are mutually exclusive per run: explicit
wins outright, auto only runs when explicit found nothing at all). Kept as a pure, independently-tested
utility rather than deleted with its one-time caller.

Stdlib-only (this lives under the same "no torch, no model, no GPU" rule as replay.py/explain.py -- pure
string/offset logic over plain dicts, fully unit-testable against fixture messages). Every public function
here degrades rather than raises: bad input (wrong types, out-of-range offsets, non-dict messages) is
dropped or clamped, never an exception, so a malformed request can never take the sectioning code down
with it -- callers already wrap these calls defensively (see openai.py / app.py), but the functions are
built to be safe even if a future caller forgets to.
"""
from __future__ import annotations

import re

# ------- tunables (the deterministic auto-chunker's only "policy" knobs) -----------------------------

LONG_MESSAGE_CHARS = 600     # gates the WEAK tier only: a strong segment at or under this length is kept
                             # whole; only a longer one gets further paragraph-split (see _chunk_text).
                             # Strong boundaries (headers/hr/doc-markers/protected spans) split at ANY
                             # length -- this knob never suppresses them, only the paragraph fallback.
MIN_CHUNK_CHARS = 200        # any WEAK (paragraph) fragment under this is folded into a neighbor, but only
                             # within the strong segment it came from (`_merge_small_spans`); a small STRONG
                             # chunk (a short "## Header" section) is never folded -- its size was the point.
MAX_SECTIONS = 16            # hard cap on the auto-chunker's output (rule d)
PREVIEW_CHARS = 80           # "first ~80 chars" (schema's `preview` field)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# structural boundary patterns, checked in this priority order (rule b) -------------------------------
_HEADER_RE = re.compile(r"^#{1,6} ", re.MULTILINE)                       # "# Title", "## Sub-title"
_HR_RE = re.compile(r"^(?:---|\*\*\*)[ \t]*$", re.MULTILINE)             # a lone "---" or "***" line
_DOC_MARKER_RE = re.compile(r"^[ \t]*(?:Document[ \t]+\d+\b|\[\d+\])",   # "Document 3", "Document 3:", "[3]"
                            re.MULTILINE | re.IGNORECASE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)                         # ```...``` kept intact (rule b)
_XML_WRAP_RE = re.compile(r"<([a-zA-Z_][\w:-]*)>.*?</\1>", re.DOTALL)   # <context>...</context> etc.
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")                              # paragraph break (lowest priority)

_WS = " \t\r\n"


# ======================================================================================================
# small shared helpers
# ======================================================================================================

def _slugify(name: str) -> str:
    """lowercase, non-alnum runs -> a single '_', no leading/trailing '_'. Never empty (falls back to
    "section" so a pathological name -- all-punctuation, empty string -- still yields a usable id)."""
    s = _SLUG_RE.sub("_", str(name or "").lower()).strip("_")
    return s or "section"


def _make_id(name: str, used: set) -> str:
    """"sec_" + slug(name), suffixed _2/_3/... on collision -- the schema's id rule, scoped to ONE call's
    output (see the module docstring's ID UNIQUENESS note for why cross-list collisions need `dedupe_ids`
    instead)."""
    base = "sec_" + _slugify(name)
    candidate = base
    n = 2
    while candidate in used:
        candidate = f"{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def _text_of(value) -> str:
    return value if isinstance(value, str) else ""


def _final_user_index(messages: list) -> int | None:
    """Index of the LAST message with role "user" -- the query being answered, per _last_user's convention
    used elsewhere in this codebase (memory_assembly._last_user, memory_assembly._provenance_of): scan from
    the end, not just messages[-1], so a conversation that (unusually) ends in a non-user message still
    identifies the real final turn."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            return i
    return None


def _is_excluded_final(messages: list, idx: int) -> bool:
    """True for the one message that must NEVER become (or carry) a section: the final user turn. Checked
    two ways -- by role-scan (_final_user_index) AND by raw list position -- so a conversation whose last
    message is oddly not role "user" is still protected (belt-and-suspenders; see rule (e))."""
    if idx == len(messages) - 1:
        return True
    fu = _final_user_index(messages)
    return fu is not None and idx == fu


# ======================================================================================================
# 1. explicit API tags (chat shape): sections_from_messages
# ======================================================================================================

def sections_from_messages(messages: list) -> list | None:
    """Explicit `clozn_section` tags on chat messages -> the section manifest, or None if no (non-final)
    message carries the field at all -- callers use None as the "fall through to the auto-chunker" signal
    (see clozn.server.routes.openai's try_post). Multiple messages sharing one `clozn_section` name are
    grouped into ONE section with multiple parts, each part spanning that message's ENTIRE content (an
    explicit tag marks a whole message, not a sub-range within it -- for a sub-range use the native shape's
    `sections_from_native` instead). Sections are returned in the order their name FIRST appears.

    A `clozn_section` value that isn't a non-empty string is treated as "not tagged" (ignored, not an
    error) -- a stray `clozn_section: null` or `clozn_section: 3` from a buggy client degrades silently."""
    if not isinstance(messages, list) or not messages:
        return None
    groups: dict[str, list[tuple[dict, str]]] = {}
    order: list[str] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or _is_excluded_final(messages, idx):
            continue
        tag = msg.get("clozn_section")
        if not isinstance(tag, str) or not tag.strip():
            continue
        name = tag.strip()
        content = _text_of(msg.get("content"))
        part = {"message_index": idx, "start": 0, "end": len(content)}
        groups.setdefault(name, [])
        if name not in order:
            order.append(name)
        groups[name].append((part, content))
    if not order:
        return None
    used_ids: set = set()
    out = []
    for name in order:
        parts = [p for p, _ in groups[name]]
        text = "".join(t for _, t in groups[name])
        out.append({
            "id": _make_id(name, used_ids),
            "name": name,
            "source": "api",
            "parts": parts,
            "char_count": sum(p["end"] - p["start"] for p in parts),
            "preview": text[:PREVIEW_CHARS],
        })
    return out


# ======================================================================================================
# 2. the deterministic auto-chunker (rules a-f)
# ======================================================================================================

def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Fenced-code and XML-ish-wrapper spans, merged if they overlap -- these are never split internally
    (rule b: "kept intact as single chunks") and their boundaries win over anything found inside them
    (a '#' inside a code fence is not a markdown header)."""
    spans = sorted(
        [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]
        + [(m.start(), m.end()) for m in _XML_WRAP_RE.finditer(text)]
    )
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def _boundary_offsets(text: str, protected: list[tuple[int, int]]) -> set:
    offsets = {m.start() for regex in (_HEADER_RE, _HR_RE, _DOC_MARKER_RE)
               for m in regex.finditer(text) if not _inside(m.start(), protected)}
    for s, e in protected:
        offsets.add(s)
        offsets.add(e)
    return offsets


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _WS:
        start += 1
    while end > start and text[end - 1] in _WS:
        end -= 1
    return start, end


def _split_paragraphs(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Lowest-priority fallback split: blank-line paragraph breaks within one plain-text segment."""
    spans = []
    pos = start
    for m in _BLANK_LINE_RE.finditer(text, start, end):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < end:
        spans.append((pos, end))
    return spans or [(start, end)]


def _combine_spans(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0], b[1])


def _merge_small_spans(text: str, spans: list[tuple[int, int]],
                        min_chars: int = MIN_CHUNK_CHARS) -> list[tuple[int, int]]:
    """The WEAK-tier counterpart of `_merge_small`: folds a paragraph fragment under `min_chars` into its
    PRECEDING neighbor, same rule (c), but operating on plain (start, end) offsets within ONE strong
    segment rather than cross-message `units` dicts -- there's no separate `parts` list to concatenate
    here, the two fragments are already contiguous slices of the same segment, so "merge" is just "widen
    the range" (`_combine_spans`). A fragment with no preceding neighbor (it's first) instead folds
    FORWARD into the following one, mirroring `_merge_small` exactly, so a tiny leading paragraph never
    survives alone just because of where it happened to sit. `text` is accepted (unused) for symmetry with
    this module's other span-producing helpers, which all take the source text alongside offsets into it."""
    if not spans:
        return spans
    merged = [spans[0]]
    for s, e in spans[1:]:
        if (e - s) < min_chars:
            merged[-1] = _combine_spans(merged[-1], (s, e))
        else:
            merged.append((s, e))
    while len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_chars:
        head = merged.pop(0)
        merged[0] = _combine_spans(head, merged[0])
    return merged


def _chunk_text(text: str) -> list[tuple[int, int]]:
    """The two-tier auto-chunker's core (see the module docstring's TWO-TIER CHUNKING POLICY section for
    the full rationale -- this replaces the old `_structural_chunks`, whose single LONG_MESSAGE_CHARS gate
    let a short multi-header message fall through as one useless whole-message chunk).

    Splits `text` at every STRONG boundary (`_boundary_offsets`: markdown headers, horizontal rules,
    "Document N"/"[N]" markers, and protected-span edges) regardless of message length -- a protected span
    itself (fenced code / XML wrapper) rides out whole, at whatever size it is. Every OTHER resulting
    segment is trimmed and then, individually: kept whole if its trimmed length is at or under
    LONG_MESSAGE_CHARS (the common case for an author-marked section -- a "## Reference" block a few
    sentences long is exactly one chunk, small or not); otherwise it's still a long unstructured blob
    within its own strong boundaries, so it gets the WEAK-tier fallback -- paragraph-split
    (`_split_paragraphs`) and merge-small'd (`_merge_small_spans`) -- strictly WITHIN that one segment,
    never across a strong boundary, so a tiny strong chunk is never folded into its neighbor.

    A text with no strong boundaries at all reduces to the single segment (0, len(text)): at or under
    LONG_MESSAGE_CHARS that's one whole chunk (an honest "nothing to split on"), otherwise it's the plain
    paragraph-split-and-merge behavior. Always returns at least one (start, end) span."""
    if not text:
        return [(0, 0)]
    protected = _protected_spans(text)
    points = sorted(_boundary_offsets(text, protected) | {0, len(text)})
    spans: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        if b <= a:
            continue
        if (a, b) in protected:
            spans.append((a, b))
            continue
        ts, te = _trim(text, a, b)
        if te <= ts:
            continue
        if (te - ts) > LONG_MESSAGE_CHARS:
            frags = [_trim(text, s, e) for s, e in _split_paragraphs(text, ts, te)]
            frags = [(s, e) for s, e in frags if e > s]
            spans.extend(_merge_small_spans(text, frags))
        else:
            spans.append((ts, te))
    return spans or [(0, len(text))]


def structural_spans(text: str) -> list[tuple[int, int]]:
    """Return the uncapped text-only structural decomposition used by the chunker.

    This is intentionally a thin public seam over ``_chunk_text``.  It exposes
    the deterministic semantic boundaries without exposing the legacy section
    manifest's message filtering, global cap, IDs, or output shape.  In
    particular, callers that need a complete partition must assign the
    whitespace between these trimmed semantic spans themselves.
    """
    return _chunk_text(text) if isinstance(text, str) else []


def _unit_chars(unit: dict) -> int:
    return sum(e - s for _, s, e in unit["parts"])


def _combine_units(a: dict, b: dict) -> dict:
    return {"parts": a["parts"] + b["parts"]}


def _cap(units: list[dict], limit: int = MAX_SECTIONS) -> list[dict]:
    """Rule (d): while over `limit` sections, repeatedly merge the adjacent PAIR with the smallest combined
    size (leftmost wins on a tie) -- deterministic, since ties are resolved by position, not by whatever
    order a set/dict would iterate in."""
    units = list(units)
    while len(units) > limit:
        sizes = [_unit_chars(units[i]) + _unit_chars(units[i + 1]) for i in range(len(units) - 1)]
        i = sizes.index(min(sizes))
        units[i:i + 2] = [_combine_units(units[i], units[i + 1])]
    return units


def _sections_from_units(units: list[dict], texts: dict) -> list:
    """units -> the schema-shaped list, named auto_1, auto_2, ... in document order (rule f). `texts` maps
    a part's message_index (or None) to the string its (start, end) offsets are cut from."""
    used_ids: set = set()
    out = []
    for i, unit in enumerate(units, start=1):
        name = f"auto_{i}"
        parts = [{"message_index": mi, "start": s, "end": e} for mi, s, e in unit["parts"]]
        text = "".join(texts[mi][s:e] for mi, s, e in unit["parts"])
        out.append({
            "id": _make_id(name, used_ids),
            "name": name,
            "source": "auto",
            "parts": parts,
            "char_count": sum(e - s for _, s, e in unit["parts"]),
            "preview": text[:PREVIEW_CHARS],
        })
    return out


def auto_chunk_messages(messages: list) -> list:
    """The deterministic auto-chunker for chat-shaped requests (rules a-f; see the module docstring for
    the full policy). Candidates are every message that is NEITHER role "user" NOR the final turn (rule a
    reduces to plain role-filtering: a user message can never be anything but excluded, since the one
    "final" position the exclusion rule (e) cares about is itself always a user message in a well-formed
    conversation -- `_is_excluded_final` also guards the raw list-position case defensively). Each
    candidate's content is split by `_chunk_text` -- STRONG author-marked boundaries at any length, WEAK
    paragraph fallback + merge-small only within a still-long strong segment (the module docstring's
    TWO-TIER CHUNKING POLICY). Each resulting span becomes its own auto_N section (this is the whole
    point: one big RAG dump, or even a short multi-header system message, becomes several independently-
    ablatable pieces, not one) -- until cap-16 (d) folds some back together GLOBALLY, across every
    candidate message, as the one merge step left after chunking. Pure function of `messages`: no
    randomness, no clock, no model calls -- same input always produces byte-identical output."""
    if not isinstance(messages, list) or not messages:
        return []
    texts: dict = {}
    units: list[dict] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") == "user" or _is_excluded_final(messages, idx):
            continue
        content = _text_of(msg.get("content"))
        if not content:
            continue
        texts[idx] = content
        spans = _chunk_text(content)
        units.extend({"parts": [(idx, s, e)]} for s, e in spans if e > s)
    if not units:
        return []
    units = _cap(units, MAX_SECTIONS)
    return _sections_from_units(units, texts)


def auto_chunk_prompt(prompt: str) -> list:
    """The auto-chunker's raw-prompt counterpart, for the native (`/api/clozn/generate`) shape, which has
    no chat structure at all -- see this module's own docstring divergence note and
    clozn.cli.commands.run._log_run_cli (the ONE place that actually calls this today: native_completion
    itself is a transparent proxy with no server-side record() call, so there is no request body to read
    an explicit `sections` field from on that path; every native/CLI generation gets an auto manifest).

    Honest tradeoff, stated plainly: a raw prompt string has no reliable way to detect "the trailing user
    turn" the way a chat `messages` list does (that structure is exactly what got flattened away by
    whatever rendered this string), so unlike `auto_chunk_messages` this chunks the ENTIRE prompt, including
    whatever its final turn is. Same two-tier policy as `auto_chunk_messages` (`_chunk_text`); `message_index`
    is always None per the schema's raw-prompt convention -- parts are offsets into the run's `final_prompt`,
    not a `messages` list index."""
    if not isinstance(prompt, str) or not prompt:
        return []
    spans = _chunk_text(prompt)
    units = [{"parts": [(None, s, e)]} for s, e in spans if e > s]
    if not units:
        return []
    units = _cap(units, MAX_SECTIONS)
    return _sections_from_units(units, {None: prompt})


# ======================================================================================================
# 2b. drill-down: splitting ONE section's own text into finer sentence/line-level sub-spans
# ======================================================================================================
# A SIBLING to the section-granularity chunker above, one grain size down. `_chunk_text` answers "which
# section of the prompt mattered" by splitting on structure an AUTHOR marked (headers, hr, doc markers,
# fenced/XML spans). Once one of ITS sections is known to matter, the next question is "which PART of
# that section" -- and a single section's text (a RAG paragraph, a memory card, a few-shot block) rarely
# has any of that author-marked structure left inside it; what it does have is ordinary prose structure:
# sentences and lines. `drill_split` is the pure text->spans function for that; clozn.server.routes.
# section_drill is the one (and, today, only) caller, which remaps its (start, end) pairs -- relative to
# the ONE section's own concatenated text -- back into real prompt coordinates (see that module for the
# offset-remapping half of this feature; that half is NOT this module's job, per the "Touch ONLY"
# boundary this deliverable was scoped to).

DRILL_MIN_CHARS = 40   # a drill sub-span under this many chars folds into its preceding neighbor (see
                       # drill_split's docstring) -- the sentence-level analogue of MIN_CHUNK_CHARS, just
                       # an order of magnitude smaller: a drill target is already ONE section's text, not
                       # a whole message, so "a few words" is the right notion of "too small to be useful"
                       # here, not "a whole paragraph".

_DRILL_SENTENCE_END_RE = re.compile(r"[.!?]+(?=[ \t]|\n|$)")   # ". "/"! "/"? " or end-of-text/end-of-line
_DRILL_NEWLINE_RE = re.compile(r"\n+")                          # a hard newline is ALWAYS a boundary on
                                                                 # BOTH sides -- keeps a one-line list item
                                                                 # from bleeding into whatever follows it,
                                                                 # and keeps a run of blank lines from
                                                                 # surviving as its own (empty) span.


def drill_split(text: str) -> list[tuple[int, int]]:
    """ONE section's text -> finer (start, end) sub-spans -- turns "which section" into "which sentence".
    This is `_chunk_text`'s sibling, one grain size down: `_chunk_text` looks for structure an AUTHOR
    marked (headers, hr, doc markers, fenced/XML spans) at SECTION granularity; `drill_split` looks for
    the finest structure ordinary PROSE already has -- a sentence boundary (a run of `.`/`!`/`?`
    immediately followed by whitespace, a newline, or the end of the text) or a hard newline (a manually
    broken line, a list item). Both sides of a newline run are boundary points, so a blank line never
    survives as a lone empty span and a one-line item never bleeds into its neighbor.

    Every resulting fragment under `DRILL_MIN_CHARS` folds into its PRECEDING neighbor (the first fragment
    folds FORWARD instead -- same edge case `_merge_small_spans` already handles for the section-level
    chunker) -- reused here VERBATIM, just at a much smaller threshold, rather than a second copy of the
    same merge math. This is a real tradeoff, stated honestly rather than hidden: a short list item (say,
    "- milk" at 6 chars) DOES get swallowed into whatever line precedes it -- the alternative (guessing
    "this short line looks like a list item, don't merge it") is its own heuristic with its own test
    surface; folding purely by length is one rule, applied uniformly, that at least never leaves a
    3-character sliver no human would call an ablatable unit.

    A text with no sentence terminator and no newline at all reduces to the single (0, len(text)) span --
    "one short sentence, nothing finer to find" is an honest answer, not a bug (see this function's caller,
    section_drill.py, which reports that case explicitly rather than pretending a split happened). Pure/
    deterministic: stdlib `re` only, no randomness, no model calls, no wall-clock -- same text in, same
    spans out, forever, like every other span-producer in this module. Always returns at least one span."""
    if not text:
        return [(0, 0)]
    points = {0, len(text)}
    for m in _DRILL_SENTENCE_END_RE.finditer(text):
        points.add(m.end())
    for m in _DRILL_NEWLINE_RE.finditer(text):
        points.add(m.start())
        points.add(m.end())
    pts = sorted(points)
    spans: list[tuple[int, int]] = []
    for a, b in zip(pts, pts[1:]):
        if b <= a:
            continue
        ts, te = _trim(text, a, b)
        if te > ts:
            spans.append((ts, te))
    if not spans:
        return [(0, len(text))]
    return _merge_small_spans(text, spans, min_chars=DRILL_MIN_CHARS)


# ======================================================================================================
# 3. explicit char ranges (native shape): sections_from_native
# ======================================================================================================

def sections_from_native(prompt: str, sections_map: dict) -> list:
    """The native API shape `{"name": {"start": int, "end": int}}` -> the schema list, `message_index`
    always None (offsets are into `prompt`/the run's `final_prompt`). Defensive per this module's "never
    raise" rule: a non-int/missing start or end, or an out-of-range/inverted span (after clamping to
    [0, len(prompt)], start >= end), drops that ONE entry silently rather than failing the whole call --
    one bad range in a caller's map must not cost every other one its section."""
    if not isinstance(prompt, str) or not isinstance(sections_map, dict):
        return []
    n = len(prompt)
    used_ids: set = set()
    out = []
    for name, span in sections_map.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(span, dict):
            continue
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if end <= start:
            continue
        out.append({
            "id": _make_id(name, used_ids),
            "name": name,
            "source": "api",
            "parts": [{"message_index": None, "start": start, "end": end}],
            "char_count": end - start,
            "preview": prompt[start:end][:PREVIEW_CHARS],
        })
    return out


# ======================================================================================================
# 4. combining manifests from more than one source
# ======================================================================================================
# A fourth source used to live here -- memory_card_sections, "applied memory cards -> sections" -- removed
# 2026-07-29 along with the rest of memory (cut from the product 2026-07-27). See the module docstring's
# note on the THREE-turned-TWO section sources for what it did and why a legacy run may still carry its
# output shape (`source: "memory_card"`) even though nothing produces it anymore.

def dedupe_ids(sections: list) -> list:
    """Re-suffix any `id` collision in a manifest ASSEMBLED from more than one of the functions above (each
    only guarantees uniqueness within its own output -- see the module docstring's ID UNIQUENESS note).
    First occurrence of a given id keeps it; every later collision gets the schema's _2/_3/... suffix,
    walking past any suffix that's already in use (however it got there). Order-preserving; never mutates
    the input list or its dicts (each renamed section is a shallow copy)."""
    seen: dict[str, int] = {}
    out = []
    for sec in sections or []:
        base = str((sec or {}).get("id") or "sec_section")
        if base not in seen:
            seen[base] = 1
            out.append(sec)
            continue
        n = seen[base] + 1
        candidate = f"{base}_{n}"
        while candidate in seen:
            n += 1
            candidate = f"{base}_{n}"
        seen[base] = n
        seen[candidate] = 1
        renamed = dict(sec)
        renamed["id"] = candidate
        out.append(renamed)
    return out


# ======================================================================================================
# 5. resolving a section back to text (for tests, and for future receipts/inspector consumers)
# ======================================================================================================

def resolve(run: dict, name: str) -> str:
    """The current text a manifest entry (matched by `id` OR `name`) points at, reconstructed from `run`'s
    own `messages` (or `final_prompt`, for message_index: None parts). Multi-part sections are joined in
    part order. Never raises -- a missing section, an out-of-range message_index, a moved/edited message
    that's shrunk past a recorded offset: every failure mode returns "" rather than an exception, since this
    is a read-time convenience, not a write-time invariant that anything depends on holding."""
    try:
        sections = (run or {}).get("sections") or []
        target = next((s for s in sections if s.get("id") == name or s.get("name") == name), None)
        if target is None:
            return ""
        messages = run.get("messages") or []
        final_prompt = run.get("final_prompt")
        pieces = []
        for part in target.get("parts") or []:
            mi = part.get("message_index")
            if mi is None:
                src = _text_of(final_prompt)
            elif isinstance(mi, int) and 0 <= mi < len(messages):
                src = _text_of((messages[mi] or {}).get("content"))
            else:
                continue
            start, end = part.get("start"), part.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            start = max(0, min(start, len(src)))
            end = max(0, min(end, len(src)))
            if end > start:
                pieces.append(src[start:end])
        return "".join(pieces)
    except Exception:
        return ""
