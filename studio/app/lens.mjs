/* clozn studio -- the Lens page (Build 2): one run, context and reply side by side, a row of pill
   "lenses" that swap the OVERLAY on that same text rather than re-rendering a different copy of it.
   Route: #/runs/<id>, wired in app.mjs -> renderLens(view, id, light). Anatomy + honesty rules per the
   locked rebuild spec (local notes/UX_INFORMATION_ARCHITECTURE.md §2) -- referenced by path only, never
   quoted into this file.

   Sources is the hero lens and the only one wired end-to-end this build: it calls the real
   context<->answer influence-map route (span strengths) and the real provenance route (the
   context-vs-parametric verdict), tints context spans by measured strength, links a hovered answer
   span to the context span(s) that carried it, and groups coarse-to-fine REFINEMENT sub-spans under
   their parent source ("Doc 2 · part 1/2") instead of showing them as flat, confusing duplicates.
   Every other lens is an honest declared skeleton: it states what it will measure and says so plainly
   -- it never paints a color or a number it did not earn. */

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function getJSON(url, opts) {
  try {
    const r = await fetch(url, opts);
    let body = null;
    try { body = await r.json(); } catch { /* non-JSON or empty body */ }
    return { ok: r.ok, status: r.status, body };
  } catch (e) {
    return { ok: false, status: 0, body: null, networkError: String(e) };
  }
}
const postJSON = (url, payload) => getJSON(url, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}),
});
function fmtConfVal(x) { return Number.isFinite(x) ? x.toFixed(3) : "—"; }

const LENSES = [
  { id: "sources", label: "Sources" },
  { id: "shakiness", label: "Shakiness" },
  { id: "influences", label: "Influences" },
  { id: "concepts", label: "Concepts" },
  { id: "compare", label: "Compare" },
];

/* ======================================================================== entry point */

export async function renderLens(view, runId, light) {
  view.innerHTML = `<p class="quiet breathing">opening run…</p>`;
  const runRes = await getJSON(`/runs/${encodeURIComponent(runId)}`);
  if (!runRes.ok || !runRes.body || typeof runRes.body !== "object" || runRes.body.error) {
    view.innerHTML = `
      <h1 class="view-title">Run not found</h1>
      <p class="quiet">${esc((runRes.body && runRes.body.error)
        || "no run at this id -- it may have aged out of the local journal, or the id is wrong.")}</p>
      <p><a class="machine back-link" href="#/runs">&larr; back to runs</a></p>`;
    return;
  }
  const run = runRes.body;
  const rid = run.id || runId;
  const spansRes = await getJSON(`/runs/${encodeURIComponent(rid)}/spans`);
  const spanSummary = (spansRes.ok && spansRes.body) ? spansRes.body : null;

  const state = {
    runId: rid,
    run,
    light,
    lens: "sources",
    influence: { status: "idle", data: null, reason: "" },
    provenance: { status: "idle", data: null },
    spanSummary,
    copyTargets: [],
    // Build 3: the four declared-skeleton lenses, wired to real routes --
    // see the per-lens compute-on-demand functions near the bottom of this file.
    influences: { status: "idle", data: null, error: null },
    concepts: { status: "idle", data: null, reason: null },
    compareLens: { status: "idle", candidates: null, pickedId: null, diff: null },
  };

  view.innerHTML = shell(state);
  wire(view, state);
  renderPanes(view, state);
  renderDrawer(view, state);
  renderQuietStripInto(view, state);
  ensureSourcesComputed(view, state);
}

/* ======================================================================== shell + identity strip */

function shell(state) {
  return `
    <div class="lens-page">
      <a class="back-link machine" href="#/runs">&larr; runs</a>
      ${identityStrip(state.run)}
      <div class="lens-picker" role="tablist" aria-label="lens">
        ${LENSES.map(l => `<button class="lens-tab" role="tab" id="tab-${l.id}"
          aria-selected="${l.id === state.lens}" tabindex="${l.id === state.lens ? 0 : -1}"
          data-lens="${l.id}">${esc(l.label)}</button>`).join("")}
        <span class="lens-busy quiet breathing" id="lensBusy" hidden>computing sources · checking provenance…</span>
      </div>
      <div class="lens-panes" data-lens="${state.lens}">
        <section class="pane ctx-pane" aria-label="context">
          <div class="pane-head"><span class="label">context</span></div>
          <div class="pane-body" id="ctxBody"></div>
        </section>
        <section class="pane reply-pane" aria-label="reply">
          <div class="pane-head"><span class="label">reply</span>
            <button class="copy-btn" data-copy-idx="0" title="copy the reply text">copy</button></div>
          <div class="pane-body" id="replyBody"></div>
          <div class="quiet-strip" id="quietStrip"></div>
        </section>
      </div>
      <div class="honesty-drawer panel" id="drawer" role="region" aria-label="honesty drawer"></div>
    </div>`;
}

function identityStrip(run) {
  const meta = run.meta || {};
  const decode = meta.decode || {};
  const identity = run.identity || {};
  const model = meta.model_id || meta.family || run.model || "—";
  const quant = meta.quant || "—";
  const seedVal = decode.seed ?? meta.seed;
  const seed = (seedVal === undefined || seedVal === null) ? "—" : String(seedVal);
  const mode = decode.mode || meta.sampler_mode || "—";
  const temp = decode.temperature;
  const decodeText = mode + (typeof temp === "number" ? ` t=${temp}` : "");
  const when = run.created_at || "—";
  const reproducible = Boolean(identity.model_sha256 && identity.template_fingerprint);
  const rid = run.id || "";
  const permalinkUrl = `${location.origin}/r/${rid}`;
  return `
  <div class="identity-strip">
    <span class="id-item"><span class="id-k">run</span><span class="machine">${esc(rid)}</span></span>
    <span class="id-item"><span class="id-k">model</span><span class="machine">${esc(model)}</span></span>
    <span class="id-item"><span class="id-k">quant</span><span class="machine">${esc(quant)}</span></span>
    <span class="id-item"><span class="id-k">seed</span><span class="machine">${esc(seed)}</span></span>
    <span class="id-item"><span class="id-k">decode</span><span class="machine">${esc(decodeText)}</span></span>
    <span class="id-item"><span class="id-k">at</span><span class="machine">${esc(when)}</span></span>
    ${reproducible ? `<span class="id-item repro"><span class="machine">✓ reproducible</span></span>` : ""}
    <span class="spacer"></span>
    <a class="permalink machine" href="#/runs/${esc(rid)}/observatory"
      title="open this run in the Observatory -- the casting storm visualization">open the casting</a>
    <button class="permalink machine" data-copy-permalink="${esc(permalinkUrl)}"
      title="${esc(permalinkUrl)} — click to copy">/r/${esc(rid)}</button>
  </div>`;
}

/* ======================================================================== panes: context + reply */

function renderPanes(view, state) {
  const ctxBody = view.querySelector("#ctxBody");
  const replyBody = view.querySelector("#replyBody");
  if (ctxBody) ctxBody.innerHTML = renderContextPane(state);
  const { html, copyTargets } = renderReplyPane(state);
  state.copyTargets = copyTargets;
  if (replyBody) replyBody.innerHTML = html;
}

function renderContextPane(state) {
  const map = state.influence.status === "done" ? state.influence.data : null;
  if (map && Array.isArray(map.prompt_sources) && map.prompt_sources.length) {
    return renderContextFromSources(map);
  }
  return renderContextBase(state.run);
}

/* Pre-map rendering: best-effort, from the run's own recorded messages/prompt. Once the influence map
   lands this is REPLACED by renderContextFromSources, which reads the exact source text the map itself
   segmented (see that function) -- the two can differ slightly when a run's block was folded into
   assembled_messages, so the authoritative text always wins once it exists. */
function renderContextBase(run) {
  const msgs = (Array.isArray(run.messages) && run.messages.length) ? run.messages
    : (Array.isArray(run.assembled_messages) ? run.assembled_messages : []);
  const withText = msgs.filter(m => m && typeof m.content === "string" && m.content.trim());
  if (withText.length) {
    return withText.map(m => `
      <div class="ctx-source">
        <div class="ctx-source-role label">${esc(m.role || "?")}</div>
        <div class="ctx-source-text">${esc(m.content)}</div>
      </div>`).join("");
  }
  if (typeof run.final_prompt === "string" && run.final_prompt.trim()) {
    return `
      <div class="ctx-source">
        <div class="ctx-source-role label">prompt (raw)</div>
        <div class="ctx-source-text">${esc(run.final_prompt)}</div>
      </div>`;
  }
  return `<p class="quiet">no recorded context for this run.</p>`;
}

/* Authoritative rendering, once the influence map exists: map.prompt_sources carries the EXACT text
   the map segmented (whichever list -- raw messages or assembled_messages -- actually reached the
   model), so span offsets always line up. THE FIX: coarse spans that were refined into finer sub-spans
   are never shown alongside their own parent (that reads as duplicates); only the finer children are
   shown, grouped and labeled "<source> · part i/n" so it reads as a within-doc zoom. */
function renderContextFromSources(map) {
  const groups = buildSpanGroups(map);
  const labels = docLabels(map.prompt_sources);
  const strengths = spanStrength(map);
  let maxStrength = 0;
  for (const v of strengths.values()) if (v > maxStrength) maxStrength = v;

  return map.prompt_sources.map(src => {
    const spansForSource = groups.get(src.id) || null;
    let body;
    if (spansForSource && spansForSource.length) {
      const marks = spansForSource.map(s => {
        const strength = strengths.get(s.id) || 0;
        return {
          start: s.start, end: s.end, id: s.id,
          intensity: maxStrength > 0 ? Math.min(1, strength / maxStrength) : 0,
          strength, effect: dominantEffect(map, s.id),
          label: spanLabel(s, spansForSource, labels),
        };
      });
      body = applyMarks(src.text, 0, marks, srcMarkHtml);
    } else {
      body = esc(src.text);
    }
    const roleLabel = (src.role === "system" || src.role === "developer")
      ? "system" : (src.role || "?");
    const omitted = src.selected === false;
    return `
      <div class="ctx-source${omitted ? " ctx-omitted" : ""}">
        <div class="ctx-source-role label">${esc(roleLabel)}${omitted
          ? ` <span class="ctx-omitted-tag">not measured — over the span budget</span>` : ""}</div>
        <div class="ctx-source-text">${body}</div>
      </div>`;
  }).join("");
}

function renderReplyPane(state) {
  const run = state.run;
  const text = String(run.response || "");
  if (!text.trim()) {
    return { html: `<p class="quiet">no recorded reply text for this run.</p>`, copyTargets: [] };
  }

  // Marks are computed PER LENS, exclusively -- only the active lens's marks are built each render, and
  // activateLens() re-renders the panes on every tab switch (see below) so the DOM always carries exactly
  // one lens's marks at a time. This sidesteps merging independently-grouped, generally-OVERLAPPING span
  // partitions (source-carried runs vs confidence bands vs cross-run divergence runs) into one non-
  // overlapping mark list, which applyMarks() cannot do -- each lens gets its own clean, non-overlapping
  // partition of the SAME text instead.
  let marks = [];
  if (state.lens === "sources" && state.influence.status === "done") {
    const map = state.influence.data;
    const answer = map && map.answer;
    // Marks are offsets into the EXACT text the map measured (answer.scored_text). Only trust them
    // against the rendered reply when the map itself says they match the recorded text character-for-
    // character -- an honest skip, not a mis-aligned highlight, when they don't.
    if (answer && answer.scored_text_matches_recorded && answer.recorded_text === text) {
      marks = buildAnswerMarks(map);
    }
  } else if (state.lens === "shakiness") {
    marks = buildShakinessMarks(state, text);
  } else if (state.lens === "compare") {
    marks = buildCompareMarks(state, text);
  }

  const copyTargets = [text];
  const blocks = tokenizeBlocks(text);
  const html = blocks.map(b => {
    if (b.kind === "code") {
      copyTargets.push(b.text);
      const idx = copyTargets.length - 1;
      return `
        <div class="code-block-wrap">
          <button class="copy-btn code-copy" data-copy-idx="${idx}" title="copy code">copy</button>
          <pre class="code-block machine"><code>${esc(b.text)}</code></pre>
        </div>`;
    }
    return splitParagraphs(b.text, b.start).map(p => `<p>${renderInline(p, marks)}</p>`).join("");
  }).join("");

  return { html, copyTargets };
}

/* ======================================================================== span grouping + labeling */

function sourceIdOf(spanId) { return spanId.split(".").slice(0, 2).join("."); }

/* Displayed spans = coarse spans that were NOT refined, plus every fine child of the ones that were.
   A refined coarse parent's own row still exists in the measured matrix (it counts toward the honesty
   drawer's numbers) but is never itself painted -- only its finer children are, which is exactly the
   sub-span-grouping fix: no source ever shows both a coarse block AND its own finer pieces at once. */
function buildSpanGroups(map) {
  const refined = new Set(
    (map.selection && map.selection.refinement && map.selection.refinement.refined_context_span_ids) || []
  );
  const spans = map.prompt_spans || [];
  const displayed = spans.filter(s => s.level === "fine" || (s.level === "coarse" && !refined.has(s.id)));
  const bySource = new Map();
  for (const s of displayed) {
    const sid = sourceIdOf(s.id);
    if (!bySource.has(sid)) bySource.set(sid, []);
    bySource.get(sid).push(s);
  }
  return bySource;
}

function docLabels(sources) {
  const labels = new Map();
  let n = 0;
  for (const src of sources || []) {
    if (src.role === "system" || src.role === "developer") labels.set(src.id, "System");
    else { n += 1; labels.set(src.id, `Doc ${n}`); }
  }
  return labels;
}

function spanLabel(span, group, labels) {
  const base = labels.get(sourceIdOf(span.id)) || sourceIdOf(span.id);
  if (span.level === "fine") {
    const sibs = group.filter(s => s.level === "fine" && s.parent_id === span.parent_id);
    const idx = sibs.findIndex(s => s.id === span.id) + 1;
    return `${base} · part ${idx}/${sibs.length}`;
  }
  const coarseSibs = group.filter(s => s.level === "coarse");
  if (coarseSibs.length > 1) {
    const idx = coarseSibs.findIndex(s => s.id === span.id) + 1;
    return `${base} · segment ${idx}/${coarseSibs.length}`;
  }
  return base;
}

function spanStrength(map) {
  const m = new Map();
  for (const link of map.links || []) {
    const cur = m.get(link.context_span_id) || 0;
    if (link.abs_delta_nats > cur) m.set(link.context_span_id, link.abs_delta_nats);
  }
  return m;
}

function dominantEffect(map, spanId) {
  let best = null;
  for (const link of map.links || []) {
    if (link.context_span_id !== spanId) continue;
    if (!best || link.abs_delta_nats > best.abs_delta_nats) best = link;
  }
  return best ? best.effect : "neutral";
}

/* Wrap ranges of `text` (marks[].start/end in the SAME coordinate space as [baseStart, baseStart+len))
   with toHtml(slice, mark). Marks are assumed non-overlapping -- true both for one source's own spans
   (the segmenter partitions disjoint char ranges) and for the reply's grouped, contiguous answer runs. */
function applyMarks(text, baseStart, marks, toHtml) {
  if (!text) return "";
  if (!marks || !marks.length) return esc(text);
  const end = baseStart + text.length;
  const rel = marks
    .filter(mk => mk.start < end && mk.end > baseStart)
    .map(mk => Object.assign({}, mk, {
      s: Math.max(mk.start, baseStart) - baseStart,
      e: Math.min(mk.end, end) - baseStart,
    }))
    .sort((a, b) => a.s - b.s);
  let out = "", cursor = 0;
  for (const mk of rel) {
    if (mk.s > cursor) out += esc(text.slice(cursor, mk.s));
    if (mk.e > mk.s) out += toHtml(text.slice(mk.s, mk.e), mk);
    cursor = Math.max(cursor, mk.e);
  }
  if (cursor < text.length) out += esc(text.slice(cursor));
  return out;
}

function srcMarkHtml(slice, m) {
  const suppress = m.effect === "suppresses" ? " src-suppress" : "";
  return `<mark class="src-mark${suppress}" data-span-id="${esc(m.id)}" style="--mark-a:${m.intensity.toFixed(3)}"
    title="${esc(m.label)} — ${m.strength.toFixed(3)} nats · ${esc(m.effect)}">${esc(slice)}</mark>`;
}

/* ======================================================================== reply: markdown-ish + marks */

function tokenizeBlocks(text) {
  const blocks = [];
  const re = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) blocks.push({ kind: "p", start: last, end: m.index, text: text.slice(last, m.index) });
    const codeStart = m.index + 3 + m[1].length + 1;
    blocks.push({ kind: "code", start: codeStart, end: codeStart + m[2].length, text: m[2], lang: m[1] });
    last = re.lastIndex;
  }
  if (last < text.length) blocks.push({ kind: "p", start: last, end: text.length, text: text.slice(last) });
  return blocks;
}

function splitParagraphs(text, baseStart) {
  const out = [];
  const re = /\n{2,}/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    out.push({ text: text.slice(last, m.index), start: baseStart + last });
    last = re.lastIndex;
  }
  out.push({ text: text.slice(last), start: baseStart + last });
  return out.filter(p => p.text.trim().length > 0);
}

function renderInline(para, marks) {
  const { text, start } = para;
  const re = /`([^`\n]+)`/g;
  let last = 0, m, out = "";
  while ((m = re.exec(text))) {
    if (m.index > last) out += applyMarks(text.slice(last, m.index), start + last, marks, replyMarkHtml);
    out += `<code class="machine">${applyMarks(m[1], start + m.index + 1, marks, replyMarkHtml)}</code>`;
    last = re.lastIndex;
  }
  if (last < text.length) out += applyMarks(text.slice(last), start + last, marks, replyMarkHtml);
  return out.replace(/\n/g, "<br>");
}

/* One dispatcher for every reply-pane mark kind, since renderInline/applyMarks call a single toHtml --
   each lens tags its own marks with `.kind` (sources' buildAnswerMarks marks carry none, so they fall to
   the default/ans-mark branch, UNCHANGED from before this dispatcher existed). */
function replyMarkHtml(slice, m) {
  if (m.kind === "shaky") return shkMarkHtml(slice, m);
  if (m.kind === "cmp") return cmpMarkHtml(slice, m);
  return ansMarkHtml(slice, m);
}

function ansMarkHtml(slice, m) {
  return `<mark class="ans-mark" data-ctx="${esc(m.ctxIds.join(","))}">${esc(slice)}</mark>`;
}

/* Group consecutive answer TOKENS that share the same carrying context span(s) into one contiguous,
   hoverable run -- never a per-token gradient. Only tokens the map itself marks clear_source get a
   mark at all; the rest stay plain text (no claim where none was earned). */
function buildAnswerMarks(map) {
  const spans = (map.answer_spans || []).slice().sort((a, b) => a.start - b.start);
  const info = new Map((map.summary && map.summary.answer_to_context || []).map(a => [a.answer_span_id, a]));
  const groups = [];
  let cur = null;
  for (const sp of spans) {
    const a = info.get(sp.id);
    const clear = !!(a && a.clear_source && (a.top_context_span_ids || []).length);
    const key = clear ? a.top_context_span_ids.join(",") : null;
    if (clear && cur && cur.key === key && cur.end === sp.start) {
      cur.end = sp.end;
    } else {
      if (cur) groups.push(cur);
      cur = clear ? { start: sp.start, end: sp.end, key, ctxIds: a.top_context_span_ids } : null;
    }
  }
  if (cur) groups.push(cur);
  return groups;
}

/* ======================================================================== shakiness: reply-pane marks */

/* True only when this run's OWN recorded trace tokens reconstruct its OWN recorded reply text, character
   for character -- the same alignment discipline buildAnswerMarks/cast-feed.mjs already apply before
   trusting any token-index -> char-offset mapping. False means an honest skip, never a guessed offset. */
function traceTokensAlignToText(run) {
  const tokens = run && run.trace && Array.isArray(run.trace.tokens) ? run.trace.tokens : null;
  const text = String((run && run.response) || "");
  return !!(tokens && tokens.length && tokens.join("") === text);
}

/* GET /runs/<id>/spans (clozn.runs.confidence_spans) is fetched ONCE at page open regardless of lens
   (state.spanSummary) -- shakiness never issues its own fetch. It already reshapes the run's own
   trace.confidence into maximal same-band (strong/okay/shaky) runs that never cross a sentence boundary
   -- a SPAN grouping, not per-token shading (the page's "color spans/sets, not single tokens" rule).
   spans[].start/end are INCLUSIVE TOKEN indices; converted to char offsets here via the same cumulative-
   join trick cast-feed.mjs uses, only once alignment is verified. */
function buildShakinessMarks(state, text) {
  const spans = state.spanSummary && Array.isArray(state.spanSummary.spans) ? state.spanSummary.spans : null;
  if (!spans || !spans.length || !traceTokensAlignToText(state.run)) return [];
  const tokens = state.run.trace.tokens;
  const offsets = [0];
  for (const t of tokens) offsets.push(offsets[offsets.length - 1] + t.length);
  return spans
    .map(s => ({
      start: offsets[s.start], end: offsets[s.end + 1], kind: "shaky",
      band: s.band, meanConf: s.mean_conf, minConf: s.min_conf, nTokens: s.n_tokens,
    }))
    .filter(m => m.end > m.start && text.slice(m.start, m.end).length);
}

/* Every span becomes a <mark> (so every span -- including "strong" -- is hoverable for its exact numbers),
   but only "okay"/"shaky" get a background tint; "strong" renders as plain text (the base `mark{background:
   none}` rule) since it's the unflagged default -- the legend states all three thresholds regardless. */
function shkMarkHtml(slice, m) {
  const cls = m.band === "shaky" ? " shk-shaky" : m.band === "okay" ? " shk-okay" : " shk-strong";
  const title = `${m.band} \xb7 mean conf ${m.meanConf.toFixed(3)} \xb7 min ${m.minConf.toFixed(3)} \xb7 `
    + `${m.nTokens} token${m.nTokens === 1 ? "" : "s"}`;
  return `<mark class="shk-mark${cls}" title="${esc(title)}">${esc(slice)}</mark>`;
}

/* ======================================================================== compare: reply-pane marks */

// Mirrors compare.mjs's OWN LATENT_THRESHOLD constant + classify() exactly (a fixed, disclosed cutoff --
// not re-derived here). compare.mjs's internals aren't exported (only renderCompare is), and this file may
// only edit lens.mjs -- so this is a faithful, documented copy, not an independent judgment call; keep it
// in lockstep with compare.mjs if that constant/formula ever changes.
const CMP_LATENT_THRESHOLD = 0.15;
function classifyCmp(p) {
  if (p.same === false) return "flip";
  const hasBoth = Number.isFinite(p.a_conf) && Number.isFinite(p.b_conf);
  if (hasBoth && Math.abs(p.a_conf - p.b_conf) > CMP_LATENT_THRESHOLD) return "latent";
  return "same";
}

/* Builds marks for THIS run's OWN reply text only (a = this run, always -- see loadCompareDiff), grouping
   consecutive same-classification positions into one contiguous span (never per-token shading). Identical
   positions stay unmarked plain text; only latent/flipped runs get painted -- this lens overlays the
   picked run's divergence ONTO this run's own reply, unlike the Compare canvas's two-channel fork view. */
function buildCompareMarks(state, text) {
  const d = state.compareLens.diff;
  if (!d || d.ok !== true || d.trace_available === false) return [];
  const positions = (d.positions || []).filter(p => p.a_piece != null).slice().sort((a, b) => a.i - b.i);
  if (!positions.length || positions.map(p => p.a_piece).join("") !== text) return [];  // can't trust offsets
  const marks = [];
  let cur = null, pos = 0;
  for (const p of positions) {
    const start = pos, end = pos + p.a_piece.length;
    pos = end;
    const cls = classifyCmp(p);
    if (cls === "same") { cur = null; continue; }
    const title = cls === "flip"
      ? `flipped -- this run said "${p.a_piece}", the compared run said "${p.b_piece == null ? "∅" : p.b_piece}"`
      : `latent divergence -- same token, confidence a=${fmtConfVal(p.a_conf)} vs b=${fmtConfVal(p.b_conf)}`;
    if (cur && cur.state === cls && cur.end === start) { cur.end = end; cur.title = title; }
    else { cur = { start, end, kind: "cmp", state: cls, title }; marks.push(cur); }
  }
  return marks;
}

function cmpMarkHtml(slice, m) {
  const cls = m.state === "flip" ? " cmpl-flip" : " cmpl-latent";
  return `<mark class="cmpl-mark${cls}" title="${esc(m.title)}">${esc(slice)}</mark>`;
}

/* ======================================================================== honesty drawer */

const DRAWERS = {
  sources: sourcesDrawer,
  shakiness: state => shakinessDrawer(state),
  influences: state => influencesDrawer(state),
  concepts: state => conceptsDrawer(state),
  compare: state => compareLensDrawer(state),
};
function renderDrawer(view, state) {
  const el = view.querySelector("#drawer");
  if (!el) return;
  const fn = DRAWERS[state.lens];
  el.innerHTML = fn ? fn(state) : skeletonDrawer(state.lens);
}

function row(label, val) {
  return `<div class="drawer-row"><span class="label">${esc(label)}</span><span class="drawer-val">${esc(val)}</span></div>`;
}
// Same shape as row(), but `html` is trusted markup (a button/select the caller built with esc() already
// applied to any user/data-derived text inside it) rather than plain text to escape.
function rawRow(label, html) {
  return `<div class="drawer-row"><span class="label">${esc(label)}</span><span class="drawer-val">${html}</span></div>`;
}

const VERDICT_LABEL = {
  CONTEXT_CARRIED: "Answered from your context",
  MIXED: "From context + the model",
  PARAMETRIC: "From the model's own knowledge",
  INCONCLUSIVE: "Couldn't determine the source",
};
function plainVerdict(v) { return VERDICT_LABEL[String(v)] || "Couldn't determine the source"; }

function sourcesDrawer(state) {
  const { influence, provenance } = state;
  const rows = [];

  if (provenance.status === "idle" || provenance.status === "busy") {
    rows.push(row("verdict", provenance.status === "busy"
      ? "checking — attention knockout, a few seconds…" : "not checked yet"));
  } else {
    const p = provenance.data;
    if (p && p.ok) {
      rows.push(row("verdict", plainVerdict(p.verdict)));
      rows.push(row("control ratio", typeof p.best_control_ratio === "number"
        ? p.best_control_ratio.toFixed(1) + "× vs matched-random cut" : "n/a"));
    } else {
      rows.push(row("verdict", (p && (p.blocked || p.error)) || "provenance unavailable"));
    }
  }

  rows.push(row("method", "each context span is swapped for a matched-length neutral filler and the "
    + "recorded answer is re-scored teacher-forced (no generation); the verdict above is a separate "
    + "attention-knockout pass over the model's own recorded answer token."));

  if (influence.status === "busy") {
    rows.push(row("span map", "computing…"));
  } else if (influence.status === "error") {
    rows.push(row("span map", influence.reason || "influence map unavailable"));
  } else if (influence.status === "done") {
    const thr = (influence.data && influence.data.thresholds) || {};
    const n = (influence.data.prompt_spans || []).length;
    rows.push(row("span map", `${n} measured span${n === 1 ? "" : "s"} · clears floor at `
      + `${typeof thr.cell_abs_delta_nats === "number" ? thr.cell_abs_delta_nats + " nats" : "n/a"}`));
  }

  rows.push(row("caveat", "attention-knockout + matched-filler replacement, validated across two model "
    + "families — read every verdict here as evidence, not proof."));
  return rows.join("");
}

const SKELETONS = {
  shakiness: {
    what: "per-span token-probability banding (strong / okay / shaky), read straight off the trace's "
      + "own recorded logprobs — zero re-generation.",
    caveat: "a probability band, not a hallucination detector: a confident wrong answer reads identically "
      + "to a confident right one.",
  },
  influences: {
    what: "leave-one-out on each active steering dial, compared against a no-dial control.",
    caveat: "steering only — memory-as-context is gone from this surface; never a percentage of “why.”",
  },
  concepts: {
    what: "a fitted linear (J-lens) “disposed to say” readout at each layer.",
    caveat: "“disposed to say,” not a verified thought — a linear lens always emits something, even from noise.",
  },
  compare: {
    what: "another run's per-span divergences overlaid on this one.",
    caveat: "observational — a downstream difference, never independent evidence for either run's correctness.",
  },
};
function skeletonDrawer(lens) {
  const s = SKELETONS[lens] || { what: "not yet specified.", caveat: "" };
  return `
    <div class="drawer-row"><span class="label">measures</span><span class="drawer-val">${esc(s.what)}</span></div>
    <div class="drawer-row"><span class="label">verdict</span><span class="drawer-val">not computed in this build</span></div>
    ${s.caveat ? row("caveat", s.caveat) : ""}
    <p class="quiet drawer-arriving">arriving in a later build — no placeholder numbers, only the honest plan.</p>`;
}

/* ======================================================================== shakiness drawer */

function bandLegendHTML() {
  return `<div class="lens-legend">
    <span><i class="lg-strong"></i>strong (&ge; 0.80)</span>
    <span><i class="lg-okay"></i>okay (0.50 &ndash; 0.80)</span>
    <span><i class="lg-shaky"></i>shaky (&lt; 0.50)</span>
  </div>`;
}

function shakinessDrawer(state) {
  const rows = [bandLegendHTML()];
  const spans = state.spanSummary && Array.isArray(state.spanSummary.spans) ? state.spanSummary.spans : null;
  if (!spans || !spans.length) {
    rows.push(row("spans", "no per-token confidence trace recorded for this run -- nothing to band."));
    rows.push(row("caveat", SKELETONS.shakiness.caveat));
    return rows.join("");
  }
  if (!traceTokensAlignToText(state.run)) {
    rows.push(row("spans", `${spans.length} span(s) computed, but this run's own trace tokens don't `
      + `reconstruct its own reply text character-for-character -- can't safely paint them onto the text; `
      + `shown as counts only.`));
  }
  const counts = { strong: 0, okay: 0, shaky: 0 };
  spans.forEach(s => { counts[s.band] = (counts[s.band] || 0) + 1; });
  rows.push(row("verdict", state.spanSummary.summary || "—"));
  rows.push(row("spans", `${counts.strong || 0} strong \xb7 ${counts.okay || 0} okay \xb7 ${counts.shaky || 0} `
    + `shaky (${spans.length} total) -- maximal same-band token runs, split at sentence boundaries`));
  rows.push(row("method", "each token banded by its OWN recorded confidence (trace.confidence, i.e. "
    + "exp(recorded logprob)); a span is a maximal run of same-band tokens that never crosses a sentence "
    + "boundary. Zero re-generation -- GET /runs/<id>/spans, already fetched when this run opened."));
  rows.push(row("caveat", SKELETONS.shakiness.caveat));
  return rows.join("");
}

/* ======================================================================== influences drawer + measure */

async function runInfluencesMeasure(view, state) {
  state.influences.status = "busy";
  renderDrawer(view, state);
  const res = await postJSON(`/runs/${encodeURIComponent(state.runId)}/receipts`, { mode: "both" });
  if (res.ok && res.body && typeof res.body === "object" && !res.body.error) {
    state.influences.status = "done";
    state.influences.data = res.body;
  } else {
    state.influences.status = "error";
    state.influences.error = extractReason(res.body) || `the receipts route did not answer (status ${res.status})`;
  }
  state.light && state.light.pulse(.55);
  renderDrawer(view, state);
}

function influencesDrawer(state) {
  const rows = [];
  const dials = (state.run.behavior && state.run.behavior.active_dials) || {};
  const dialNames = Object.keys(dials);
  rows.push(row("recorded on this run", dialNames.length
    ? dialNames.map(n => `${n}=${dials[n]}`).join(", ")
    : "no active steering dials recorded on this run"));

  const inf = state.influences;
  if (inf.status === "idle") {
    rows.push(rawRow("measure", `<button class="btn-ghost small" type="button" data-measure-influences>`
      + `leave one out &rarr;</button><br><span class="quiet small-note">cost: regenerates one greedy `
      + `baseline, then one no-dial control per active dial (decode-time only -- the prompt KV stays `
      + `reusable, cheap) plus one teacher-forced dependence pass per dial (no generation) -- a few `
      + `seconds, on demand. POST /runs/&lt;id&gt;/receipts, mode "both".</span>`));
    rows.push(row("caveat", SKELETONS.influences.caveat));
    return rows.join("");
  }
  if (inf.status === "busy") { rows.push(row("measuring", "regenerating the no-dial control(s)…")); return rows.join(""); }
  if (inf.status === "error") {
    rows.push(row("error", inf.error));
    rows.push(row("caveat", SKELETONS.influences.caveat));
    return rows.join("");
  }

  const d = inf.data;
  const dialReceipts = (d.receipts || []).filter(r => r.influence && r.influence.dial);
  const dialForced = new Map((d.forced_receipts || []).filter(r => r.influence && r.influence.dial)
    .map(r => [r.influence.dial, r]));
  const dialSkipped = (d.skipped || []).filter(s => s.influence && s.influence.dial);

  if (!dialReceipts.length && !dialSkipped.length) {
    rows.push(row("result", "no active steering dials were fired on this run -- nothing to leave one out."));
    rows.push(row("caveat", SKELETONS.influences.caveat));
    return rows.join("");
  }

  for (const r of dialReceipts) {
    const name = r.influence.dial;
    const forced = dialForced.get(name);
    const parts = [r.has_effect ? "regen: changed the greedy reply" : "regen: no change to the greedy reply"];
    if (r.delta) {
      parts.push(`words ${r.delta.words[0]}→${r.delta.words[1]} \xb7 word-set changed ${r.delta.changed}%`);
    }
    if (r.ablated_reply_truncated) parts.push("(ablated reply early-stopped at first divergence)");
    if (forced && forced.causal_verified) {
      parts.push(`forced: ${forced.has_effect ? "measurable dependence" : "below the dependence floor"} `
        + `(${forced.mean_nats_per_token.toFixed(3)} mean nats/token)`);
      const floor = forced.null_floor;
      if (floor && Number.isFinite(floor.ratio_real_over_floor)) {
        parts.push(floor.exceeds_floor_by_order_of_magnitude
          ? `clears a random-vector null floor by ${floor.ratio_real_over_floor.toFixed(1)}\xd7`
          : `NOT clearly above a random-vector-sized null floor (ratio ${floor.ratio_real_over_floor.toFixed(2)})`);
      }
      if (!r.has_effect && forced.has_effect) {
        parts.push("silent influence: the greedy TEXT didn't change, but the model's confidence in it did");
      }
    } else if (forced) {
      parts.push(`forced: ${forced.note || "not measurable"}`);
    }
    rows.push(row(`dial \xb7 ${name}=${r.influence.value}`, parts.join(" \xb7 ")));
  }
  for (const s of dialSkipped) {
    rows.push(row(`dial \xb7 ${s.influence.dial}`, `skipped -- ${s.reason}`));
  }
  const dialRedundant = (d.redundant_pairs || []).filter(p => (p.redundant || []).some(k => k.startsWith("dial:")));
  for (const p of dialRedundant) rows.push(row("redundant pair", `${p.redundant.join(" + ")} -- ${p.note}`));
  rows.push(row("method", d.perf_note || "leave-one-out over every fired dial, plus a pairwise redundancy guard."));
  rows.push(row("caveat", SKELETONS.influences.caveat));
  return rows.join("");
}

/* ======================================================================== concepts drawer + J-lens fetch */

async function fetchConceptsLayer(view, state, layer) {
  state.concepts.status = "busy";
  renderDrawer(view, state);
  const body = { topk: 5 };
  if (layer !== undefined && layer !== null) body.layer = layer;
  const res = await postJSON(`/runs/${encodeURIComponent(state.runId)}/jlens`, body);
  if (res.ok && res.body && res.body.available) {
    state.concepts.status = "done";
    state.concepts.data = res.body;
  } else {
    state.concepts.status = "error";
    state.concepts.reason = (res.body && res.body.reason) || `J-lens unavailable (status ${res.status})`;
  }
  state.light && state.light.pulse(.5);
  renderDrawer(view, state);
}

function ensureConceptsComputed(view, state) {
  if (state.concepts.status !== "idle") return;
  fetchConceptsLayer(view, state, undefined);
}

function conceptsDrawer(state) {
  const c = state.concepts;
  if (c.status === "idle" || c.status === "busy") {
    return row("reading", c.status === "busy" ? "reading the J-lens…" : "not read yet");
  }
  if (c.status === "error") {
    return row("unavailable", c.reason) + row("caveat", SKELETONS.concepts.caveat);
  }
  const d = c.data;
  const layerOptions = (d.available_layers || []).map(l =>
    `<option value="${l}"${l === d.layer ? " selected" : ""}>layer ${l}</option>`).join("");
  const rows = [];
  rows.push(rawRow("layer", `<select class="text-input" id="cptLayerPick">${layerOptions}</select> `
    + `<span class="quiet small-note">of [${esc((d.available_layers || []).join(", "))}] fitted layers</span>`));
  rows.push(row("reading", `this run's own recorded ${d.text_source === "response" ? "reply" : esc(String(d.text_source))} `
    + `text (${d.n_tokens} token(s))`));
  const tableRows = (d.tokens || []).map((tok, i) => {
    const top = (d.readouts[i] || []).slice(0, 3)
      .map(r => `${esc(r.piece)} <span class="quiet" style="padding:0">(${Number(r.score).toFixed(2)})</span>`)
      .join(" \xb7 ");
    return `<tr><td>${esc(tok)}</td><td>${top || "—"}</td></tr>`;
  }).join("");
  rows.push(`<table class="readout-table">
    <thead><tr><th>token</th><th>disposed to say next (top-3, raw lens logit)</th></tr></thead>
    <tbody>${tableRows}</tbody></table>`);
  const prov = d.provenance || {};
  if (prov.fit_model) rows.push(row("fitted on", prov.fit_model));
  rows.push(row("caveat", prov.note || SKELETONS.concepts.caveat));
  return rows.join("");
}

/* ======================================================================== compare-lens drawer + diff */

async function ensureCompareCandidatesLoaded(view, state) {
  if (state.compareLens.candidates) return;
  state.compareLens.status = "busy";
  renderDrawer(view, state);
  const res = await getJSON("/runs");
  const all = (res.ok && res.body && Array.isArray(res.body.runs)) ? res.body.runs : [];
  const run = state.run, rid = state.runId;
  const promptSummary = run.prompt_summary || null;
  const parentId = run.parent_run_id || null;
  const candidates = [];
  for (const r of all) {
    const id = r.id || r.run_id;
    if (!id || id === rid) continue;
    let tag = null;
    if (r.parent_run_id === rid) tag = "forked child";
    else if (parentId && id === parentId) tag = "parent";
    else if (promptSummary && r.prompt_summary === promptSummary) tag = "same prompt";
    if (tag) candidates.push({ id, label: r.prompt_summary || id, when: r.created_at || "", tag });
  }
  state.compareLens.candidates = candidates;
  state.compareLens.status = "idle";
  renderDrawer(view, state);
}

async function loadCompareDiff(view, state, pickedId) {
  state.compareLens.pickedId = pickedId || null;
  state.compareLens.diff = null;
  if (!pickedId) { renderDrawer(view, state); renderPanes(view, state); return; }
  state.compareLens.status = "busy";
  renderDrawer(view, state);
  const res = await postJSON("/diff/runs", { a: state.runId, b: pickedId });
  state.compareLens.diff = (res.body && typeof res.body === "object")
    ? res.body : { ok: false, error: "no response from the server" };
  state.compareLens.status = "done";
  state.light && state.light.pulse(.5);
  renderDrawer(view, state);
  renderPanes(view, state);   // the diff just resolved -- (re)paint the reply pane's divergence marks
}

function compareLensDrawer(state) {
  const cl = state.compareLens;
  const rows = [];

  if (cl.status === "busy" && !cl.candidates) {
    rows.push(row("candidates", "loading runs that share this run's prompt or lineage…"));
    return rows.join("");
  }
  const cands = cl.candidates || [];
  if (!cands.length) {
    rows.push(row("compare to", "no other run shares this run's prompt, and this run has no forked "
      + "children or parent -- nothing to compare against here."));
    rows.push(row("caveat", SKELETONS.compare.caveat));
    return rows.join("");
  }
  const options = cands.map(c => `<option value="${esc(c.id)}"${c.id === cl.pickedId ? " selected" : ""}>`
    + `${esc(String(c.label).slice(0, 56))} — ${esc(c.tag)}${c.when ? " \xb7 " + esc(c.when) : ""}</option>`).join("");
  rows.push(rawRow("compare to", `<select class="text-input" id="cmpLensPick" style="max-width:100%">`
    + `<option value="">— pick a run —</option>${options}</select>`));

  if (cl.status === "busy" && cl.pickedId) { rows.push(row("diff", "diffing…")); return rows.join(""); }

  const d = cl.diff;
  if (!d) { rows.push(row("caveat", SKELETONS.compare.caveat)); return rows.join(""); }
  if (d.ok !== true) {
    rows.push(row("diff", d.error || "diff unavailable."));
    return rows.join("");
  }
  if (d.warn) rows.push(row("warn", d.warn));
  if (d.trace_available === false) {
    const s = d.summary || {};
    rows.push(row("diff", d.note || "no per-token trace on the compared run -- text-only comparison."));
    rows.push(row("surface similarity", `${s.char_similarity} — ${s.char_similarity_label || ""}`));
  } else {
    const positions = (d.positions || []).filter(p => p.a_piece != null);
    const counts = { same: 0, latent: 0, flip: 0 };
    positions.forEach(p => { counts[classifyCmp(p)]++; });
    rows.push(row("this run's spans", `${counts.same} identical \xb7 ${counts.latent} latent \xb7 `
      + `${counts.flip} flipped (of ${positions.length}) -- painted on the reply at left; identical stays plain`));
    rows.push(row("latent threshold", `flagged when the SAME committed token's confidence differs by more `
      + `than ${CMP_LATENT_THRESHOLD.toFixed(2)} between runs -- a fixed, disclosed cutoff (mirrors the `
      + `Compare canvas's own), not a measured boundary.`));
    if (d.positions_truncated) rows.push(row("note", "positions truncated at the server's cap."));
  }
  rows.push(row("method", "POST /diff/runs -- clozn.analysis.model_diff.diff_runs(); a pure, observational "
    + "record diff, no re-generation. The same route the Compare canvas uses."));
  rows.push(row("caveat", d.caveat || SKELETONS.compare.caveat));
  if (cl.pickedId) {
    rows.push(`<p class="quiet small-note"><a class="machine" href="#/compare/${esc(state.runId)}/${esc(cl.pickedId)}">`
      + `open the full Compare canvas for these two runs &rarr;</a></p>`);
  }
  return rows.join("");
}

/* ======================================================================== quiet strip */

function quietStripHTML(state) {
  const chips = [];
  if (state.spanSummary && state.spanSummary.summary) {
    const shaky = /shaky/i.test(state.spanSummary.summary);
    chips.push(`<span class="chip ${shaky ? "chip-shaky" : "chip-confident"}"><i></i>${esc(state.spanSummary.summary)}</span>`);
  }
  if (state.provenance.status === "done" && state.provenance.data && state.provenance.data.ok) {
    chips.push(`<span class="chip chip-evidence"><i></i>${esc(plainVerdict(state.provenance.data.verdict))}</span>`);
  }
  // Abstain note: no stored run record carries a persisted ask/abstain calibration band today (it is
  // response/stream-only metadata at generation time, never written into the run journal) -- so this
  // chip is honestly always absent rather than reading a field that doesn't exist.
  return chips.slice(0, 3).join("");
}
function renderQuietStripInto(view, state) {
  const el = view.querySelector("#quietStrip");
  if (el) el.innerHTML = quietStripHTML(state);
}

/* ======================================================================== compute-on-demand */

function updateBusyIndicator(view, state) {
  const el = view.querySelector("#lensBusy");
  if (el) el.hidden = !(state.influence.status === "busy" || state.provenance.status === "busy");
}

function extractReason(body) {
  if (!body) return null;
  if (typeof body.error === "string") return body.error;
  if (body.error && typeof body.error.message === "string") return body.error.message;
  return null;
}

async function computeInfluence(view, state) {
  let res = await getJSON(`/runs/${encodeURIComponent(state.runId)}/influence-map`);
  if (res.status === 404) {
    res = await getJSON(`/runs/${encodeURIComponent(state.runId)}/influence-map`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
  }
  if (res.ok && res.body && res.body.available) {
    state.influence.status = "done";
    state.influence.data = res.body;
  } else {
    state.influence.status = "error";
    state.influence.reason = extractReason(res.body) || "the context-answer influence map is unavailable for this run.";
  }
  state.light && state.light.pulse(.55);
  renderPanes(view, state);
  renderDrawer(view, state);
  renderQuietStripInto(view, state);
  updateBusyIndicator(view, state);
}

async function computeProvenance(view, state) {
  const res = await getJSON(`/runs/${encodeURIComponent(state.runId)}/provenance`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
  });
  state.provenance.status = "done";
  state.provenance.data = (res.body && typeof res.body === "object") ? res.body
    : { ok: false, blocked: "the server did not answer" };
  state.light && state.light.pulse(.55);
  renderDrawer(view, state);
  renderQuietStripInto(view, state);
  updateBusyIndicator(view, state);
}

function ensureSourcesComputed(view, state) {
  const tasks = [];
  if (state.influence.status === "idle") { state.influence.status = "busy"; tasks.push(computeInfluence(view, state)); }
  if (state.provenance.status === "idle") { state.provenance.status = "busy"; tasks.push(computeProvenance(view, state)); }
  if (tasks.length) { renderDrawer(view, state); updateBusyIndicator(view, state); }
}

/* ======================================================================== interactivity */

function activateLens(view, state, lensId) {
  if (!LENSES.some(l => l.id === lensId)) return;
  state.lens = lensId;
  view.querySelectorAll(".lens-tab").forEach(btn => {
    const active = btn.dataset.lens === lensId;
    btn.setAttribute("aria-selected", String(active));
    btn.tabIndex = active ? 0 : -1;
  });
  const panes = view.querySelector(".lens-panes");
  if (panes) panes.dataset.lens = lensId;
  state.light && state.light.pulse(.45);
  // Reply-pane marks are lens-exclusive (see renderReplyPane) -- re-render the panes on every switch so
  // the DOM always carries the NEWLY active lens's own marks, not whichever lens rendered last.
  renderPanes(view, state);
  renderDrawer(view, state);
  if (lensId === "sources") ensureSourcesComputed(view, state);
  else if (lensId === "concepts") ensureConceptsComputed(view, state);
  else if (lensId === "compare") ensureCompareCandidatesLoaded(view, state);
}

async function doCopy(btn, text) {
  try {
    await navigator.clipboard.writeText(text || "");
    const orig = btn.textContent;
    btn.textContent = "copied";
    setTimeout(() => { btn.textContent = orig; }, 1200);
  } catch { /* clipboard denied -- the button just doesn't confirm; nothing to fake here */ }
}

function linkMarks(view, mark, on) {
  if (mark.classList.contains("ans-mark")) {
    const ids = (mark.dataset.ctx || "").split(",").filter(Boolean);
    ids.forEach(id => view.querySelectorAll(`mark.src-mark[data-span-id="${id}"]`)
      .forEach(el => el.classList.toggle("xlink", on)));
    mark.classList.toggle("xlink-self", on);
  } else if (mark.classList.contains("src-mark")) {
    const id = mark.dataset.spanId;
    view.querySelectorAll("mark.ans-mark[data-ctx]").forEach(el => {
      const ids = (el.dataset.ctx || "").split(",");
      if (ids.includes(id)) el.classList.toggle("xlink", on);
    });
    mark.classList.toggle("xlink-self", on);
  }
}

function wire(view, state) {
  view.addEventListener("click", e => {
    const tab = e.target.closest(".lens-tab");
    if (tab) { activateLens(view, state, tab.dataset.lens); return; }
    const copyBtn = e.target.closest("[data-copy-idx]");
    if (copyBtn) { doCopy(copyBtn, state.copyTargets[Number(copyBtn.dataset.copyIdx)]); return; }
    const permaBtn = e.target.closest("[data-copy-permalink]");
    if (permaBtn) { doCopy(permaBtn, permaBtn.dataset.copyPermalink); return; }
    const measureBtn = e.target.closest("[data-measure-influences]");
    if (measureBtn) { runInfluencesMeasure(view, state); return; }
  });
  view.addEventListener("change", e => {
    if (e.target.id === "cptLayerPick") { fetchConceptsLayer(view, state, Number(e.target.value)); return; }
    if (e.target.id === "cmpLensPick") { loadCompareDiff(view, state, e.target.value || null); return; }
  });
  view.addEventListener("keydown", e => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const tabs = Array.from(view.querySelectorAll(".lens-tab"));
    const i = tabs.indexOf(document.activeElement);
    if (i === -1) return;
    e.preventDefault();
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    next.focus();
    activateLens(view, state, next.dataset.lens);
  });
  view.addEventListener("mouseover", e => {
    const mark = e.target.closest("mark");
    if (mark) linkMarks(view, mark, true);
  });
  view.addEventListener("mouseout", e => {
    const mark = e.target.closest("mark");
    if (mark) linkMarks(view, mark, false);
  });
}
