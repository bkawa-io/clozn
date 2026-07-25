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

  let marks = [];
  if (state.influence.status === "done") {
    const map = state.influence.data;
    const answer = map && map.answer;
    // Marks are offsets into the EXACT text the map measured (answer.scored_text). Only trust them
    // against the rendered reply when the map itself says they match the recorded text character-for-
    // character -- an honest skip, not a mis-aligned highlight, when they don't.
    if (answer && answer.scored_text_matches_recorded && answer.recorded_text === text) {
      marks = buildAnswerMarks(map);
    }
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
    if (m.index > last) out += applyMarks(text.slice(last, m.index), start + last, marks, ansMarkHtml);
    out += `<code class="machine">${applyMarks(m[1], start + m.index + 1, marks, ansMarkHtml)}</code>`;
    last = re.lastIndex;
  }
  if (last < text.length) out += applyMarks(text.slice(last), start + last, marks, ansMarkHtml);
  return out.replace(/\n/g, "<br>");
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

/* ======================================================================== honesty drawer */

function renderDrawer(view, state) {
  const el = view.querySelector("#drawer");
  if (el) el.innerHTML = state.lens === "sources" ? sourcesDrawer(state) : skeletonDrawer(state.lens);
}

function row(label, val) {
  return `<div class="drawer-row"><span class="label">${esc(label)}</span><span class="drawer-val">${esc(val)}</span></div>`;
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
  renderDrawer(view, state);
  if (lensId === "sources") ensureSourcesComputed(view, state);
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
