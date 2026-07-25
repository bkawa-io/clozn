/* clozn studio -- the Compare canvas (Build 4): its OWN forking canvas, not a lens (notes/
   UX_INFORMATION_ARCHITECTURE.md SS0.1/SS2.1 -- referenced by path only, never quoted into this file).
   Route: #/compare (pick two runs) and #/compare/<idA>/<idB> (direct load), wired in app.mjs ->
   renderCompare(view, idA, idB, light). Style matches lens.mjs/observatory.mjs: one state object,
   small render-into-container functions, a delegated wire(view,state).

   LIVE-WIRED: POST /diff/runs (clozn.analysis.model_diff.diff_runs -- the only diff route that exists;
   clozn/receipts/quant_receipts.py's per-token-logprob/argmax-flip shape is a REAL but DEFERRED seam,
   never wired to any server route -- see the commit message) drives everything below: the shared
   prefix, the fork marker, both post-fork channels, and -- the load-bearing finding for this build --
   PER-TOKEN CONFIDENCE FOR BOTH RUNS, at every position (positions[].a_conf/b_conf), which is exactly
   what makes LATENT DIVERGENCE renderable: same committed token on both runs, confidence pulled apart,
   a thing no text diff can see. GET /runs/<id> is used only for the text-only degrade (a run missing a
   per-token trace) to still show the two reply texts plainly.

   Three states, computed per position from the route's own same/a_conf/b_conf fields (see classify()):
     identical -- same token, confidence within a small, DISCLOSED threshold (LATENT_THRESHOLD, shown in
       the honesty drawer -- a fixed, stated cutoff, never a claimed "measured" boundary).
     latent    -- same token, confidence differs by more than the threshold. The clozn-special.
     flipped   -- different committed token.
   Auto mode picks "inline diff" when the shared prefix is a small fraction of the longer reply (an
   early, near-total rewrite reads better linearly) and "forking view" otherwise; a manual toggle always
   overrides. Honesty drawer reuses the route's OWN caveat text verbatim (never restates or trims it).
*/

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function getJSON(url, opts) {
  try {
    const r = await fetch(url, opts);
    let body = null;
    try { body = await r.json(); } catch { /* empty/non-JSON body */ }
    return { ok: r.ok, status: r.status, body };
  } catch (e) {
    return { ok: false, status: 0, body: null, networkError: String(e) };
  }
}
const postJSON = (url, payload) => getJSON(url, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}),
});

const LATENT_THRESHOLD = 0.15;   // disclosed in the honesty drawer -- a fixed cutoff, not a measured one

function row(label, val) {
  return `<div class="drawer-row"><span class="label">${esc(label)}</span><span class="drawer-val">${esc(val)}</span></div>`;
}
function fmtConfVal(x) { return Number.isFinite(x) ? x.toFixed(3) : "—"; }
function identityLabel(id) {
  if (!id) return "—";
  return String(id.model || id.run_id || "?") + (id.quant ? ` · ${id.quant}` : "");
}

/* ======================================================================== entry point */

export async function renderCompare(view, idA, idB, light) {
  view.innerHTML = shell();
  const state = { light, idA: idA || null, idB: idB || null, runsList: null, diff: null, mode: "auto" };
  wire(view, state);
  await loadRunList(view, state);
  if (state.idA && state.idB) await loadDiff(view, state);
  else renderCanvasEmpty(view, state);
  return undefined;
}

function shell() {
  return `
    <div class="compare-page">
      <h1 class="view-title">Compare</h1>
      <div class="view-sub">its own forking canvas -- two runs, where they hold together and where they let go</div>

      <section class="cmp-section" aria-labelledby="cmp-pick-h">
        <h2 class="obs-h" id="cmp-pick-h">Pick two runs</h2>
        <p class="obs-sub quiet">same prompt, different model/quant/config is the controlled comparison
          this canvas is built for -- a mismatch is warned, not refused.</p>
        <div class="cmp-toolbar">
          <span class="chip chip-evidence"><i></i><span id="cmpASlotLabel">A: none</span></span>
          <span class="chip"><i style="background:var(--fork)"></i><span id="cmpBSlotLabel">B: none</span></span>
          <span class="spacer"></span>
          <button class="btn-ghost small primary" type="button" data-do-compare disabled>compare &rarr;</button>
        </div>
        <ul class="cmp-pick-list" id="cmpList"><li class="quiet breathing" style="padding:26px 0">loading runs&hellip;</li></ul>
      </section>

      <div id="cmpCanvas"></div>
    </div>`;
}

/* ======================================================================== run picker */

async function loadRunList(view, state) {
  const res = await getJSON("/runs");
  state.runsList = (res.ok && res.body && Array.isArray(res.body.runs)) ? res.body.runs : [];
  renderList(view, state);
}

function pickRowHTML(r, state) {
  const id = r.id || r.run_id || "";
  const label = String(r.prompt_summary || r.prompt || id || "(untitled run)");
  const when = r.created_at || "";
  const isA = id === state.idA, isB = id === state.idB;
  return `<li class="panel cmp-pick-row${isA ? " cmp-pick-a" : ""}${isB ? " cmp-pick-b" : ""}">
    <button type="button" class="btn-ghost small" data-set-a="${esc(id)}" aria-pressed="${isA}">${isA ? "A ✓" : "set A"}</button>
    <button type="button" class="btn-ghost small" data-set-b="${esc(id)}" aria-pressed="${isB}">${isB ? "B ✓" : "set B"}</button>
    <span class="cmp-pick-text" title="${esc(label)}">${esc(label)}</span>
    <span class="cmp-pick-meta">${esc(id)}${when ? " · " + esc(when) : ""}</span>
  </li>`;
}

function renderList(view, state) {
  const el = view.querySelector("#cmpList");
  if (el) {
    const items = state.runsList || [];
    el.innerHTML = items.length ? items.slice(0, 80).map(r => pickRowHTML(r, state)).join("")
      : `<li class="quiet" style="padding:26px 0">no runs reachable -- make one through the API and it lands here.</li>`;
  }
  const aLabel = view.querySelector("#cmpASlotLabel");
  const bLabel = view.querySelector("#cmpBSlotLabel");
  if (aLabel) aLabel.textContent = "A: " + (state.idA || "none");
  if (bLabel) bLabel.textContent = "B: " + (state.idB || "none");
  const btn = view.querySelector("[data-do-compare]");
  if (btn) btn.disabled = !(state.idA && state.idB);
}

/* ======================================================================== the diff + canvas */

async function loadDiff(view, state) {
  const el = view.querySelector("#cmpCanvas");
  if (el) el.innerHTML = `<section class="cmp-section"><p class="quiet breathing">diffing&hellip;</p></section>`;
  const res = await postJSON("/diff/runs", { a: state.idA, b: state.idB });
  state.diff = (res.body && typeof res.body === "object") ? res.body : { ok: false, error: "no response from the server" };
  state.light && state.light.pulse(.5);
  renderCanvas(view, state);
}

function renderCanvasEmpty(view, state) {
  void state;
  const el = view.querySelector("#cmpCanvas");
  if (el) el.innerHTML = `<section class="cmp-section"><p class="quiet">pick two runs above, then "compare &rarr;".</p></section>`;
}

function autoMode(d) {
  const total = Math.max(d.summary.a_reply_tokens || 0, d.summary.b_reply_tokens || 0, 1);
  const ratio = (d.common_prefix_len || 0) / total;
  return ratio < 0.2 ? "inline" : "fork";
}
function effectiveMode(state, d) { return state.mode === "auto" ? autoMode(d) : state.mode; }

function renderCanvas(view, state) {
  const el = view.querySelector("#cmpCanvas");
  if (!el) return;
  const d = state.diff;
  if (!d) { el.innerHTML = ""; return; }
  if (d.ok !== true) {
    el.innerHTML = `<section class="cmp-section"><p class="quiet">${esc(d.error || "diff unavailable.")}</p>
      ${Array.isArray(d.missing) && d.missing.length ? `<p class="quiet small-note">missing: ${esc(d.missing.join(", "))}</p>` : ""}</section>`;
    return;
  }
  if (d.trace_available === false) {
    el.innerHTML = textOnlySkeleton(d);
    loadTextOnlyBodies(view, state);
    return;
  }
  const mode = effectiveMode(state, d);
  el.innerHTML = `
    <section class="cmp-section" aria-labelledby="cmp-canvas-h">
      <h2 class="obs-h" id="cmp-canvas-h">${d.summary.identical ? "Identical" : "Forking view"}</h2>
      ${d.warn ? `<p class="quiet" style="color:var(--shaky);padding:0 0 10px">${esc(d.warn)}</p>` : ""}
      <div class="cmp-toolbar" role="group" aria-label="view mode">
        <button type="button" class="btn-ghost small cmp-mode-btn" data-cmp-mode="fork" aria-pressed="${mode === "fork"}">forking view</button>
        <button type="button" class="btn-ghost small cmp-mode-btn" data-cmp-mode="inline" aria-pressed="${mode === "inline"}">inline diff</button>
        <span class="quiet small-note" style="margin:0;padding:0">${state.mode === "auto"
          ? `auto-chosen (${mode}) -- ${d.common_prefix_len} shared token(s) before the split`
          : "manual"}</span>
      </div>
      <div class="cmp-legend">
        <span class="lg-identical"><i></i>identical</span>
        <span class="lg-latent"><i></i>latent -- same token, confidence pulled apart</span>
        <span class="lg-flip"><i></i>flipped</span>
      </div>
      ${d.summary.identical ? renderIdentical(d) : (mode === "inline" ? renderInline(d) : renderFork(d))}
      ${renderHonestyDrawer(d)}
    </section>`;
}

/* ------------------------------------------------------------------------ per-position classification */

function classify(p) {
  if (p.same === false) return "flip";
  const hasBoth = Number.isFinite(p.a_conf) && Number.isFinite(p.b_conf);
  if (hasBoth && Math.abs(p.a_conf - p.b_conf) > LATENT_THRESHOLD) return "latent";
  return "same";
}
function latentTitle(p, cls) {
  const parts = [`a_conf ${fmtConfVal(p.a_conf)}`, `b_conf ${fmtConfVal(p.b_conf)}`];
  if (cls === "latent") {
    parts.unshift(`latent divergence -- same token, confidence Δ=${Math.abs((p.a_conf || 0) - (p.b_conf || 0)).toFixed(2)}`);
  }
  return parts.join(" · ");
}
function tokenSpanHTML(piece, cls, title) {
  const cssClass = cls === "latent" ? "cmp-tok cmp-tok-latent" : cls === "flip" ? "cmp-tok cmp-tok-flip" : "cmp-tok";
  return `<span class="${cssClass}" title="${esc(title)}">${esc(piece)}</span>`;
}

/* ------------------------------------------------------------------------ identical (still may be latent) */

function renderIdentical(d) {
  const positions = d.positions || [];
  const hasLatent = positions.some(p => classify(p) === "latent");
  const ribbon = positions.map(p => {
    const cls = classify(p);
    return tokenSpanHTML(p.a_piece == null ? "" : p.a_piece, cls, latentTitle(p, cls));
  }).join("");
  return `
    <p class="quiet">${hasLatent
      ? "both runs produced the identical token stream -- but their recorded confidence still disagrees "
        + "at some positions, marked below. A text diff alone would never show this."
      : "both runs produced the identical token stream, with matching confidence throughout."}</p>
    <div class="pane cmp-prefix">${ribbon || `<span class="quiet">(empty reply)</span>`}</div>
    ${d.positions_truncated ? `<p class="quiet small-note">positions truncated at the server's cap.</p>` : ""}`;
}

/* ------------------------------------------------------------------------ forking view */

function renderFork(d) {
  const fd = d.first_divergence;
  const positions = d.positions || [];
  const prefix = positions.filter(p => p.i < fd.index);
  const prefixHTML = prefix.map(p => {
    const cls = classify(p);
    return tokenSpanHTML(p.a_piece == null ? "" : p.a_piece, cls, latentTitle(p, cls));
  }).join("");
  const marker = `
    <div class="cmp-fork-marker">
      <span>⑂ fork at token ${fd.index}</span>
      <span>A: "${esc(fd.a_piece == null ? "∅" : fd.a_piece)}"${Number.isFinite(fd.a_conf) ? ` (conf ${fd.a_conf.toFixed(2)})` : ""}</span>
      <span>B: "${esc(fd.b_piece == null ? "∅" : fd.b_piece)}"${Number.isFinite(fd.b_conf) ? ` (conf ${fd.b_conf.toFixed(2)})` : ""}</span>
    </div>`;
  const afterA = positions.filter(p => p.i >= fd.index && p.a_piece != null);
  const afterB = positions.filter(p => p.i >= fd.index && p.b_piece != null);
  const chanHTML = (list, side) => list.map(p => {
    const cls = classify(p);
    const piece = side === "a" ? p.a_piece : p.b_piece;
    return tokenSpanHTML(piece, cls, latentTitle(p, cls));
  }).join("");
  return `
    <div class="pane cmp-prefix">${prefixHTML || `<span class="quiet">(no shared prefix)</span>`}</div>
    ${marker}
    <div class="cmp-channels">
      <section class="pane cmp-channel" aria-label="run A, from the fork">
        <div class="pane-head"><span class="cmp-dot cmp-dot-a" aria-hidden="true"></span>
          <span class="label">A &middot; ${esc(identityLabel(d.a))}</span></div>
        <div class="pane-body">${chanHTML(afterA, "a") || `<span class="quiet">(nothing after the fork)</span>`}</div>
      </section>
      <section class="pane cmp-channel" aria-label="run B, from the fork">
        <div class="pane-head"><span class="cmp-dot cmp-dot-b" aria-hidden="true"></span>
          <span class="label">B &middot; ${esc(identityLabel(d.b))}</span></div>
        <div class="pane-body">${chanHTML(afterB, "b") || `<span class="quiet">(nothing after the fork)</span>`}</div>
      </section>
    </div>
    ${d.positions_truncated ? `<p class="quiet small-note">positions truncated at the server's cap -- only the first ${positions.length} shown.</p>` : ""}`;
}

/* ------------------------------------------------------------------------ inline diff (diverge-early) */

function renderInline(d) {
  const positions = d.positions || [];
  const rows = positions.map(p => {
    const cls = classify(p);
    return `<div class="cmp-inline-row">
      <span class="idx">${p.i}</span>
      ${tokenSpanHTML(p.a_piece == null ? "∅" : p.a_piece, cls, latentTitle(p, cls))}
      ${tokenSpanHTML(p.b_piece == null ? "∅" : p.b_piece, cls, latentTitle(p, cls))}
    </div>`;
  }).join("");
  return `
    <div class="pane">
      <div class="cmp-inline-head"><span></span><span>A</span><span>B</span></div>
      ${rows}
    </div>
    ${d.positions_truncated ? `<p class="quiet small-note">positions truncated at the server's cap -- only the first ${positions.length} shown.</p>` : ""}`;
}

/* ------------------------------------------------------------------------ text-only degrade (no trace) */

function textOnlySkeleton(d) {
  const s = d.summary || {};
  return `
    <section class="cmp-section">
      <h2 class="obs-h">Text-only comparison</h2>
      <p class="quiet">${esc(d.note || "no per-token trace on at least one run -- per-token states "
        + "(identical / latent / flipped) aren't computable here.")}</p>
      <p class="quiet small-note">${esc(s.char_similarity_label || "")}: ${esc(s.char_similarity)} ·
        a: ${esc(s.a_reply_chars)} chars · b: ${esc(s.b_reply_chars)} chars</p>
      <div class="cmp-channels" id="cmpTextOnlyChannels"><p class="quiet breathing">loading reply text&hellip;</p></div>
      ${renderHonestyDrawer(d)}
    </section>`;
}

async function loadTextOnlyBodies(view, state) {
  const [ra, rb] = await Promise.all([
    getJSON(`/runs/${encodeURIComponent(state.idA)}`), getJSON(`/runs/${encodeURIComponent(state.idB)}`),
  ]);
  const el = view.querySelector("#cmpTextOnlyChannels");
  if (!el) return;
  const textA = (ra.ok && ra.body && ra.body.response) || "";
  const textB = (rb.ok && rb.body && rb.body.response) || "";
  el.innerHTML = `
    <section class="pane cmp-channel" aria-label="run A reply text">
      <div class="pane-head"><span class="cmp-dot cmp-dot-a" aria-hidden="true"></span><span class="label">A</span></div>
      <div class="pane-body" style="white-space:pre-wrap">${textA ? esc(textA) : `<span class="quiet">(no text)</span>`}</div>
    </section>
    <section class="pane cmp-channel" aria-label="run B reply text">
      <div class="pane-head"><span class="cmp-dot cmp-dot-b" aria-hidden="true"></span><span class="label">B</span></div>
      <div class="pane-body" style="white-space:pre-wrap">${textB ? esc(textB) : `<span class="quiet">(no text)</span>`}</div>
    </section>`;
}

/* ------------------------------------------------------------------------ honesty drawer */

function almostLine(a) {
  if (!a) return "n/a";
  if (!a.checked) return a.note || "unknown";
  if (!a.found) return "not in a's recorded alternatives at the split";
  return `yes -- rank ${a.rank} in a's recorded alternatives${Number.isFinite(a.prob) ? ` (p=${a.prob.toFixed(3)})` : ""}`;
}

function renderHonestyDrawer(d) {
  const s = d.summary || {};
  return `
    <div class="honesty-drawer panel" role="region" aria-label="honesty drawer">
      ${row("method", "POST /diff/runs -- clozn.analysis.model_diff.diff_runs(); a pure, observational "
        + "record diff, no re-generation.")}
      ${row("a mean conf", fmtConfVal(s.a_mean_confidence))}
      ${row("b mean conf", fmtConfVal(s.b_mean_confidence))}
      ${row("surface similarity", `${s.char_similarity} -- ${s.char_similarity_label || ""}`)}
      ${row("b's token, was it a's almost?", almostLine(s.b_was_alternative_in_a))}
      ${row("latent threshold", `flagged when the SAME committed token's recorded confidence differs by `
        + `more than ${LATENT_THRESHOLD.toFixed(2)} between runs -- a fixed, disclosed cutoff, not a `
        + `measured boundary.`)}
      ${row("caveat", d.caveat || "")}
    </div>`;
}

/* ======================================================================== interactivity */

function wire(view, state) {
  view.addEventListener("click", e => {
    const setA = e.target.closest("[data-set-a]");
    if (setA) { state.idA = setA.dataset.setA; renderList(view, state); return; }
    const setB = e.target.closest("[data-set-b]");
    if (setB) { state.idB = setB.dataset.setB; renderList(view, state); return; }
    const doCompare = e.target.closest("[data-do-compare]");
    if (doCompare && !doCompare.disabled) {
      location.hash = `#/compare/${encodeURIComponent(state.idA)}/${encodeURIComponent(state.idB)}`;
      return;
    }
    const modeBtn = e.target.closest("[data-cmp-mode]");
    if (modeBtn) { state.mode = modeBtn.dataset.cmpMode; renderCanvas(view, state); return; }
  });
}
