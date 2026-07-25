/* clozn studio -- the Model page (Build 3): currently-serving status, the model-gated capability list
   (which lenses this exact GGUF can run, and how to change that), local-inventory, and a dev-facing
   Health sub-panel. Route: #/model, wired in app.mjs -> renderModel(view, light). Style matches
   lens.mjs/behavior.mjs: one state object, small render-into-container functions, one delegated
   wire(view,state), and the "declared skeleton" pattern (state the plan/reason, never a control that
   silently does nothing) for anything with no backing server route.

   LIVE-WIRED: /engine/health (serving block + capabilities.* + a live /steer/axes reuse for the
   calibration count in Health), POST /jlens (a real availability probe -- {available:false, reason} is
   the honest signal this route is built to give; used both to gate the Concepts-lens row and to read the
   fitted J-lens's own provenance when it IS available).
   DECLARED SKELETON (no backing route found -- see the commit message): local GGUF inventory + pull.
   `clozn/cli/commands/models.py` (_scan_models/cmd_models/cmd_pull) is CLI-only -- no server route lists
   or fetches models on disk, so this section names the CLI rather than drawing a fake "load"/"pull"
   button. Likewise "last CI" / "drift": no server route surfaces `clozn test-model` / `clozn ci` /
   `clozn quant-check` / `clozn diff-model` results -- named, not faked. */

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

/* Mirrors clozn/server/substrates.py's own _quant_from_name -- /engine/health carries no separate
   `quant` field, but GGUF files name their own quant in the basename, so this reads it off the same
   `model` path the server already reports (real data, not a guess at a number). */
function quantFromName(name) {
  const m = String(name || "").match(/(IQ\d+[A-Z0-9_]*|Q\d+(?:_[A-Z0-9]+)+|Q\d+|BF16|F16|F32)/i);
  return m ? m[1].toUpperCase() : null;
}
function basename(p) { return String(p || "").split(/[\\/]/).pop() || ""; }

/* ======================================================================== entry point */

export async function renderModel(view, light) {
  view.innerHTML = shell();
  const state = { light, health: null, axes: [], jlens: { status: "idle", data: null } };
  wire(view, state);
  await loadHealth(view, state);
  await Promise.all([loadCalibrationCount(view, state), probeJlens(view, state)]);
  renderCapabilities(view, state);
  renderHealthPanel(view, state);
}

function shell() {
  return `
    <div class="model-page">
      <h1 class="view-title">Model</h1>
      <div class="view-sub">your model · your machine · any GGUF · nothing leaves the box</div>

      <section class="mdl-section" aria-labelledby="mdl-serving-h">
        <h2 class="bhv-h" id="mdl-serving-h">Currently serving</h2>
        <div id="mdlServing"><p class="quiet breathing">reaching the engine…</p></div>
      </section>

      <section class="mdl-section" aria-labelledby="mdl-cap-h">
        <h2 class="bhv-h" id="mdl-cap-h">Capabilities on this model</h2>
        <p class="bhv-sub quiet">which lenses and features this exact GGUF can run -- a lens that can't
          run here is quietly absent with a reason, never a dead button.</p>
        <div id="mdlCapabilities"><p class="quiet breathing">checking…</p></div>
      </section>

      <section class="mdl-section" aria-labelledby="mdl-local-h">
        <h2 class="bhv-h" id="mdl-local-h">Local models</h2>
        <div id="mdlLocal">${localModelsSkeleton()}</div>
      </section>

      <section class="mdl-section" aria-labelledby="mdl-health-h">
        <h2 class="bhv-h" id="mdl-health-h">Health</h2>
        <p class="bhv-sub quiet">dev-facing: whatever of identity/calibration/CI/drift is actually
          readable from here; the rest names the CLI that owns it.</p>
        <div id="mdlHealth"><p class="quiet breathing">…</p></div>
      </section>
    </div>`;
}

/* ======================================================================== currently serving */

async function loadHealth(view, state) {
  const res = await getJSON("/engine/health");
  const el = view.querySelector("#mdlServing");
  if (!el) return;
  if (!res.ok || !res.body || !res.body.engine) {
    el.innerHTML = `<p class="quiet">no engine reachable -- start one with <span class="machine">clozn serve</span>.</p>`;
    return;
  }
  const h = res.body.engine;
  state.health = h;
  const name = basename(h.model) || h.model || "?";
  const quant = quantFromName(h.model);
  const shaText = h.model_sha256 ? String(h.model_sha256).slice(0, 12) : null;
  el.innerHTML = `
    <div class="kv pane serving-kv">
      <span>model <b class="machine" title="${esc(h.model || "")}">${esc(name)}</b></span>
      ${quant ? `<span>quant <b class="machine" title="read from the served filename">${esc(quant)}</b></span>` : ""}
      <span>ctx <b class="machine">${esc(h.n_ctx ?? "?")}</b></span>
      <span>layers <b class="machine">${esc(h.n_layer ?? "?")}</b></span>
      <span>embd <b class="machine">${esc(h.n_embd ?? "?")}</b></span>
      <span>device <b class="machine">${esc(h.device ?? "?")}</b>${h.gpu_layers ? ` <span class="quiet">(${esc(h.gpu_layers)} layers offloaded)</span>` : ""}</span>
      <span>mode <b class="machine">${esc(h.mode ?? "?")}</b></span>
      <span>status <b class="machine">up</b></span>
      ${shaText ? `<span>sha256 <b class="machine" title="${esc(h.model_sha256)}">${esc(shaText)}…</b></span>` : ""}
    </div>`;
}

/* ======================================================================== capabilities */

const LENS_DEFS = [
  { id: "sources", label: "Sources", cap: null,
    note: cap => cap.attn_knockout
      ? "attention-knockout is on -- the full source verdict runs."
      : "attention-knockout is off on this build (needs --no-flash-attn) -- span highlighting still runs, the verdict line reads \"provenance unavailable.\"" },
  { id: "shakiness", label: "Shakiness", cap: null, note: () => "reads the trace's own recorded token probabilities -- no model-specific requirement." },
  { id: "influences", label: "Influences", cap: "steering", note: cap => cap.steering ? "tone + concept dials are live on this engine." : "this engine build reports no steering capability." },
  { id: "concepts", label: "Concepts", cap: "jlens", jlensGated: true },
  { id: "compare", label: "Compare", cap: null, note: () => "structural -- needs two runs, not a model feature." },
];

function renderCapabilities(view, state) {
  const el = view.querySelector("#mdlCapabilities");
  if (!el) return;
  if (!state.health) {
    el.innerHTML = `<p class="quiet">no engine reachable -- capabilities arrive with it.</p>`;
    return;
  }
  const cap = state.health.capabilities || {};
  const rows = LENS_DEFS.map(def => {
    let available, reason;
    if (def.jlensGated) {
      available = state.jlens.status === "available";
      if (available) {
        const prov = state.jlens.data && state.jlens.data.provenance;
        reason = (prov && prov.fit_model)
          ? `fitted on ${esc(prov.fit_model)}, layers ${esc((state.jlens.data.available_layers || []).join(", ") || "?")}.`
          : "a J-lens is loaded on this engine.";
      } else if (state.jlens.status === "checking") {
        reason = "checking…";
      } else {
        const why = esc((state.jlens.data && state.jlens.data.reason) || "no fitted J-lens for this model.");
        reason = `${why} check qualification: <span class="machine">clozn qualify-whitebox `
          + `${esc(basename(state.health.model))}</span> -- fitting a new lens is an offline lab pipeline, `
          + `not a single CLI step in this build.`;
      }
    } else {
      available = def.cap ? !!cap[def.cap] : true;
      reason = def.note(cap);
    }
    return `
      <div class="cap-row${available ? "" : " cap-absent"}">
        <span class="cap-dot${available ? " cap-dot-on" : ""}" aria-hidden="true"></span>
        <span class="cap-label machine">${esc(def.label)}</span>
        <span class="cap-state">${available ? "available" : "not available"}</span>
        <span class="cap-reason quiet">${reason}</span>
      </div>`;
  }).join("");
  const flags = ["sae", "readout", "score_arms", "infill", "revise"]
    .filter(k => k in cap)
    .map(k => `<span class="dial-tag${cap[k] ? " dial-tag-calib" : ""}">${esc(k)}: ${cap[k] ? "on" : "off"}</span>`)
    .join(" ");
  el.innerHTML = `<div class="cap-list">${rows}</div>` + (flags ? `<div class="cap-flags">${flags}</div>` : "");
}

async function probeJlens(view, state) {
  state.jlens.status = "checking";
  const res = await getJSON("/jlens", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "clozn" }),
  });
  if (res.ok && res.body && res.body.available) {
    state.jlens.status = "available";
    state.jlens.data = res.body;
  } else {
    state.jlens.status = "unavailable";
    state.jlens.data = res.body || null;
  }
}

/* ======================================================================== local models (skeleton) */

function localModelsSkeleton() {
  return `
    <div class="skeleton-block">
      <div class="drawer-row"><span class="label">what's here</span><span class="drawer-val">no server route
        lists the GGUFs on this machine, or loads/pulls a different one -- this is CLI-only today
        (<span class="machine">clozn/cli/commands/models.py</span>).</span></div>
      <div class="drawer-row"><span class="label">list</span><span class="drawer-val machine">clozn models</span></div>
      <div class="drawer-row"><span class="label">fetch</span><span class="drawer-val machine">clozn pull &lt;name&gt;</span></div>
      <div class="drawer-row"><span class="label">fit check</span><span class="drawer-val machine">clozn plan &lt;name&gt;</span></div>
      <p class="quiet drawer-arriving">a real inventory route (and the header model switcher it would
        feed) is the honest next step here -- not invented in this build.</p>
    </div>`;
}

/* ======================================================================== health panel */

async function loadCalibrationCount(view, state) {
  // /steer/axes is POST-only (substrate-generic dispatch; see behavior.mjs's loadDials for the same note).
  const res = await postJSON("/steer/axes", {});
  if (res.ok && res.body && Array.isArray(res.body.axes)) state.axes = res.body.axes;
}

function renderHealthPanel(view, state) {
  const el = view.querySelector("#mdlHealth");
  if (!el) return;
  const nCalib = state.axes.filter(a => a.calibrated).length;
  const nTotal = state.axes.length;
  const sha = state.health && state.health.model_sha256;
  el.innerHTML = `
    <div class="drawer-row"><span class="label">identity</span><span class="drawer-val">${
      sha ? `<span class="machine" title="${esc(sha)}">${esc(sha.slice(0, 16))}…</span> (this exact GGUF's sha256, from /engine/health)`
          : "no engine reachable."}</span></div>
    <div class="drawer-row"><span class="label">calibration</span><span class="drawer-val">${
      nTotal ? `${nCalib}/${nTotal} tone dials calibrated on this model (from /steer/axes)` : "no axes reported."}</span></div>
    <div class="drawer-row"><span class="label">last CI</span><span class="drawer-val">no route surfaces this yet --
      run <span class="machine">clozn test-model</span> or <span class="machine">clozn ci check</span> and read its exit code / report.</span></div>
    <div class="drawer-row"><span class="label">drift</span><span class="drawer-val">no route surfaces this yet --
      <span class="machine">clozn quant-check &lt;A&gt; &lt;B&gt;</span> (quant ladder) or
      <span class="machine">clozn diff-model &lt;ref&gt; &lt;candidate&gt;</span> (base vs. fine-tune/merge).</span></div>
    <p class="quiet drawer-arriving">this whole block is the CI-flows-into-the-inspector seam -- the gate
      stays CLI/pipeline, this page is where its results would land next.</p>`;
}

/* ======================================================================== interactivity */

function wire(view, _state) {
  // No mutating controls on this page yet (every action here is either a real read or a named CLI
  // command) -- kept for parity with the other views and to avoid a silent dead spot if one is added.
  void view;
}
