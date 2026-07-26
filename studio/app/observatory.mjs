/* clozn studio -- the Observatory (Build 4): THE CASTING as the hero, per notes/
   UX_INFORMATION_ARCHITECTURE.md SS3 ("show the river, not the molecules") and SS7.6 (the casting
   storm-grammar spec, LOCKED) -- referenced by path only, never quoted into this file. Route:
   #/observatory (no run) and #/runs/<id>/observatory (a specific run), both wired in app.mjs ->
   renderObservatory(view, runId, light), which returns a teardown fn the router calls on navigation
   away (THE CASTING owns a canvas RAF loop that must be destroy()ed, not just abandoned).

   Style matches lens.mjs/model.mjs/behavior.mjs: one state object, small render-into-container
   functions, a delegated wire(view,state), and the "declared skeleton" pattern for anything with no
   backing route -- it states the plan/reason and never paints a control (or a cloud shape) that
   silently implies a measurement that wasn't taken.

   ============================================================================ per-panel status
   HERO (the casting): LIVE, best-effort, from a real run's own records when one is picked -- the
   real-cast assembly itself (getting from a run record to one castable.mjs data-contract object) now
   lives in studio/app/cast-feed.mjs (split out once it grew a second live-computed layer below); this
   file owns the shell, the panels, and turning a click into a real POST /runs/<id>/fork round trip.
     * sky + ground + per-token provenance threads: GET/POST /runs/<id>/influence-map (span excerpts
       stand in for individual "sky words"; this run's own recorded response is the ground).
     * per-token entropy: this run's own recorded trace.topk_entropy (a real, already-labeled top-k
       APPROXIMATION -- see clozn/runs/trace.py -- never full-vocabulary entropy).
     * ghosts (almosts): this run's own recorded trace.alternatives, each FORKABLE -- clicking one calls
       handleFork() below, which POSTs /runs/<id>/fork (position + the alt's own piece text), and on
       success re-assembles the cast from the returned CHILD run and casting.update()s with it in
       place (via history.pushState, not a hash navigation, so the canvas/RAF loop never tears down
       mid-storm); the fork-status line always shows the real route response, success or failure, and
       a link back to the parent run. See cast-feed.mjs's buildAlmosts for why every alt qualifies.
     * cloud shape (entropyByDepth): POST /jlens, once per a capped, evenly-sampled subset of the
       engine's fitted J-lens layers, read at this reply's OWN final token position, entropy computed
       as a top-k(k=5) renormalized softmax over the RAW LENS LOGITS the engine returns -- a real,
       computable, HONESTLY-APPROXIMATE quantity (never full-vocabulary entropy, which /jlens's top-k
       response cannot supply).
     * candidate cells / lead-change arcs / commit stratum: LIVE where the J-lens's own tokenization of
       this reply verifiably lines up with the recorded trace tokens -- cast-feed.mjs's buildArgument()
       reads the SAME per-layer /jlens rows already fetched for the cloud shape above (they carry a
       full per-POSITION readout, not just the last one -- confirmed live against a real run on the
       :8131 gateway), at the K<=8 highest-recorded-entropy answer positions, merged into one capped,
       labeled cell/arc set (see buildArgument's own comment for the exact cap counts and the honest
       "a linear lens always emits something, even from noise" caveat -- shallow layers routinely
       surface unrelated tokens). Genuinely unavailable (no J-lens, or the tokenization didn't align)
       still omits rather than guesses -- the component's own default (one uncontested cell) is the
       honest result of that omission, same as before.
     * PROVENANCE IS A HARD GATE: an empty tokens[].sources is a specific claim casting.mjs renders as
       "from the weights -- not from your words." This build never emits that claim without having
       actually computed and char-offset-verified the influence map against the run's own recorded
       response text -- if that verification fails, the WHOLE hero falls back to a visibly-labeled
       DEMO_CASTING cast rather than risk a false claim on any single token.
   Readout trajectory: LIVE when the hero's own J-lens calls succeeded (reused, not refetched) --
     else a named skeleton (no fitted J-lens / no run picked).
   Wiring (causal trace): LIVE, on demand -- POST /runs/<id>/causal-trace (the residual-site tracer;
     click-to-run, since it is not cheap).
   Whole-trace scorecard: LIVE -- GET /runs/<id>/spans (confidence_spans, zero re-generation).
   Residual telemetry: LIVE -- POST /engine/layers (per-layer activation L2 norm; needs an engine,
     NOT a fitted J-lens, so this stays live even when the cloud shape degrades).
   Trust strip: three DECLARED SKELETONS (see renderTrustStrip): quant fidelity (a real, unit-tested
     SEAM in clozn/receipts/quant_receipts.py, not wired to any server route -- a genuine gap, flagged,
     not faked), the causal-trace scorecard (the published 91.7% research result, explicitly labeled as
     a global finding, never presented as this run's own number), and the SAE brain map (lab-gated,
     POST /engine/concepts 409s on the product engine by design).
   Probe (free text): LIVE -- POST /jlens directly on typed text, a visibly different input mode than
     "this run" (SS0.2/SS5's "Scope," folded in).
*/
import { mountCasting, DEMO_CASTING } from "./casting.mjs";
import { assembleRealCast } from "./cast-feed.mjs";
import { mountCastingOptics } from "./casting-optics.mjs";
import { mountObservatoryWorkspace } from "./observatory-workspace.mjs";

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

const TELEMETRY_TEXT_CAP = 1200;
const JLENS_TOPK = 5;   // matches cast-feed.mjs's own copy -- the page-wide default topk for /jlens calls
                        // made directly from this file (the readout-trajectory label + the free probe)

function row(label, val) {
  return `<div class="drawer-row"><span class="label">${esc(label)}</span><span class="drawer-val">${esc(val)}</span></div>`;
}
function skeletonBlock(text) {
  return `<div class="skeleton-block"><p class="quiet" style="padding:0">${esc(text)}</p></div>`;
}
function fmtNum(x) { return Number.isFinite(x) ? Number(x).toFixed(4) : "—"; }

/* ======================================================================== entry point */

export async function renderObservatory(view, runId, light) {
  view.innerHTML = shell();
  const state = {
    light, runId, run: null, health: null, runsList: null, casting: null,
    castingOptics: null, workspace: null, currentCast: null, currentCastIsDemo: false,
    influence: null, jlensProbe: null, jlensTrajectory: null, spans: null, telemetry: null,
    wiring: { status: "idle", data: null, position: 0 },
    probe: { status: "idle", data: null },
  };
  wire(view, state);
  await Promise.all([loadRunPicker(view, state), loadHealth(view, state)]);
  const unwatchTheme = await mountHero(view, state);
  state.workspace = mountObservatoryWorkspace({
    workspace: window.cloznWorkspace,
    casting: state.casting,
  });
  if (state.currentCast) {
    state.workspace.updateCast(state.currentCast, { isDemo: state.currentCastIsDemo });
  }
  renderPanels(view, state);
  return () => {
    unwatchTheme();
    state.castingOptics && state.castingOptics.destroy();
    state.workspace && state.workspace.destroy();
    state.casting && state.casting.destroy();
  };
}

function shell() {
  return `
    <div class="observatory-page">
      <h1 class="view-title">Observatory</h1>
      <div class="view-sub">the shadow one question casts through the model -- the river, not the molecules</div>

      <div class="obs-hero-row">
        <span class="label">cast</span>
        <select id="obsRunPick" aria-label="pick a recorded run to cast">
          <option value="">-- pick a recorded run --</option>
        </select>
        <span class="quiet breathing" id="obsPickBusy" style="font-size:11px;padding:0">loading runs&hellip;</span>
        <span class="spacer" style="flex:1"></span>
        <span id="obsDemoBadge"></span>
      </div>
      <div class="obs-hero-wrap"><div class="obs-hero-mount" id="obsHero"></div></div>
      <p class="obs-hero-note" id="obsHeroNote"></p>
      <p class="quiet small-note" id="obsForkNote" style="padding:0"></p>

      <section class="obs-section" aria-labelledby="obs-readout-h">
        <h2 class="obs-h" id="obs-readout-h">Readout trajectory</h2>
        <p class="obs-sub quiet">the J-lens's own "disposed to say" readout, layer by layer -- the same
          numbers the cloud above is shaped from.</p>
        <div id="obsReadout"><p class="quiet breathing">&hellip;</p></div>
      </section>

      <section class="obs-section" aria-labelledby="obs-wiring-h">
        <h2 class="obs-h" id="obs-wiring-h">Wiring -- causal trace</h2>
        <p class="obs-sub quiet">which (layer, position) sites causally support one answer token,
          against a matched-random control -- the residual-site tracer, on demand (it is not cheap;
          nothing runs until you ask).</p>
        <div id="obsWiring"><p class="quiet breathing">&hellip;</p></div>
      </section>

      <section class="obs-section" aria-labelledby="obs-scorecard-h">
        <h2 class="obs-h" id="obs-scorecard-h">Whole-trace scorecard</h2>
        <p class="obs-sub quiet">the reply's own recorded per-token confidence, reshaped into a few
          honest bands -- zero re-generation.</p>
        <div id="obsScorecard"><p class="quiet breathing">&hellip;</p></div>
      </section>

      <section class="obs-section" aria-labelledby="obs-telemetry-h">
        <h2 class="obs-h" id="obs-telemetry-h">Residual telemetry</h2>
        <p class="obs-sub quiet">per-layer activation L2 norm, one causal forward over this reply's own
          text -- works on any engine, no fitted J-lens required.</p>
        <div id="obsTelemetry"><p class="quiet breathing">&hellip;</p></div>
      </section>

      <section class="obs-section" aria-labelledby="obs-trust-h">
        <h2 class="obs-h" id="obs-trust-h">Trust strip</h2>
        <p class="obs-sub quiet">the uniquely-clozn overlays -- shown here as what they honestly are
          today, live route or not.</p>
        <div class="trust-list" id="obsTrust"></div>
      </section>

      <section class="obs-section" aria-labelledby="obs-probe-h">
        <h2 class="obs-h" id="obs-probe-h">Probe -- free text</h2>
        <p class="obs-sub quiet">a different input mode than "this run": type anything and read the
          model's own layer-by-layer disposition toward it directly.</p>
        <div class="probe-row">
          <label class="label" for="obsProbeText">probe text</label>
          <input type="text" id="obsProbeText" placeholder="type anything&hellip;" autocomplete="off">
          <label class="label" for="obsProbeLayer">layer</label>
          <select id="obsProbeLayer"><option value="">default</option></select>
          <button class="btn-ghost small primary" type="button" data-probe-run>read</button>
        </div>
        <div id="obsProbeOut"></div>
      </section>
    </div>`;
}

/* ======================================================================== run picker + health */

async function loadRunPicker(view, state) {
  const res = await getJSON("/runs");
  const busy = view.querySelector("#obsPickBusy");
  if (busy) busy.hidden = true;
  const items = (res.ok && res.body && Array.isArray(res.body.runs)) ? res.body.runs : [];
  state.runsList = items;
  const sel = view.querySelector("#obsRunPick");
  if (!sel) return;
  const opts = items.slice(0, 60).map(r => {
    const id = r.id || r.run_id || "";
    const label = String(r.prompt_summary || r.prompt || id || "(untitled run)").slice(0, 64);
    return `<option value="${esc(id)}"${id === state.runId ? " selected" : ""}>${esc(label)}</option>`;
  }).join("");
  sel.innerHTML = `<option value="">-- pick a recorded run --</option>${opts}`;
}

async function loadHealth(view, state) {
  const res = await getJSON("/engine/health");
  if (res.ok && res.body && res.body.engine) state.health = res.body.engine;
}

/* ======================================================================== the hero: THE CASTING */

async function mountHero(view, state) {
  const mount = view.querySelector("#obsHero");
  if (!mount) return () => {};
  state.casting = mountCasting(mount, {
    onFork: (tokenIndex, altText) => handleFork(view, state, tokenIndex, altText),
    onSelect: tokenIndex => state.workspace && state.workspace.selectToken(tokenIndex, { source: "casting" }),
  });
  if (state.casting) {
    state.castingOptics = mountCastingOptics(mount, { activity: state.runId ? .68 : .28 });
  }
  const unwatch = state.casting ? watchTheme(state) : (() => {});
  await loadCast(view, state);
  return unwatch;
}

/* casting.mjs already self-refreshes its colors from tokens.css on BOTH a [data-theme] mutation and an
   OS prefers-color-scheme change (see its own MutationObserver/matchMedia listeners) -- but the spec
   asks THE HOST to also pass its night state and call setNight() explicitly on a theme change, for an
   immediate (not next-paint) flip. This reads the SAME localStorage key app.mjs's own theme toggle
   writes ("clozn.theme"), a small shared contract rather than reaching into app.mjs's private state. */
function watchTheme(state) {
  const mq = matchMedia("(prefers-color-scheme: dark)");
  const currentNight = () => {
    const t = localStorage.getItem("clozn.theme");
    return t ? t === "night" : mq.matches;
  };
  const onChange = () => { state.casting && state.casting.setNight(currentNight()); };
  mq.addEventListener("change", onChange);
  const mo = new MutationObserver(onChange);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  onChange();
  return () => { mq.removeEventListener("change", onChange); mo.disconnect(); };
}

function setForkStatus(view, html) {
  const el = view.querySelector("#obsForkNote");
  if (el) el.innerHTML = html;
}

/* Fetches state.runId's run record and assembles+shows its cast -- shared by the initial loadCast()
   below and by handleFork() (whose child run record it already has in hand, no extra GET needed). */
async function applyCastForRun(view, state, run) {
  state.run = run;
  const attempt = await assembleRealCast(run, state);
  if (!attempt) {
    const why = state.castUnavailableReason || "no response text, or its context-answer influence map "
      + "couldn't be computed/verified against the recorded reply";
    showDemoCast(view, state, DEMO_CASTING.fact,
      `this run doesn't have what's needed to cast honestly (${why}) -- showing a scripted demo cast `
      + `instead. The panels below still run live against this run where they can.`);
    return;
  }
  showRealCast(view, state, attempt);
}

async function loadCast(view, state) {
  if (!state.runId) {
    showDemoCast(view, state, DEMO_CASTING.fact, "no run selected -- showing a scripted demo cast.");
    return;
  }
  const runRes = await getJSON(`/runs/${encodeURIComponent(state.runId)}`);
  if (!runRes.ok || !runRes.body || runRes.body.error) {
    showDemoCast(view, state, DEMO_CASTING.fact,
      `run "${state.runId}" not found -- showing a scripted demo cast instead.`);
    return;
  }
  await applyCastForRun(view, state, runRes.body);
}

/* THE PLAY FEATURE: turn a ⑂ ghost click into a real POST /runs/<id>/fork round trip. On success the
   returned CHILD run's own cast replaces the hero IN PLACE (casting.update(), not a fresh mount --
   history.pushState updates the address bar without firing the app.mjs hash router, so the canvas/RAF
   loop never tears down mid-storm) and the below-hero panels re-render against the child. On failure
   (engine down, bad position, ...) the route's own error message is shown verbatim -- this never
   pretends a fork happened. */
async function handleFork(view, state, tokenIndex, altText) {
  const parentId = state.runId;
  setForkStatus(view, esc(`forking at token ${tokenIndex} → "${altText}"…`));
  const res = await postJSON(`/runs/${encodeURIComponent(parentId)}/fork`, { position: tokenIndex, token: altText });
  const child = (res.body && typeof res.body === "object") ? res.body : null;
  if (!res.ok || !child || child.error) {
    const msg = (child && child.error) || res.networkError || `fork request failed (HTTP ${res.status})`;
    setForkStatus(view, `⑂ fork at token ${esc(tokenIndex)} → "${esc(altText)}" `
      + `<b>failed</b>: ${esc(msg)}`);
    return;
  }
  const childId = child.id;
  if (!childId) {
    setForkStatus(view, `⑂ fork at token ${esc(tokenIndex)} → "${esc(altText)}" `
      + `<b>failed</b>: the server didn't return a child run id.`);
    return;
  }
  history.pushState(null, "", `#/runs/${encodeURIComponent(childId)}/observatory`);
  state.runId = childId;
  state.wiring = { status: "idle", data: null, position: 0 };
  state.spans = null; state.telemetry = null;
  await applyCastForRun(view, state, child);
  renderPanels(view, state);
  loadRunPicker(view, state);
  const parentRid = child.parent_run_id || parentId;
  // child.note already says so when retokenization was UNVERIFIABLE; a real DETECTED shift (retok
  // True from an actual mismatch, not just "couldn't check") isn't in that note's text, so it's
  // called out here too -- never silently dropped either way.
  const noteHasUnverified = /retokenization could not be verified/.test(child.note || "");
  const retokCaveat = (child.retokenized && !noteHasUnverified)
    ? " A token boundary shifted at the splice point (verified against the engine's own scorer)."
    : "";
  setForkStatus(view, `⑂ forked at token ${esc(tokenIndex)} → "${esc(altText)}" -- now viewing `
    + `<a class="machine" href="#/runs/${esc(childId)}/observatory">${esc(childId)}</a>. `
    + `${esc(child.note || "greedy what-if, not a sample.")}${esc(retokCaveat)} &middot; `
    + `<a class="machine" href="#/runs/${esc(parentRid)}/observatory">back to the parent run</a>.`);
  state.light && state.light.pulse(.5);
}

function showDemoCast(view, state, cast, reason) {
  state.currentCast = cast;
  state.currentCastIsDemo = true;
  state.casting && state.casting.update(cast);
  state.workspace && state.workspace.updateCast(cast, { isDemo: true });
  setDemoBadge(view, true);
  setHeroNote(view, `<b>Scripted demo cast.</b> ${esc(reason)}`);
}
function showRealCast(view, state, { cast, notes }) {
  state.currentCast = cast;
  state.currentCastIsDemo = false;
  state.casting && state.casting.update(cast);
  state.workspace && state.workspace.updateCast(cast, { isDemo: false });
  setDemoBadge(view, false);
  setHeroNote(view, `<b>Real cast, from this run's own records.</b> ${notes.map(esc).join(" ")}`);
}
function setDemoBadge(view, isDemo) {
  const el = view.querySelector("#obsDemoBadge");
  if (el) el.innerHTML = isDemo ? `<span class="dial-tag dial-tag-warn">scripted demo cast</span>` : "";
}
function setHeroNote(view, html) {
  const el = view.querySelector("#obsHeroNote");
  if (el) el.innerHTML = html;
}

/* Real-cast assembly (influence map -> sky/tokens/provenance, J-lens -> cloud shape + the argument
   cells/leadFlips/commitLayer) now lives in ./cast-feed.mjs -- see its own header comment for the full
   per-field breakdown. */

/* ======================================================================== below-hero panels */

function renderPanels(view, state) {
  renderReadoutTrajectory(view, state);
  renderWiringPanel(view, state);
  loadScorecard(view, state);
  loadTelemetry(view, state);
  renderTrustStrip(view, state);
}

function renderReadoutTrajectory(view, state) {
  const el = view.querySelector("#obsReadout");
  if (!el) return;
  const jl = state.jlensTrajectory;
  if (!state.runId) { el.innerHTML = skeletonBlock("pick a recorded run above -- the trajectory reads that run's own reply."); return; }
  if (!jl) { el.innerHTML = skeletonBlock("not computed for this run (see the hero note above)."); return; }
  if (!jl.available) { el.innerHTML = skeletonBlock(jl.reason || "J-lens unavailable on the currently serving engine."); return; }
  const rows = jl.perLayer.map((pl, i) => {
    const top1 = (pl.readouts[pl.readouts.length - 1] || [])[0];
    return `<tr><td>L${esc(pl.layer)}</td><td>${jl.entropyByDepth[i].toFixed(2)}</td>`
      + `<td>${top1 ? esc(top1.piece) + ` (${Number(top1.score).toFixed(2)})` : "—"}</td></tr>`;
  }).join("");
  el.innerHTML = `
    <table class="readout-table">
      <thead><tr><th>layer</th><th>entropy (bits, top-k&asymp;)</th><th>top-1 @ last position</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="quiet small-note">entropy is a top-k(k=${JLENS_TOPK}) renormalized approximation (raw lens
      logits softmaxed over just the returned candidates) -- a real but lower-bound-ish read, never true
      full-vocabulary entropy. "top-1 @ last position" is this reply's own final token position, read at
      each layer.</p>`;
}

function renderWiringPanel(view, state) {
  const el = view.querySelector("#obsWiring");
  if (!el) return;
  if (!state.runId || !state.run) {
    el.innerHTML = skeletonBlock("pick a recorded run above -- the tracer needs a run's own recorded final_prompt + response.");
    return;
  }
  const ent = (state.run.trace && Array.isArray(state.run.trace.topk_entropy)) ? state.run.trace.topk_entropy : [];
  let defaultPos = 0, best = -1;
  ent.forEach((e, i) => { if (Number.isFinite(e) && e > best) { best = e; defaultPos = i; } });
  if (!state.wiring.touched) state.wiring.position = defaultPos;
  el.innerHTML = `
    <div class="wiring-controls">
      <label class="label" for="obsWiringPos">token index</label>
      <input type="number" id="obsWiringPos" min="0" value="${state.wiring.position}">
      <button class="btn-ghost small primary" type="button" data-trace-token>trace this token</button>
      <span class="quiet small-note" style="margin:0;padding:0">defaults to this reply's most-entropic recorded token.</span>
    </div>
    <div id="obsWiringOut">${renderWiringResult(state)}</div>`;
}

function renderWiringResult(state) {
  const w = state.wiring;
  if (w.status === "idle") return `<p class="quiet" style="padding:0">not run yet.</p>`;
  if (w.status === "busy") return `<p class="quiet breathing" style="padding:0">tracing (matched-random controls, can take a while)&hellip;</p>`;
  const r = w.data;
  if (!r || r.ok !== true) {
    return `<p class="quiet" style="padding:0">${esc((r && (r.blocked || r.error)) || "trace unavailable.")}</p>`;
  }
  const t = r.target || {};
  const nodes = Array.isArray(r.nodes) ? r.nodes.length : 0;
  const cand = Array.isArray(r.all_candidates) ? r.all_candidates.length : 0;
  const ctl = r.controls || {};
  const sc = r.prediction_scorecard;
  const scLine = sc
    ? row("prediction check", `${sc.correct_predictions == null ? "—" : sc.correct_predictions} correct / `
      + `${sc.wrong_predictions == null ? "—" : sc.wrong_predictions} wrong of `
      + `${(sc.correct_predictions || 0) + (sc.wrong_predictions || 0)} observed`)
    : "";
  return `
    ${row("target", `token "${t.piece == null ? "" : t.piece}" @ position ${t.pos}`)}
    ${row("survivors", `${nodes} of ${cand} screened site(s) beat the matched-random control`)}
    ${row("control", `verdict ${ctl.verdict || "—"} · median|Δ| ${fmtNum(ctl.median_abs)} · noise floor ${fmtNum(ctl.noise_floor)}`)}
    ${scLine}
    <p class="quiet small-note">single-site causal support is usually near zero on these models (the
      distributed-function result) -- this is the causal skeleton for one token, not a full explanation.</p>`;
}

async function traceToken(view, state) {
  const input = view.querySelector("#obsWiringPos");
  const pos = Math.max(0, parseInt((input && input.value) || "0", 10) || 0);
  state.wiring.position = pos;
  state.wiring.touched = true;
  state.wiring.status = "busy";
  const out = view.querySelector("#obsWiringOut");
  if (out) out.innerHTML = renderWiringResult(state);
  const res = await postJSON(`/runs/${encodeURIComponent(state.runId)}/causal-trace`, { position: pos });
  state.wiring.status = "done";
  state.wiring.data = (res.body && typeof res.body === "object") ? res.body : { ok: false, blocked: "no response from the server" };
  state.light && state.light.pulse(.5);
  if (out) out.innerHTML = renderWiringResult(state);
}

async function loadScorecard(view, state) {
  const el = view.querySelector("#obsScorecard");
  if (!el) return;
  if (!state.runId) { el.innerHTML = skeletonBlock("pick a recorded run above."); return; }
  const res = await getJSON(`/runs/${encodeURIComponent(state.runId)}/spans`);
  state.spans = (res.ok && res.body) ? res.body : null;
  if (!state.spans || !Array.isArray(state.spans.spans) || !state.spans.spans.length) {
    el.innerHTML = skeletonBlock("no confidence-span data for this run.");
    return;
  }
  const bands = { strong: 0, okay: 0, shaky: 0 };
  let hesitations = 0;
  state.spans.spans.forEach(s => { if (bands[s.band] != null) bands[s.band]++; hesitations += s.hesitations || 0; });
  el.innerHTML = `
    ${row("summary", state.spans.summary || "—")}
    ${row("bands", `${bands.strong} strong · ${bands.okay} okay · ${bands.shaky} shaky (of ${state.spans.spans.length} spans)`)}
    ${row("hesitations", `${hesitations} token(s) flagged shaky (below this server's own confidence floor)`)}
    <p class="quiet small-note">reshaped straight from this run's own recorded per-token confidence --
      nothing re-generated, nothing estimated.</p>`;
}

async function loadTelemetry(view, state) {
  const el = view.querySelector("#obsTelemetry");
  if (!el) return;
  if (!state.runId || !state.run) { el.innerHTML = skeletonBlock("pick a recorded run above."); return; }
  const full = String(state.run.response || state.run.final_prompt || "");
  const text = full.slice(0, TELEMETRY_TEXT_CAP);
  if (!text.trim()) { el.innerHTML = skeletonBlock("this run has no text to read activations from."); return; }
  const res = await postJSON("/engine/layers", { text });
  state.telemetry = (res.ok && res.body) ? res.body : null;
  if (!state.telemetry || !Array.isArray(state.telemetry.layer_mean) || !state.telemetry.layer_mean.length) {
    el.innerHTML = skeletonBlock((res.body && res.body.error) || "no engine reachable, or /engine/layers errored.");
    return;
  }
  const norms = state.telemetry.layer_mean;
  const max = Math.max(...norms, 0.001);
  const rows = norms.map((n, i) => `
    <div class="telem-row"><span>L${i}</span>
      <span class="telem-bar"><i style="width:${Math.max(2, Math.round(n / max * 100))}%"></i></span>
      <span class="telem-val">${n.toFixed(1)}</span></div>`).join("");
  el.innerHTML = `${rows}<p class="quiet small-note">mean residual L2 norm per layer, one causal forward
    over ${state.telemetry.n_tokens} token(s) of this reply's own text${text.length < full.length ? " (truncated)" : ""}.</p>`;
}

function renderTrustStrip(view, state) {
  const el = view.querySelector("#obsTrust");
  if (!el) return;
  el.innerHTML = [
    trustRow(false, "Quant fidelity (Q4-vs-FP cosine)", "no live route serves this today -- "
      + "clozn/receipts/quant_receipts.py's quant_receipt_for_run() is a real, unit-tested seam for a "
      + "one-run, two-quant-arm diff, but it isn't wired to any server route yet (it needs two live "
      + "engine processes, one per quant file). Flagged as a genuine gap, not faked here."),
    trustRow(false, "Causal-trace scorecard", "the published 91.7% predicted-vs-observed result is a "
      + "GLOBAL research finding (docs/research/DISTRIBUTED_FUNCTION.md), never this run's own number "
      + "-- the Wiring panel above runs the SAME tracer live, for one token of the run you picked."),
    trustRow(false, "SAE brain map, with its own null", "lab-gated by design: POST /engine/concepts "
      + "409s on the product engine (“available in `clozn lab qwen`”). The showcase pairing "
      + "-- feature activation alongside its causal-mass-vs-random-control debunk -- lives in the lab "
      + "build, not here."),
  ].join("");
}
function trustRow(live, title, body) {
  return `<div class="trust-row"><span class="cap-dot${live ? " cap-dot-on" : ""}" aria-hidden="true"></span>
    <div class="trust-row-body"><div class="trust-row-h">${esc(title)}</div>
      <div class="quiet" style="padding:0">${esc(body)}</div></div></div>`;
}

/* ======================================================================== probe (free text) */

async function runProbe(view, state) {
  const textEl = view.querySelector("#obsProbeText");
  const layerEl = view.querySelector("#obsProbeLayer");
  const text = ((textEl && textEl.value) || "").trim();
  const out = view.querySelector("#obsProbeOut");
  if (!text) { if (out) out.innerHTML = `<p class="quiet" style="padding:0">type something to probe first.</p>`; return; }
  const layerVal = layerEl && layerEl.value !== "" ? Number(layerEl.value) : undefined;
  if (out) out.innerHTML = `<p class="quiet breathing" style="padding:0">reading&hellip;</p>`;
  const body = { text, topk: JLENS_TOPK };
  if (layerVal !== undefined) body.layer = layerVal;
  const res = await postJSON("/jlens", body);
  state.probe.status = "done";
  state.probe.data = (res.body && typeof res.body === "object") ? res.body : null;
  state.light && state.light.pulse(.45);
  const d = state.probe.data;
  if (!d || !d.available) {
    if (out) out.innerHTML = skeletonBlock((d && d.reason) || "J-lens unavailable.");
    return;
  }
  if (layerEl && layerEl.dataset.filled !== "1" && Array.isArray(d.available_layers) && d.available_layers.length) {
    layerEl.dataset.filled = "1";
    const cur = layerEl.value;
    layerEl.innerHTML = `<option value="">default (L${esc(d.layer)})</option>`
      + d.available_layers.map(l => `<option value="${l}">L${l}</option>`).join("");
    layerEl.value = cur || "";
  }
  const rows = (d.tokens || []).map((tok, i) => {
    const top = (d.readouts[i] || []).slice(0, 3)
      .map(r => `${esc(r.piece)} <span class="quiet" style="padding:0">(${Number(r.score).toFixed(2)})</span>`)
      .join(" · ");
    return `<tr><td>${esc(tok)}</td><td>${top || "—"}</td></tr>`;
  }).join("");
  const prov = d.provenance || {};
  if (out) out.innerHTML = `
    <table class="readout-table">
      <thead><tr><th>token</th><th>disposed to say next (top-3, raw lens logit)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="quiet small-note">layer ${esc(d.layer)} of [${(d.available_layers || []).join(", ")}]
      ${prov.fit_model ? ` · J-lens fitted on ${esc(prov.fit_model)}` : ""} -- a deterministic
      linear read, not generation; scores are raw lens logits, not probabilities.</p>`;
}

/* ======================================================================== interactivity */

function wire(view, state) {
  view.addEventListener("change", e => {
    if (e.target.id === "obsRunPick") {
      const id = e.target.value;
      location.hash = id ? `#/runs/${encodeURIComponent(id)}/observatory` : "#/observatory";
    }
  });
  view.addEventListener("click", e => {
    if (e.target.closest("[data-trace-token]")) { traceToken(view, state); return; }
    if (e.target.closest("[data-probe-run]")) { runProbe(view, state); return; }
  });
  view.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.target.id === "obsProbeText") runProbe(view, state);
  });
}
