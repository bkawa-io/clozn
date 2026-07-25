/* clozn studio -- the Behavior page (Build 3): the one READ-WRITE surface. Config here is per-model
   (a slim "applies to: <model>" line says which -- picking the model happens on Model/the header
   switcher, never here). Sections, in the order the locked rebuild spec lays them out (local notes/
   UX_INFORMATION_ARCHITECTURE.md, referenced by path only, never quoted into this file):

     dials (tone axes + concept steering) -> anchored memory (the centerpiece) -> settings -> profiles

   Route: #/behavior, wired in app.mjs -> renderBehavior(view, light). Matches lens.mjs's style: a plain
   state object, small render-into-container functions, one delegated wire(view,state) for events, and an
   honest "declared skeleton" for anything with no backing route -- it states the plan/reason and never
   paints a control that does nothing (a disabled button with a visible reason is fine; a live-looking one
   that silently no-ops is not).

   LIVE-WIRED this build: /steer/axes + /steer/set (+ /steer/check preview, /steer/custom_delete),
   /steer/concept/set + /steer/concept/check, /memory/anchored/list + /fit + /toggle + /delete_term +
   /whatlearned, /memory/cards (read, to pick a card to fit), /sampling/mode (GET+POST),
   /profiles/list + /save + /switch + /delete, /engine/health (for the applies-to line),
   /guard/mode (GET+POST) -- the disposition guard's persisted server-wide default now has a real toggle
   here (a per-request `clozn_guard` field on /v1/chat/completions still always overrides it -- this page
   only ever sets what a request that says nothing about clozn_guard falls back to). */

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

function basename(p) { return String(p || "").split(/[\\/]/).pop() || ""; }

/* ======================================================================== entry point */

export async function renderBehavior(view, light) {
  view.innerHTML = shell();
  const state = {
    light, health: null, axes: [], concepts: new Map(), cards: null,
    bags: null, sampling: null, guard: null, profiles: null, activeProfile: null,
  };
  wire(view, state);
  await Promise.all([
    loadModelLine(view, state),
    loadDials(view, state),
    loadAnchored(view, state),
    loadSettings(view, state),
    loadProfiles(view, state),
  ]);
}

function shell() {
  return `
    <div class="behavior-page">
      <h1 class="view-title">Behavior</h1>
      <div class="view-sub" id="bhvModelLine">applies to: <span class="quiet breathing">checking…</span></div>

      <section class="bhv-section" aria-labelledby="bhv-dials-h">
        <h2 class="bhv-h" id="bhv-dials-h">Dials</h2>
        <p class="bhv-sub quiet">tone axes bend how it says things; concept steering leans the reply toward
          a word's direction in the model's own space. Every change here applies immediately.</p>
        <div id="bhvDials"><p class="quiet breathing">reading axes…</p></div>
        <h3 class="bhv-h3">steer toward a concept</h3>
        <div id="bhvConcepts"><p class="quiet breathing">…</p></div>
      </section>

      <section class="bhv-section" aria-labelledby="bhv-anchor-h">
        <h2 class="bhv-h" id="bhv-anchor-h">Anchored memory</h2>
        <p class="bhv-sub quiet">a card's memory decomposed into a readable, weighted bag of words --
          read it, delete a word, watch the rest re-fit. This is memory you can edit word by word, not a
          black box you can only clear.</p>
        <div id="bhvAnchoredFit"><p class="quiet breathing">…</p></div>
        <button class="btn-ghost small" type="button" data-whatlearned>what do you remember?</button>
        <div id="bhvWhatlearned"></div>
        <div id="bhvAnchored"><p class="quiet breathing">reading anchored bags…</p></div>
      </section>

      <section class="bhv-section" aria-labelledby="bhv-settings-h">
        <h2 class="bhv-h" id="bhv-settings-h">Settings</h2>
        <div id="bhvSettings"><p class="quiet breathing">reading decode settings…</p></div>
      </section>

      <section class="bhv-section" aria-labelledby="bhv-profiles-h">
        <h2 class="bhv-h" id="bhv-profiles-h">Profiles</h2>
        <p class="bhv-sub quiet">named bundles of the above (dials + active cards) -- switch personas
          instantly, no retrain.</p>
        <div id="bhvProfiles"><p class="quiet breathing">reading profiles…</p></div>
      </section>

      <p class="bhv-caveat quiet">every edit on this page is a real, reversible change to the model as it
        is served right now -- and every one of them is disclosed on the Influences lens.</p>
    </div>`;
}

/* ======================================================================== applies-to line */

async function loadModelLine(view, state) {
  const res = await getJSON("/engine/health");
  const el = view.querySelector("#bhvModelLine");
  if (!el) return;
  if (!res.ok || !res.body || !res.body.engine) {
    el.innerHTML = `no model currently serving -- see <a href="#/model">Model</a>.`;
    return;
  }
  state.health = res.body.engine;
  const name = basename(state.health.model) || state.health.model || "?";
  el.innerHTML = `applies to: <b class="machine">${esc(name)}</b>
    <span class="quiet"> -- pick a different model on </span><a href="#/model">Model</a>`;
}

/* ======================================================================== dials: tone axes */

function axisRowHTML(ax) {
  const poles = Array.isArray(ax.poles) && ax.poles.length === 2 ? ax.poles : ["+", "-"];
  const [pos, neg] = poles;
  const max = typeof ax.max === "number" && ax.max > 0 ? ax.max : 1.5;
  const val = typeof ax.value === "number" ? ax.value : 0;
  const tag = ax.library ? `<span class="dial-tag">library</span>`
            : ax.custom ? `<span class="dial-tag dial-tag-custom">yours</span>` : "";
  const calib = ax.calibrated
    ? `<span class="dial-tag dial-tag-calib" title="usable range measured live on this model">calibrated</span>`
    : "";
  return `
    <div class="dial-row" data-axis="${esc(ax.name)}">
      <div class="dial-head">
        <label class="dial-label" for="dial-${esc(ax.name)}">
          <span class="pole-neg">${esc(neg)}</span>
          <span class="dial-name machine">${esc(ax.name)}</span>
          <span class="pole-pos">${esc(pos)}</span>
        </label>
        ${tag}${calib}
        <span class="dial-val machine" data-dial-val="${esc(ax.name)}">${val.toFixed(2)}</span>
      </div>
      <input type="range" class="dial-slider" id="dial-${esc(ax.name)}" data-dial-input="${esc(ax.name)}"
        min="${(-max).toFixed(2)}" max="${max.toFixed(2)}" step="0.05" value="${val}"
        aria-label="${esc(ax.name)} axis, from ${esc(neg)} to ${esc(pos)}"
        aria-valuetext="${val.toFixed(2)} toward ${val >= 0 ? esc(pos) : esc(neg)}">
      <div class="dial-actions">
        <button class="btn-ghost small" type="button" data-preview-axis="${esc(ax.name)}">preview</button>
        ${ax.custom ? `<button class="btn-ghost small" type="button" data-delete-axis="${esc(ax.name)}">remove</button>` : ""}
      </div>
      <div class="dial-preview quiet" data-preview-out="${esc(ax.name)}" hidden></div>
    </div>`;
}

async function loadDials(view, state) {
  // /steer/axes is dispatched through the substrate's generic handler (Substrate._steer), which the
  // server only wires up for POST -- there is no GET route for it, even though it's a pure read.
  const res = await postJSON("/steer/axes", {});
  const el = view.querySelector("#bhvDials");
  if (!res.ok || !res.body || !Array.isArray(res.body.axes)) {
    if (el) el.innerHTML = `<p class="quiet">axes unavailable -- ${esc((res.body && res.body.error) || "no engine reachable.")}</p>`;
  } else {
    state.axes = res.body.axes;
    if (el) el.innerHTML = state.axes.length
      ? `<div class="dial-list">${state.axes.map(axisRowHTML).join("")}</div>`
      : `<p class="quiet">no axes reported.</p>`;
  }
  renderConcepts(view, state);
}

/* ======================================================================== dials: concept steering */

function renderConcepts(view, state) {
  const el = view.querySelector("#bhvConcepts");
  if (!el) return;
  const chips = [...state.concepts.entries()].map(([c, v]) => `
    <span class="chip chip-concept"><i></i><span class="machine">${esc(c)}</span>
      <span class="chip-val machine">${(+v).toFixed(2)}</span>
      <button class="chip-x" type="button" data-remove-concept="${esc(c)}"
        aria-label="stop steering toward ${esc(c)}">&times;</button></span>`).join("");
  el.innerHTML = `
    <div class="concept-chips">${chips || `<span class="quiet">nothing steered this session.</span>`}</div>
    <div class="concept-add">
      <label class="label" for="conceptWord">steer toward</label>
      <input class="text-input machine" id="conceptWord" placeholder="a word, e.g. nautical" autocomplete="off">
      <label class="label" for="conceptStrength">strength</label>
      <input class="text-input machine narrow" id="conceptStrength" type="number" step="0.1" min="-2" max="2" value="1">
      <button class="btn-ghost small" type="button" data-concept-preview>preview</button>
      <button class="btn-ghost small primary" type="button" data-concept-steer>+ steer</button>
    </div>
    <div class="concept-preview quiet" id="conceptPreviewOut" hidden></div>
    <p class="quiet small-note">there is no listing route for active concept dials yet -- chips above
      reflect only what changed this session (the server's own active-concept map rides back on every
      change, so a dial set earlier this session, or already live from before you opened this page, still
      shows up the moment you touch it).</p>`;
}

/* ======================================================================== anchored memory */

async function loadAnchored(view, state) {
  // /memory/cards is the same substrate-generic-dispatch story as /steer/axes above: POST-only.
  const [cardsRes, bagsRes] = await Promise.all([postJSON("/memory/cards", {}), getJSON("/memory/anchored/list")]);
  state.cards = (cardsRes.ok && cardsRes.body && Array.isArray(cardsRes.body.cards)) ? cardsRes.body.cards : [];
  renderAnchorFit(view, state);
  if (bagsRes.ok && bagsRes.body && Array.isArray(bagsRes.body.bags)) {
    state.bags = bagsRes.body.bags;
  } else {
    state.bags = [];
  }
  renderAnchored(view, state);
}

function renderAnchorFit(view, state) {
  const el = view.querySelector("#bhvAnchoredFit");
  if (!el) return;
  if (!state.cards.length) {
    el.innerHTML = `<p class="quiet">no cards to anchor yet -- cards arrive from your own client's calls
      to <span class="machine">/memory/add</span> (or the propose-memory loop); once one exists here, you
      can fit it into a readable bag below.</p>`;
    return;
  }
  const options = state.cards.map(c => {
    const already = state.bags && state.bags.some(b => b.card_id === c.id);
    const preview = String(c.text || "").slice(0, 64);
    return `<option value="${esc(c.id)}">${already ? "[refit] " : ""}${esc(preview)}${c.text && c.text.length > 64 ? "…" : ""}</option>`;
  }).join("");
  el.innerHTML = `
    <div class="anchor-fit-row">
      <label class="label" for="fitCard">fit from card</label>
      <select class="text-input machine" id="fitCard">${options}</select>
      <label class="label" for="fitK">k</label>
      <input class="text-input machine narrow" id="fitK" type="number" min="1" max="8" value="4">
      <button class="btn-ghost small primary" type="button" data-fit-bag>fit</button>
    </div>
    <div class="quiet" id="fitResult" hidden></div>`;
}

function alphaTableHTML(bag) {
  const terms = Array.isArray(bag.terms) ? bag.terms : [];
  const rows = terms.map(t => {
    const a = Number(t.alpha) || 0;
    return `<tr>
      <td class="machine alpha-val">${a >= 0 ? "+" : ""}${a.toFixed(3)}</td>
      <td class="alpha-word">${esc(t.token)}</td>
      <td><button class="term-x" type="button" data-delete-term data-card="${esc(bag.card_id)}"
        data-token="${esc(t.token)}" aria-label="remove '${esc(t.token)}' from this memory">&times;</button></td>
    </tr>`;
  }).join("");
  return `<table class="alpha-table"><tbody>${rows}</tbody></table>`;
}

function bagBlockHTML(bag) {
  const cos = typeof bag.reconstruction_cos === "number" ? bag.reconstruction_cos.toFixed(2) : "?";
  const fitted = bag.fitted_at || "?";
  return `
    <div class="pane anchor-bag" data-card-id="${esc(bag.card_id)}">
      <div class="pane-head">
        <span class="label">card</span>
        <span class="anchor-card-text">"${esc(String(bag.card_text || "").slice(0, 90))}${(bag.card_text || "").length > 90 ? "…" : ""}"</span>
        <label class="anchor-toggle">
          <input type="checkbox" data-toggle-bag="${esc(bag.card_id)}" ${bag.on !== false ? "checked" : ""}>
          <span class="machine">${bag.on !== false ? "on" : "off"}</span>
        </label>
      </div>
      <div class="pane-body">
        ${alphaTableHTML(bag)}
        <div class="anchor-meta quiet machine">fit cos ${cos} · k=${esc(bag.k)} · L${esc(bag.layer)} · fitted ${esc(fitted)}</div>
      </div>
    </div>`;
}

function renderAnchored(view, state) {
  const el = view.querySelector("#bhvAnchored");
  if (!el) return;
  const bags = state.bags || [];
  el.innerHTML = bags.length
    ? `<div class="anchor-list">${bags.map(bagBlockHTML).join("")}</div>`
    : `<p class="quiet">no anchored bags yet -- fit one from a card above.</p>`;
}

async function loadWhatlearned(view, state) {
  const el = view.querySelector("#bhvWhatlearned");
  if (!el) return;
  el.innerHTML = `<p class="quiet breathing">reading…</p>`;
  const res = await getJSON("/memory/anchored/whatlearned");
  if (!res.ok || !res.body) {
    el.innerHTML = `<p class="quiet">whatlearned unavailable.</p>`;
    return;
  }
  const bags = Array.isArray(res.body.bags) ? res.body.bags : [];
  const body = bags.length
    ? bags.map(b => `<pre class="machine whatlearned-table">${esc(b.table || "")}</pre>`).join("")
    : `<p class="quiet">no active anchored memory right now.</p>`;
  el.innerHTML = `
    <div class="whatlearned-block">
      <div class="label">what do you remember? -- a lookup, not a self-report</div>
      ${body}
      <p class="quiet small-note">${esc(res.body.note || "")}</p>
    </div>`;
}

/* ======================================================================== settings */

async function loadSettings(view, state) {
  const [samplingRes, guardRes] = await Promise.all([getJSON("/sampling/mode"), getJSON("/guard/mode")]);
  const el = view.querySelector("#bhvSettings");
  if (!el) return;
  if (samplingRes.ok && samplingRes.body) {
    state.sampling = samplingRes.body;
    el.innerHTML = `
      <h3 class="bhv-h3">decode</h3>
      <div class="settings-row">
        <label class="switch-label"><input type="checkbox" id="sampSampling" ${state.sampling.sampling ? "checked" : ""}>
          <span class="machine">sample (off = greedy)</span></label>
      </div>
      <div class="settings-grid">
        ${settingsNumber("sampTemp", "temperature", state.sampling.sample_temperature, 0, 2, 0.05)}
        ${settingsNumber("sampTopP", "top_p", state.sampling.sample_top_p, 0, 1, 0.01)}
        ${settingsNumber("sampTopK", "top_k", state.sampling.sample_top_k, 0, 200, 1)}
        ${settingsNumber("sampRep", "repeat_penalty", state.sampling.sample_repeat_penalty, 0.5, 2, 0.01)}
      </div>
      <button class="btn-ghost small primary" type="button" data-save-sampling>save decode settings</button>
      <div class="quiet" id="samplingSaved" hidden></div>`;
  } else {
    el.innerHTML = `<p class="quiet">decode settings unavailable.</p>`;
  }
  state.guard = (guardRes.ok && guardRes.body) ? guardRes.body : null;
  el.insertAdjacentHTML("beforeend", guardBlockHTML(state.guard));
}

function guardBlockHTML(guard) {
  if (!guard) {
    return `
      <h3 class="bhv-h3">disposition guard</h3>
      <p class="quiet">guard default unavailable.</p>`;
  }
  const g = guard.guard || {};
  const concepts = Array.isArray(g.concepts) ? g.concepts.join(", ") : "";
  const counter = typeof g.counter_strength === "number" ? g.counter_strength : -0.5;
  const maxFires = typeof g.max_fires === "number" ? g.max_fires : 3;
  return `
    <h3 class="bhv-h3">disposition guard</h3>
    <p class="bhv-sub quiet">a closed-loop guard that detects a chosen concept's disposition rising
      mid-reply and steers the continuation away from it -- present-tense detect-and-correct, not
      predictive. This is the server-wide default for a request that sends no <span class="machine">clozn_guard</span>
      field at all; an explicit <span class="machine">clozn_guard</span> on
      <span class="machine">POST /v1/chat/completions</span> always overrides it, including turning it off
      for that one call.</p>
    <div class="settings-row">
      <label class="switch-label"><input type="checkbox" id="guardEnabled" ${guard.enabled ? "checked" : ""}>
        <span class="machine">on by default</span></label>
    </div>
    <div class="settings-row settings-field">
      <label class="label" for="guardConcepts">concepts (comma-separated)</label>
      <input class="text-input machine" id="guardConcepts" placeholder="e.g. violence, self-harm"
        value="${esc(concepts)}">
    </div>
    <div class="settings-grid">
      ${settingsNumber("guardCounter", "counter_strength", counter, -2, 0, 0.1)}
      ${settingsNumber("guardMaxFires", "max_fires", maxFires, 1, 10, 1)}
    </div>
    <button class="btn-ghost small primary" type="button" data-save-guard>save guard default</button>
    <div class="quiet" id="guardSaved" hidden></div>
    <p class="quiet small-note">turning this on with no concepts is refused (400) -- an empty concepts
      list is what turns it off, not what enables it with nothing to guard against. calibration (catch/
      false-positive rate, per concept) is a small-battery signal-design result, not a public reliability
      claim at any scale.</p>`;
}

function settingsNumber(id, label, value, min, max, step) {
  const v = (typeof value === "number") ? value : "";
  return `
    <div class="settings-field">
      <label class="label" for="${id}">${esc(label)}</label>
      <input class="text-input machine narrow" id="${id}" type="number" min="${min}" max="${max}" step="${step}" value="${v}">
    </div>`;
}

/* ======================================================================== profiles */

async function loadProfiles(view, state) {
  const res = await getJSON("/profiles/list");
  const el = view.querySelector("#bhvProfiles");
  if (!el) return;
  if (!res.ok || !res.body) {
    el.innerHTML = `<p class="quiet">profiles unavailable.</p>`;
    return;
  }
  state.profiles = Array.isArray(res.body.profiles) ? res.body.profiles : [];
  state.activeProfile = res.body.active || null;
  renderProfiles(view, state);
}

function profileRowHTML(p, activeName) {
  const isActive = p.name === activeName;
  const nCards = Array.isArray(p.cards) ? p.cards.length : 0;
  const nDials = p.dials && typeof p.dials === "object" ? Object.keys(p.dials).length : 0;
  return `
    <li class="profile-row${isActive ? " profile-active" : ""}">
      <div class="profile-name machine">${esc(p.name)}${isActive ? ` <span class="dial-tag dial-tag-calib">active</span>` : ""}</div>
      <div class="profile-desc quiet">${esc(p.description || "")}</div>
      <div class="profile-meta quiet machine">${nCards} card${nCards === 1 ? "" : "s"} · ${nDials} dial${nDials === 1 ? "" : "s"}</div>
      <div class="profile-actions">
        <button class="btn-ghost small" type="button" data-switch-profile="${esc(p.name)}" ${isActive ? "disabled" : ""}
          title="${isActive ? "already active" : "apply this profile now"}">switch</button>
        <button class="btn-ghost small" type="button" data-delete-profile="${esc(p.name)}" ${isActive ? "disabled" : ""}
          title="${isActive ? "cannot delete the active profile" : "delete this profile"}">delete</button>
      </div>
    </li>`;
}

function renderProfiles(view, state) {
  const el = view.querySelector("#bhvProfiles");
  if (!el) return;
  const list = state.profiles || [];
  el.innerHTML = `
    ${list.length
      ? `<ul class="profile-list">${list.map(p => profileRowHTML(p, state.activeProfile)).join("")}</ul>`
      : `<p class="quiet">no saved profiles yet.</p>`}
    <div class="profile-save-row">
      <label class="label" for="profName">name</label>
      <input class="text-input machine" id="profName" placeholder="e.g. work" autocomplete="off">
      <label class="label" for="profDesc">description</label>
      <input class="text-input machine" id="profDesc" placeholder="optional" autocomplete="off">
      <button class="btn-ghost small primary" type="button" data-save-profile>save current as new profile</button>
    </div>
    <div class="quiet" id="profileSaved" hidden></div>`;
}

/* ======================================================================== confirm-to-destroy */

function armConfirm(btn, label = "confirm?") {
  if (btn.dataset.armed === "1") { delete btn.dataset.armed; return true; }
  btn.dataset.armed = "1";
  btn.dataset.origText = btn.textContent;
  btn.textContent = label;
  btn.classList.add("btn-armed");
  setTimeout(() => {
    if (btn.isConnected && btn.dataset.armed === "1") {
      delete btn.dataset.armed;
      btn.textContent = btn.dataset.origText || label;
      btn.classList.remove("btn-armed");
    }
  }, 3000);
  return false;
}

/* ======================================================================== interactivity */

/* Neither /steer/check nor /steer/concept/check default their `prompt` server-side (an empty prompt
   generates from nothing) -- a small fixed seed makes the A/B actually legible. */
const PREVIEW_PROMPT = "Tell me about your day.";

function wire(view, state) {
  view.addEventListener("input", e => {
    const name = e.target.dataset && e.target.dataset.dialInput;
    if (name) {
      const val = view.querySelector(`[data-dial-val="${CSS.escape(name)}"]`);
      if (val) val.textContent = Number(e.target.value).toFixed(2);
      e.target.setAttribute("aria-valuetext", `${Number(e.target.value).toFixed(2)}`);
    }
  });

  view.addEventListener("change", async e => {
    const name = e.target.dataset && e.target.dataset.dialInput;
    if (name) {
      const value = Number(e.target.value);
      const res = await postJSON("/steer/set", { name, value });
      if (res.ok) state.light && state.light.pulse(.5);
      return;
    }
    const cardId = e.target.dataset && e.target.dataset.toggleBag;
    if (cardId) {
      const on = e.target.checked;
      const label = e.target.closest(".anchor-toggle").querySelector("span");
      const res = await postJSON("/memory/anchored/toggle", { card_id: cardId, on });
      if (res.ok && res.body && res.body.ok) {
        if (label) label.textContent = on ? "on" : "off";
        state.light && state.light.pulse(.5);
      } else {
        e.target.checked = !on;   // the write didn't take -- don't leave the switch lying
      }
    }
  });

  view.addEventListener("click", async e => {
    /* -- dial preview (A/B via /steer/check) -- */
    const previewBtn = e.target.closest("[data-preview-axis]");
    if (previewBtn) {
      const name = previewBtn.dataset.previewAxis;
      const input = view.querySelector(`[data-dial-input="${CSS.escape(name)}"]`);
      const value = input ? Number(input.value) : 1;
      const out = view.querySelector(`[data-preview-out="${CSS.escape(name)}"]`);
      if (out) { out.hidden = false; out.innerHTML = `<span class="breathing">previewing…</span>`; }
      const res = await postJSON("/steer/check", { name, value, prompt: PREVIEW_PROMPT });
      if (out) {
        if (res.ok && res.body) {
          out.innerHTML = `<div class="ab-preview"><div><span class="label">baseline</span> ${esc(res.body.baseline || "")}</div>
            <div><span class="label">steered</span> ${esc(res.body.steered || "")}</div></div>`;
        } else {
          out.textContent = (res.body && res.body.error) || "preview failed.";
        }
      }
      state.light && state.light.pulse(.4);
      return;
    }

    /* -- delete a custom axis (confirm-to-destroy) -- */
    const delAxisBtn = e.target.closest("[data-delete-axis]");
    if (delAxisBtn) {
      if (!armConfirm(delAxisBtn)) return;
      const name = delAxisBtn.dataset.deleteAxis;
      const res = await postJSON("/steer/custom_delete", { name });
      if (res.ok) { state.light && state.light.pulse(.5); loadDials(view, state); }
      return;
    }

    /* -- concept steering: preview / steer / remove -- */
    if (e.target.closest("[data-concept-preview]") || e.target.closest("[data-concept-steer]")) {
      const isPreview = !!e.target.closest("[data-concept-preview]");
      const word = (view.querySelector("#conceptWord").value || "").trim();
      const strength = Number(view.querySelector("#conceptStrength").value || 1);
      const out = view.querySelector("#conceptPreviewOut");
      if (!word) { if (out) { out.hidden = false; out.textContent = "need a word to steer toward."; } return; }
      if (isPreview) {
        if (out) { out.hidden = false; out.innerHTML = `<span class="breathing">previewing…</span>`; }
        const res = await postJSON("/steer/concept/check", { concept: word, strength, prompt: PREVIEW_PROMPT });
        if (out) {
          if (res.ok && res.body && res.body.steered != null) {
            out.innerHTML = `<div class="ab-preview"><div><span class="label">baseline</span> ${esc(res.body.baseline || "")}</div>
              <div><span class="label">steered</span> ${esc(res.body.steered || "")}</div></div>`;
          } else {
            out.textContent = (res.body && (res.body.note || res.body.error || res.body.blocked)) || "preview unavailable.";
          }
        }
        state.light && state.light.pulse(.4);
      } else {
        const res = await postJSON("/steer/concept/set", { concept: word, strength });
        if (res.body && res.body.active && typeof res.body.active === "object") {
          state.concepts = new Map(Object.entries(res.body.active));
        }
        if (res.ok && res.body && res.body.ok) {
          renderConcepts(view, state);
          state.light && state.light.pulse(.55);
        } else if (out) {
          out.hidden = false;
          out.textContent = (res.body && (res.body.note || res.body.error || res.body.blocked)) || "could not steer toward that word.";
        }
      }
      return;
    }
    const rmConcept = e.target.closest("[data-remove-concept]");
    if (rmConcept) {
      const concept = rmConcept.dataset.removeConcept;
      const res = await postJSON("/steer/concept/set", { concept, strength: 0 });
      if (res.body && res.body.active && typeof res.body.active === "object") {
        state.concepts = new Map(Object.entries(res.body.active));
      } else {
        state.concepts.delete(concept);
      }
      renderConcepts(view, state);
      state.light && state.light.pulse(.45);
      return;
    }

    /* -- anchored memory: fit / toggle / delete term / whatlearned -- */
    const fitBtn = e.target.closest("[data-fit-bag]");
    if (fitBtn) {
      const cardId = view.querySelector("#fitCard").value;
      const k = Number(view.querySelector("#fitK").value || 4);
      const result = view.querySelector("#fitResult");
      if (result) { result.hidden = false; result.innerHTML = `<span class="breathing">fitting…</span>`; }
      const res = await postJSON("/memory/anchored/fit", { card_id: cardId, k });
      if (res.body && res.body.ok) {
        if (result) result.innerHTML = `fit. reconstruction cos ${(res.body.bag.reconstruction_cos || 0).toFixed(2)}.`;
        state.light && state.light.pulse(.6);
        loadAnchored(view, state);
      } else {
        const reason = (res.body && res.body.reason) || "fit failed.";
        if (result) result.innerHTML = res.body && res.body.refused
          ? `<span class="refused-note">not anchored: ${esc(reason)}</span>` : esc(reason);
      }
      return;
    }
    const toggleWatlearnedBtn = e.target.closest("[data-whatlearned]");
    if (toggleWatlearnedBtn) { loadWhatlearned(view, state); return; }
    const delTermBtn = e.target.closest("[data-delete-term]");
    if (delTermBtn) {
      if (!armConfirm(delTermBtn)) return;
      const cardId = delTermBtn.dataset.card, token = delTermBtn.dataset.token;
      const res = await postJSON("/memory/anchored/delete_term", { card_id: cardId, token });
      if (res.body && res.body.ok) {
        state.light && state.light.pulse(.55);
        loadAnchored(view, state);
      }
      return;
    }

    /* -- settings: save decode -- */
    const saveSampling = e.target.closest("[data-save-sampling]");
    if (saveSampling) {
      const payload = {
        sampling: view.querySelector("#sampSampling").checked,
        sample_temperature: Number(view.querySelector("#sampTemp").value),
        sample_top_p: Number(view.querySelector("#sampTopP").value),
        sample_top_k: Number(view.querySelector("#sampTopK").value),
        sample_repeat_penalty: Number(view.querySelector("#sampRep").value),
      };
      const res = await postJSON("/sampling/mode", payload);
      const out = view.querySelector("#samplingSaved");
      if (out) { out.hidden = false; out.textContent = res.ok ? "saved." : "could not save."; }
      if (res.ok) state.light && state.light.pulse(.5);
      return;
    }

    /* -- settings: save disposition-guard default -- */
    const saveGuard = e.target.closest("[data-save-guard]");
    if (saveGuard) {
      const enabled = view.querySelector("#guardEnabled").checked;
      const conceptsRaw = (view.querySelector("#guardConcepts").value || "").trim();
      const concepts = conceptsRaw ? conceptsRaw.split(",").map(s => s.trim()).filter(Boolean) : [];
      const payload = { enabled, concepts };
      const counterVal = view.querySelector("#guardCounter").value;
      if (counterVal !== "") payload.counter_strength = Number(counterVal);
      const maxFiresVal = view.querySelector("#guardMaxFires").value;
      if (maxFiresVal !== "") payload.max_fires = Number(maxFiresVal);
      const res = await postJSON("/guard/mode", payload);
      const out = view.querySelector("#guardSaved");
      if (out) {
        out.hidden = false;
        out.textContent = res.ok ? "saved."
          : ((res.body && res.body.error && (res.body.error.message || res.body.error)) || "could not save.");
      }
      if (res.ok) {
        state.guard = res.body;
        state.light && state.light.pulse(.5);
      }
      return;
    }

    /* -- profiles: switch / delete / save -- */
    const switchBtn = e.target.closest("[data-switch-profile]");
    if (switchBtn) {
      const res = await postJSON("/profiles/switch", { name: switchBtn.dataset.switchProfile });
      if (res.ok) {
        state.light && state.light.pulse(.55);
        await Promise.all([loadDials(view, state), loadProfiles(view, state)]);
      }
      return;
    }
    const delProfileBtn = e.target.closest("[data-delete-profile]");
    if (delProfileBtn) {
      if (!armConfirm(delProfileBtn)) return;
      const res = await postJSON("/profiles/delete", { name: delProfileBtn.dataset.deleteProfile });
      if (res.ok) { state.light && state.light.pulse(.5); loadProfiles(view, state); }
      return;
    }
    const saveProfileBtn = e.target.closest("[data-save-profile]");
    if (saveProfileBtn) {
      const name = (view.querySelector("#profName").value || "").trim();
      const description = (view.querySelector("#profDesc").value || "").trim();
      const out = view.querySelector("#profileSaved");
      if (!name) { if (out) { out.hidden = false; out.textContent = "needs a name (lowercase, digits, -, _)."; } return; }
      const dials = {};
      for (const ax of state.axes) dials[ax.name] = ax.value;
      const cardsRes = await postJSON("/memory/cards", {});
      const cards = (cardsRes.ok && cardsRes.body && Array.isArray(cardsRes.body.cards))
        ? cardsRes.body.cards.filter(c => c.status === "active").map(c => ({ text: c.text, status: "active" }))
        : [];
      const res = await postJSON("/profiles/save", { name, description, dials, cards });
      if (out) {
        out.hidden = false;
        out.textContent = res.ok ? `saved "${name}".` : ((res.body && res.body.error) || "could not save profile.");
      }
      if (res.ok) { state.light && state.light.pulse(.55); loadProfiles(view, state); }
      return;
    }
  });
}
