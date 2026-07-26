/* Observatory adapter for the shared Clozn workspace.
   It consumes the already-assembled cast contract and never fabricates fields. */

const esc = value => String(value ?? "").replace(/[&<>\"]/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
}[char]));

const finite = value => Number.isFinite(Number(value));
const fmt = (value, digits = 3) => finite(value) ? Number(value).toFixed(digits) : "not recorded";

function noOpController() {
  return {
    updateCast() {},
    selectToken() {},
    clear() {},
    destroy() {},
    get selectedIndex() { return -1; },
  };
}

function tokensFrom(cast) {
  return cast && Array.isArray(cast.tokens) ? cast.tokens : [];
}

function tokenText(token) {
  return token?.piece ?? token?.text ?? token?.token ?? "";
}

function tokenEntropy(token, cast, index) {
  if (finite(token?.entropy)) return Number(token.entropy);
  if (Array.isArray(cast?.entropy) && finite(cast.entropy[index])) return Number(cast.entropy[index]);
  if (Array.isArray(cast?.tokenEntropy) && finite(cast.tokenEntropy[index])) return Number(cast.tokenEntropy[index]);
  return null;
}

function tokenSources(token) {
  if (!Array.isArray(token?.sources)) return [];
  return token.sources.filter(Boolean);
}

function tokenAlternatives(token) {
  const candidates = token?.alternatives ?? token?.almosts ?? [];
  return Array.isArray(candidates) ? candidates.filter(Boolean) : [];
}

function sourceLabel(source) {
  if (typeof source === "string") return source;
  return source?.text ?? source?.piece ?? source?.label ?? source?.excerpt ?? "source";
}

function alternativeLabel(alternative) {
  if (typeof alternative === "string") return alternative;
  return alternative?.piece ?? alternative?.text ?? alternative?.token ?? "alternative";
}

function renderInspector(cast, index, { isDemo }) {
  const tokens = tokensFrom(cast);
  const token = tokens[index];
  if (!token) {
    return `<div class="obs-inspector-empty">Select a token in the Casting or event strip.</div>`;
  }

  const entropy = tokenEntropy(token, cast, index);
  const sources = tokenSources(token);
  const alternatives = tokenAlternatives(token);
  const commitLayer = token.commitLayer ?? token.commit_layer ?? cast?.commitLayer ?? cast?.commit_layer;
  const rank = token.rank ?? token.top1Rank ?? null;
  const confidence = token.confidence ?? token.probability ?? null;

  const sourceHTML = sources.length
    ? `<ul class="obs-inspector-list">${sources.map(source => `<li>${esc(sourceLabel(source))}</li>`).join("")}</ul>`
    : `<p class="obs-inspector-quiet">${token.fromWeights === true
        ? "The verified cast marks this token as arising from model weights rather than prompt words."
        : "Provenance is not available for this token in this cast."}</p>`;

  const alternativesHTML = alternatives.length
    ? `<ul class="obs-inspector-list obs-inspector-alts">${alternatives.map(alt => `<li>${esc(alternativeLabel(alt))}</li>`).join("")}</ul>`
    : `<p class="obs-inspector-quiet">No recorded alternatives.</p>`;

  return `
    <article class="obs-token-inspector" aria-live="polite">
      <header class="obs-token-title">
        <span class="obs-token-index">token ${index}</span>
        <strong>${esc(tokenText(token) || "∅")}</strong>
        ${isDemo ? `<span class="dial-tag dial-tag-warn">scripted demo</span>` : `<span class="dial-tag dial-tag-calib">recorded run</span>`}
      </header>

      <section>
        <h3>Signal</h3>
        <dl class="obs-inspector-kv">
          <div><dt>entropy</dt><dd>${entropy == null ? "not recorded" : `${fmt(entropy)} bits, labeled approximation`}</dd></div>
          <div><dt>commit layer</dt><dd>${commitLayer == null ? "not recorded" : `L${esc(commitLayer)}`}</dd></div>
          <div><dt>rank</dt><dd>${rank == null ? "not recorded" : esc(rank)}</dd></div>
          <div><dt>confidence</dt><dd>${confidence == null ? "not recorded" : fmt(confidence)}</dd></div>
        </dl>
      </section>

      <section>
        <h3>Verified provenance</h3>
        ${sourceHTML}
      </section>

      <section>
        <h3>Recorded alternatives</h3>
        ${alternativesHTML}
      </section>
    </article>`;
}

function makeTimeline(cast, selectedIndex, selectToken, { isDemo }) {
  const tokens = tokensFrom(cast);
  const root = document.createElement("div");
  root.className = "obs-event-strip-wrap";

  const meta = document.createElement("div");
  meta.className = "obs-event-strip-meta";
  meta.innerHTML = `<span>${tokens.length} token${tokens.length === 1 ? "" : "s"}</span><span>${isDemo ? "scripted demo" : "recorded order"}</span>`;

  const strip = document.createElement("div");
  strip.className = "obs-event-strip";
  strip.setAttribute("role", "listbox");
  strip.setAttribute("aria-label", "Casting token events");
  strip.tabIndex = 0;

  tokens.forEach((token, index) => {
    const entropy = tokenEntropy(token, cast, index);
    const option = document.createElement("button");
    option.type = "button";
    option.className = "obs-event-token";
    option.dataset.index = String(index);
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", index === selectedIndex ? "true" : "false");
    if (index === selectedIndex) option.setAttribute("aria-current", "true");
    option.title = `Token ${index}: ${tokenText(token) || "empty"}${entropy == null ? "" : ` · entropy ${fmt(entropy)} bits`}`;
    option.innerHTML = `<span class="obs-event-n">${index}</span><span class="obs-event-piece">${esc(tokenText(token) || "∅")}</span>`;
    if (entropy != null) {
      option.style.setProperty("--obs-event-signal", String(Math.max(0, Math.min(1, entropy / 8))));
    }
    option.addEventListener("click", () => selectToken(index, { source: "timeline" }));
    strip.append(option);
  });

  strip.addEventListener("keydown", event => {
    if (!tokens.length) return;
    let next = selectedIndex < 0 ? 0 : selectedIndex;
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") next -= 1;
    else if (event.key === "ArrowRight" || event.key === "ArrowDown") next += 1;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tokens.length - 1;
    else return;
    event.preventDefault();
    selectToken(Math.max(0, Math.min(tokens.length - 1, next)), { source: "keyboard" });
  });

  root.append(meta, strip);
  return root;
}

export function mountObservatoryWorkspace({ workspace = window.cloznWorkspace, casting = null } = {}) {
  if (!workspace?.setInspector || !workspace?.setTimeline) return noOpController();

  let cast = null;
  let selectedIndex = -1;
  let isDemo = false;
  let destroyed = false;

  function publishSelection(source) {
    window.dispatchEvent(new CustomEvent("clozn:observatory-selection", {
      detail: { tokenIndex: selectedIndex, source, token: tokensFrom(cast)[selectedIndex] ?? null },
    }));
  }

  function render() {
    if (destroyed) return;
    workspace.setInspector(renderInspector(cast, selectedIndex, { isDemo }));
    workspace.setTimeline(makeTimeline(cast, selectedIndex, selectToken, { isDemo }));
  }

  function selectToken(index, { source = "route" } = {}) {
    const tokens = tokensFrom(cast);
    if (!tokens.length) {
      selectedIndex = -1;
      render();
      return;
    }
    selectedIndex = Math.max(0, Math.min(tokens.length - 1, Number(index) || 0));
    if (source !== "casting" && typeof casting?.selectToken === "function") {
      casting.selectToken(selectedIndex);
    }
    render();
    publishSelection(source);
  }

  const controller = {
    updateCast(nextCast, options = {}) {
      cast = nextCast ?? null;
      isDemo = Boolean(options.isDemo);
      const count = tokensFrom(cast).length;
      selectedIndex = count ? Math.max(0, Math.min(selectedIndex < 0 ? 0 : selectedIndex, count - 1)) : -1;
      render();
    },
    selectToken,
    clear() {
      cast = null;
      selectedIndex = -1;
      render();
    },
    destroy() {
      destroyed = true;
      workspace.setInspector(`<div class="workspace-empty">Select evidence in the viewport<br>to inspect its recorded values.</div>`);
      workspace.setTimeline(`<div class="workspace-empty">A route may mount ordered inference events here.</div>`);
    },
    get selectedIndex() { return selectedIndex; },
  };

  render();
  return controller;
}
