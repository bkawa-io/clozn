/* clozn studio — the rebuilt frontend's shell (successor to studio/heavn, spec:
   notes/UX_INFORMATION_ARCHITECTURE.md). Build 1: tokens + shader core + shell + a live Runs list.
   No framework yet — the shell earns dependencies before it takes them. */
import { mountLight } from "./light.mjs";
import { mountWorkspace } from "./workspace.mjs";

mountWorkspace(document.querySelector(".shell"));

/* ---------- theme: OS preference by default, explicit choice wins, persisted ---------- */
const prefersNight = matchMedia("(prefers-color-scheme: dark)");
function currentNight() {
  const t = localStorage.getItem("clozn.theme");
  return t ? t === "night" : prefersNight.matches;
}
function applyTheme(explicit) {
  const t = localStorage.getItem("clozn.theme");
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  const night = currentNight();
  themeBtn.textContent = night ? "◑ DAWN" : "◐ NIGHT";
  light && light.setNight(night);
}
const themeBtn = document.getElementById("theme");
themeBtn.addEventListener("click", () => {
  localStorage.setItem("clozn.theme", currentNight() ? "dawn" : "night");
  applyTheme(); light && light.pulse(.5);
});
prefersNight.addEventListener("change", () => applyTheme());

/* ---------- ambient light (motion = meaning: idle is a breath, events pulse it) ---------- */
const light = mountLight(document.getElementById("ambient"), { night: currentNight() });
applyTheme();

/* ---------- runtime status ---------- */
const servingEl = document.getElementById("serving");
async function pingRuntime() {
  try {
    const r = await fetch("/healthz");
    servingEl.innerHTML = r.ok
      ? `<span class="dot up"></span><span>runtime up · local · nothing leaves the box</span>`
      : `<span class="dot"></span><span>runtime answered ${r.status}</span>`;
  } catch {
    servingEl.innerHTML = `<span class="dot"></span><span>runtime not reachable — start with <span class="machine">clozn serve</span></span>`;
  }
}
pingRuntime();

/* ---------- tiny hash router ---------- */
const view = document.getElementById("view");
const routes = { "/runs": Runs };
let currentCleanup = null;   // a view may return a teardown fn (e.g. Observatory's casting.destroy()) --
                              // called before every navigation so a WebGL/canvas RAF loop never leaks.
async function route() {
  if (typeof currentCleanup === "function") {
    try { currentCleanup(); } catch { /* a teardown error must never block navigation */ }
  }
  currentCleanup = null;
  const path = location.hash.replace(/^#/, "") || "/runs";
  const runMatch = path.match(/^\/runs\/([A-Za-z0-9_-]+)$/);           // #/runs/<id> -> the Lens page (build 2)
  const runObsMatch = path.match(/^\/runs\/([A-Za-z0-9_-]+)\/observatory$/); // #/runs/<id>/observatory
  const compareMatch = path.match(/^\/compare(?:\/([A-Za-z0-9_-]+)\/([A-Za-z0-9_-]+))?$/); // #/compare[/<a>/<b>]
  document.querySelectorAll("#nav a").forEach(a => {
    // a run detail still reads as "runs" in nav; a run's Observatory reads as "observatory"
    const current = runObsMatch ? "#/observatory" : compareMatch ? "#/compare"
      : runMatch ? "#/runs" : "#" + path;
    a.getAttribute("href") === current
      ? a.setAttribute("aria-current", "page")
      : a.removeAttribute("aria-current");
  });
  light && light.pulse(.4);                               // navigation is a real interaction
  view.innerHTML = `<p class="quiet breathing">…</p>`;
  if (runMatch) {
    const { renderLens } = await import("./lens.mjs");
    await renderLens(view, runMatch[1], light);
    return;
  }
  if (runObsMatch) {
    const { renderObservatory } = await import("./observatory.mjs");
    currentCleanup = await renderObservatory(view, runObsMatch[1], light);
    return;
  }
  if (path === "/observatory") {
    const { renderObservatory } = await import("./observatory.mjs");
    currentCleanup = await renderObservatory(view, null, light);
    return;
  }
  if (compareMatch) {
    const { renderCompare } = await import("./compare.mjs");
    currentCleanup = await renderCompare(view, compareMatch[1] || null, compareMatch[2] || null, light);
    return;
  }
  if (path === "/model") {
    const { renderModel } = await import("./model.mjs");
    await renderModel(view, light);
    return;
  }
  if (path === "/behavior") {
    const { renderBehavior } = await import("./behavior.mjs");
    await renderBehavior(view, light);
    return;
  }
  const fn = routes[path] || Runs;
  view.innerHTML = await fn();
}
addEventListener("hashchange", route);
route();

/* ---------- views (build 1: honest skeletons, real data where routes exist) ---------- */
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function Runs() {
  let runs = null;
  try { const r = await fetch("/runs"); if (r.ok) runs = await r.json(); } catch {}
  const items = Array.isArray(runs) ? runs : (runs && runs.runs) || [];
  const rows = items.slice(0, 50).map(x => {
    const id = esc(x.id || x.run_id || "?");
    const prompt = esc(x.prompt_summary || x.prompt || x.id || "(untitled run)");
    const when = esc(x.created_at || "");
    return `<li class="panel"><a href="#/runs/${id}" title="open the lens">
      <div class="run-prompt">${prompt}</div>
      <div class="run-meta"><span>${id}</span><span>${when}</span></div></a></li>`;
  }).join("");
  return `
    <h1 class="view-title">Runs</h1>
    <div class="view-sub">everything that flowed in — each opens into the lens page</div>
    ${rows ? `<ul class="run-list">${rows}</ul>`
           : `<p class="quiet">no runs reachable — make one through the API and it lands here.</p>`}`;
}

/* Model, Behavior, Observatory and Compare are their own modules (studio/app/model.mjs,
   studio/app/behavior.mjs, studio/app/observatory.mjs, studio/app/compare.mjs), dynamically imported by
   route() above -- same pattern as the Lens page's lens.mjs. Observatory/Compare are the only views that
   return a teardown fn (Observatory's casting.mjs owns a canvas RAF loop that must be destroy()ed on
   navigation away; Compare has none today but returns undefined, which the router already guards). */
