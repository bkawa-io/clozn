/**
 * Render every Studio surface to HTML in Node and assert each one actually mounts.
 *
 * WHY THIS EXISTS
 * ---------------
 * `tsc` proves the app compiles and `vite build` proves it bundles. Neither proves React mounts: a
 * panel that throws on first render, a registry that discovers nothing, a route that resolves to the
 * wrong surface, and a hook that loops are all green under both. Until this script the only way to know
 * was to open a browser, which is exactly the verification gap that made the panel-seam work risky.
 *
 * This is not a substitute for looking at it. It says nothing about layout, CSS, theming, or anything
 * that happens after the first paint -- effects do not run under `renderToString`, so data loading and
 * `useTopbar` publication are NOT exercised. What it does prove is that the module graph resolves, the
 * panel registry builds, every route resolves to its intended panel, and every panel's component tree
 * renders once without throwing. That is most of what a router rewrite can break.
 *
 * HOW
 * ---
 * Vite builds this file in SSR mode (see `smoke-render.build.mjs` invocation in package.json), which
 * transpiles the TSX and resolves the `import.meta.glob` panel discovery exactly as the browser bundle
 * does -- so the panel set under test is the real one, not a hand-listed copy that could drift.
 */
// Side-effect import FIRST: installs the browser globals App.tsx's module body and first render need.
// ES imports execute in source order, so this must stay above the React/App imports.
import "./ssr-globals.mjs";

import { renderToString } from "react-dom/server";
import { createElement } from "react";

import { App } from "../src/app/App";
import { slotPanelsFor } from "../src/components/SlotHost";
import { panelRegistry, resolveRoute } from "../src/panels/registry";

// Routes to exercise, and the panel id each MUST resolve to. Hardcoded on purpose: this is the
// assertion, so deriving it from the registry would make it vacuous.
const ROUTES = [
  ["", "runs"],
  ["#/runs", "runs"],
  ["#/lens", "lens"],
  ["#/runs/run_abc123", "lens"],
  ["#/runs/run_abc123/lens", "lens"],
  ["#/diagnostics", "diagnostics"],
  ["#/runs/run_abc123/diagnostics", "diagnostics"],
  ["#/runs/run_abc123/diagnostics/influence", "diagnostics"],
  // The Investigate section hosts the whole `lens.evidence` slot. Rendering it here is the guard that
  // caught nothing last time: the slot's only mount was deleted with the old Lens page and every panel
  // under it went unreachable while the component tests -- which mount each panel directly -- stayed green.
  ["#/runs/run_abc123/diagnostics/investigate", "diagnostics"],
  ["#/runs/run_abc123/diagnostics/timemachine", "diagnostics"],
  ["#/scope", "diagnostics"],
  ["#/runs/run_abc123/scope", "diagnostics"],
  ["#/runs/run_abc123/scope?token=7", "diagnostics"],
  ["#/compare", "compare"],
  ["#/compare/run_a/run_b", "compare"],
  ["#/behavior", "behavior"],
  ["#/model", "model"],
  ["#/experiments", "experiments"],
  ["#/experiments/exp_abc123", "experiments"],
  ["#/experiments/exp_abc123?suite=target&status=fail&cell=target%3A%3Agreet%3A%3Acandidate%3A%3A0", "experiments"],
  ["#/definitely-not-a-route", "runs"],   // unknown hash falls back to the first surface
];

const failures = [];
function check(label, condition, detail = "") {
  if (condition) return;
  failures.push(`${label}${detail ? `: ${detail}` : ""}`);
}

// --- the registry itself -----------------------------------------------------------------------
check("registry discovered panels", panelRegistry.panels.length > 0);
check("registry has no load failures", panelRegistry.loadFailures.length === 0,
      JSON.stringify(panelRegistry.loadFailures));

const ids = panelRegistry.panels.map((p) => p.id);
for (const expected of ["runs", "lens", "diagnostics", "scope", "compare", "behavior", "model", "experiments"]) {
  check(`panel "${expected}" is registered`, ids.includes(expected), `found: ${ids.join(", ")}`);
}
check("panel ids are unique", new Set(ids).size === ids.length, ids.join(", "));

// --- routing -----------------------------------------------------------------------------------
for (const [hash, expectedId] of ROUTES) {
  const resolved = resolveRoute(panelRegistry.panels, hash);
  check(`route "${hash}" resolves`, resolved != null);
  if (resolved) {
    check(`route "${hash}" -> ${expectedId}`, resolved.panel.id === expectedId,
          `got ${resolved.panel.id}`);
  }
}

// deep-link params survive matching
const scopeDeep = resolveRoute(panelRegistry.panels, "#/runs/run_abc123/scope?token=7");
check("scope deep link keeps runId", scopeDeep?.params.runId === "run_abc123",
      JSON.stringify(scopeDeep?.params));
check("retired scope deep link opens generation diagnostics", scopeDeep?.params.view === "generation",
      JSON.stringify(scopeDeep?.params));
const diagnosticsDeep = resolveRoute(panelRegistry.panels, "#/runs/run_abc123/diagnostics/influence");
check("diagnostics deep link keeps run and section",
      diagnosticsDeep?.params.runId === "run_abc123" && diagnosticsDeep?.params.view === "influence",
      JSON.stringify(diagnosticsDeep?.params));
const compareDeep = resolveRoute(panelRegistry.panels, "#/compare/run_a/run_b");
check("compare deep link keeps both ids",
      compareDeep?.params.runA === "run_a" && compareDeep?.params.runB === "run_b",
      JSON.stringify(compareDeep?.params));

// --- every surface renders ---------------------------------------------------------------------
for (const [hash, expectedId] of ROUTES) {
  location.hash = hash;
  let html = "";
  try {
    html = renderToString(createElement(App));
  } catch (error) {
    check(`render "${hash}"`, false, `${error?.name}: ${error?.message}`);
    continue;
  }
  check(`render "${hash}" produced markup`, html.length > 0);
  check(`render "${hash}" mounted the ${expectedId} workspace`, html.includes(`is-${expectedId}`),
        "workspace class missing from output");
  check(`render "${hash}" drew the nav rail`, html.includes("rail-nav"));
  check(`render "${hash}" has no failed-panel placeholder`, !html.includes("is-failed"));
}

// --- every slot panel is actually reachable ------------------------------------------------------
// The check the suite was missing. Component tests mount each slot panel directly, so they cannot tell
// a mounted panel from an orphaned one -- when the old Lens page was replaced, its lone
// `<SlotHost slot="lens.evidence">` went with it and seven panels became unreachable with every test
// still green. SlotHost renders each panel as `<section class="slot-panel" aria-label={title}>`, so
// asserting each registered title appears in real rendered output is what makes orphaning fail loudly.
location.hash = "#/runs/run_abc123/diagnostics/investigate";
let evidenceHtml = "";
try {
  evidenceHtml = renderToString(createElement(App));
} catch (error) {
  check("render lens.evidence host", false, `${error?.name}: ${error?.message}`);
}
const evidencePanels = slotPanelsFor("lens.evidence");
check("lens.evidence slot still registers panels", evidencePanels.length > 0);
for (const panel of evidencePanels) {
  check(`slot panel "${panel.id}" is mounted by a host`,
        evidenceHtml.includes(`aria-label="${panel.title}"`),
        "registered but absent from rendered output -- has its host been removed?");
}

if (failures.length) {
  console.error(`\nStudio smoke render FAILED (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`Studio smoke render passed: ${panelRegistry.panels.length} panels, `
  + `${ROUTES.length} routes, ${evidencePanels.length} slot panels reachable.`);
