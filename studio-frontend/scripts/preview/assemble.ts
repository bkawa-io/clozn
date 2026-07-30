/**
 * Pure HTML assembly for the surface-preview generator. No React, no jsdom, no DOM here -- this module
 * takes already-captured HTML fragments (see `capture-surfaces.spec.tsx`) and the real compiled CSS text
 * (see `generate-preview.mjs`) and produces ONE self-contained HTML string.
 *
 * WHY TWO IFRAMES PER STATE, NOT A CSS CLASS SWAP
 * -------------------------------------------------
 * The product's dark theme is `:root[data-theme="cathedral"]` (src/styles/tokens.css) -- a selector that
 * only ever matches the actual document root element. Reusing the real, unmodified compiled CSS (a hard
 * requirement here) for a light/dark comparison therefore needs two real documents, each with its own
 * `<html data-theme="...">` root -- i.e. two `<iframe srcdoc="...">` per state, not one DOM with a class
 * swapped on some wrapper div (that would need rewriting `:root[...]` to a class selector, which starts
 * being "approximated" CSS instead of the real thing). srcdoc iframes are inline content: no network
 * request, still one self-contained file.
 *
 * WHY THE CSS TEXT IS BASE64, AND INJECTED BY SCRIPT RATHER THAN INLINED PER IFRAME
 * ------------------------------------------------------------------------------------
 * The compiled stylesheet is applied identically to every iframe. Writing it out literally inside every
 * `srcdoc` would multiply its size by the state count (dozens of times) for zero benefit. Instead it is
 * embedded ONCE, base64-encoded (so no possible content of the CSS -- e.g. a literal "</script" substring
 * -- could ever prematurely terminate the holding <script> tag; base64's alphabet contains no "<"), and a
 * small bootstrap script decodes it once and clones a <style> into each iframe's own document after it
 * loads. This keeps the file's on-disk size proportional to (surfaces x states) + CSS, not
 * (surfaces x states x CSS).
 */

export type FragmentWrap = "instrument" | "bare" | "shell";

export interface CapturedState {
  surfaceId: string;
  surfaceTitle: string;
  surfaceSource: string;
  stateId: string;
  stateTitle: string;
  note: string;
  html: string;
  wrap: FragmentWrap;
  heightPx: number;
}

export interface AssembleInput {
  states: CapturedState[];
  cssText: string;
  cssFiles: string[];
  generatedAtIso: string;
  skippedNote: string;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeAttr(value: string): string {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function wrapFragment(html: string, wrap: FragmentWrap): string {
  if (wrap === "instrument") {
    // `.instrument` is the REAL card chrome class (src/styles/workspace.css) these four lens.evidence
    // surfaces normally inherit by living inside Lens.tsx's own `.lens-context-body` scroll column --
    // see the generator's report for why that ancestor is not reproduced here. `.preview-mount` /
    // `.preview-card-scroll` are this generator's OWN scaffold classes (always `preview-`-prefixed, see
    // the stylesheet below), used only to give the card a margin and a scroll boundary.
    return `<div class="preview-mount"><div class="instrument preview-card-scroll">${html}</div></div>`;
  }
  if (wrap === "shell") {
    return `<div class="preview-mount preview-mount-shell">${html}</div>`;
  }
  // "bare": the component already renders its own `.instrument`-classed root section(s) (
  // ConversationInvestigation, SessionPicker) or its own fully self-boxed root (ForkOutcomePanel's
  // `.fork-outcome`). `.preview-stack` is scaffold-only spacing between siblings.
  return `<div class="preview-mount preview-stack">${html}</div>`;
}

/** One PREVIEW SCAFFOLD OVERRIDE, deliberately isolated and commented so it is never mistaken for
 * product CSS: the real `body { overflow: hidden }` (src/styles/base.css) assumes the SPA fills the
 * viewport and does its OWN internal scrolling in named containers (e.g. `.lens-context-body`). A
 * standalone fragment has no such container, so without this override any state taller than the iframe
 * would be silently clipped -- content missing with no scrollbar to reveal it, exactly the kind of
 * invisible failure this whole tool exists to avoid introducing. */
const OVERRIDE_CSS = `
/* ============================================================================
   PREVIEW SCAFFOLD OVERRIDE -- NOT PRODUCT CSS.
   The real body{overflow:hidden} assumes the SPA fills the viewport and scrolls
   internally; this standalone fragment has no such shell, so restore normal
   document scrolling here, inside this iframe only.
   ============================================================================ */
html, body { overflow: auto !important; height: auto !important; min-height: 0 !important; }
`;

function iframeSrcdoc(fragmentHtml: string, theme: "halo" | "cathedral", wrap: FragmentWrap): string {
  const inner = `<!doctype html><html data-theme="${theme}"><head><meta charset="utf-8"></head>`
    + `<body>${wrapFragment(fragmentHtml, wrap)}</body></html>`;
  return escapeAttr(inner);
}

const SCAFFOLD_CSS = `
/* ============================================================================================
   PREVIEW SCAFFOLD CSS -- hand-written by the generator, NOT compiled product CSS. Every class
   name here is "preview-"-prefixed on purpose: the product never uses that prefix, so anything
   with it is this tool's own chrome, never a Studio surface. Real product CSS lives only inside
   the iframes below, injected verbatim from the actual "npm run build" output.
   ============================================================================================ */
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 0 64px;
  background: #14151a;
  color: #e8e8ec;
  font: 15px/1.5 -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: light) {
  body { background: #eef0f4; color: #1a1c22; }
}
a { color: #7faef5; }
code, .preview-mono { font-family: "SFMono-Regular", "Cascadia Code", Consolas, monospace; }

.preview-banner {
  position: sticky; top: 0; z-index: 50;
  padding: 14px 20px;
  background: #4a1414;
  border-bottom: 3px solid #e0555a;
  color: #ffecec;
}
.preview-banner strong { display: block; font-size: 15px; letter-spacing: .02em; margin-bottom: 6px; }
.preview-banner p { margin: 4px 0; font-size: 13px; max-width: 90ch; color: #ffd9d9; }
.preview-banner .preview-meta { margin-top: 8px; font-size: 12px; color: #ffb8b8; }

.preview-toc {
  margin: 20px auto 0; padding: 14px 20px; max-width: 1100px;
  border: 1px solid #333744; border-radius: 6px; background: #1b1d24;
}
@media (prefers-color-scheme: light) { .preview-toc { background: #fff; border-color: #d6d9e2; } }
.preview-toc strong { display: block; margin-bottom: 8px; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; opacity: .7; }
.preview-toc ul { columns: 3; column-gap: 24px; margin: 0; padding: 0; list-style: none; }
.preview-toc li { break-inside: avoid; margin-bottom: 4px; font-size: 13px; }
.preview-toc a { text-decoration: none; }
.preview-toc a:hover { text-decoration: underline; }
.preview-toc .preview-toc-surface { font-weight: 600; margin-top: 8px; display: block; }

main { max-width: 1100px; margin: 0 auto; padding: 0 20px; }

.preview-surface { margin-top: 40px; padding-top: 16px; border-top: 1px solid #333744; }
@media (prefers-color-scheme: light) { .preview-surface { border-color: #d6d9e2; } }
.preview-surface h2 { font-size: 20px; margin: 0 0 2px; }
.preview-surface-source { font-size: 12px; opacity: .6; margin: 0 0 18px; }

.preview-state { margin: 22px 0 34px; }
.preview-state h3 { font-size: 14px; margin: 0 0 4px; letter-spacing: .01em; }
.preview-note { font-size: 12.5px; opacity: .78; margin: 0 0 10px; max-width: 85ch; }

.preview-theme-pair {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
}
@media (max-width: 820px) {
  .preview-theme-pair { grid-template-columns: 1fr; }
}
.preview-theme-col { min-width: 0; }
.preview-theme-label {
  display: inline-block; margin-bottom: 6px; padding: 2px 8px;
  font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
  border-radius: 3px; background: #2a2d38; color: #b7bccb;
}
@media (prefers-color-scheme: light) { .preview-theme-label { background: #e4e7ef; color: #4b4f5e; } }
.preview-frame {
  display: block; width: 100%; border: 1px solid #333744; border-radius: 6px;
  background: #0c0d10; resize: vertical; overflow: auto;
}
@media (prefers-color-scheme: light) { .preview-frame { border-color: #d6d9e2; background: #fff; } }

.preview-skip {
  margin: 30px auto 0; padding: 14px 20px; max-width: 1100px;
  border: 1px dashed #6b5220; border-radius: 6px; background: #2a2210; color: #f0d9a0;
  font-size: 13px; white-space: pre-wrap;
}
`;

export function assemblePreviewHtml(input: AssembleInput): string {
  const { states, cssText, cssFiles, generatedAtIso, skippedNote } = input;

  for (const marker of ["</style", "</script", "</textarea"]) {
    if (cssText.toLowerCase().includes(marker.toLowerCase())) {
      throw new Error(
        `compiled CSS contains a literal "${marker}" -- this would break the base64 no-escaping `
        + `assumption in assemble.ts if we ever stopped base64-encoding it. Refusing to generate.`,
      );
    }
  }
  const cssB64 = Buffer.from(cssText, "utf-8").toString("base64");

  const bySurface = new Map<string, CapturedState[]>();
  for (const state of states) {
    const list = bySurface.get(state.surfaceId) ?? [];
    list.push(state);
    bySurface.set(state.surfaceId, list);
  }

  const tocItems: string[] = [];
  const sections: string[] = [];
  for (const [surfaceId, list] of bySurface) {
    const surfaceTitle = list[0].surfaceTitle;
    const surfaceSource = list[0].surfaceSource;
    tocItems.push(`<li><a class="preview-toc-surface" href="#surface-${slug(surfaceId)}">${escapeHtml(surfaceTitle)}</a></li>`);
    const stateBlocks: string[] = [];
    for (const state of list) {
      const anchor = `state-${slug(surfaceId)}-${slug(state.stateId)}`;
      tocItems.push(`<li><a href="#${anchor}">&nbsp;&nbsp;${escapeHtml(state.stateTitle)}</a></li>`);
      stateBlocks.push(`
        <article class="preview-state" id="${anchor}">
          <h3>${escapeHtml(state.stateTitle)}</h3>
          ${state.note ? `<p class="preview-note">${escapeHtml(state.note)}</p>` : ""}
          <div class="preview-theme-pair">
            <div class="preview-theme-col">
              <span class="preview-theme-label">Light &middot; halo</span>
              <iframe class="preview-frame" data-preview-frame loading="lazy"
                style="height:${state.heightPx}px"
                title="${escapeAttr(`${surfaceTitle} — ${state.stateTitle} — light theme`)}"
                srcdoc="${iframeSrcdoc(state.html, "halo", state.wrap)}"></iframe>
            </div>
            <div class="preview-theme-col">
              <span class="preview-theme-label">Dark &middot; cathedral</span>
              <iframe class="preview-frame" data-preview-frame loading="lazy"
                style="height:${state.heightPx}px"
                title="${escapeAttr(`${surfaceTitle} — ${state.stateTitle} — dark theme`)}"
                srcdoc="${iframeSrcdoc(state.html, "cathedral", state.wrap)}"></iframe>
            </div>
          </div>
        </article>`);
    }
    sections.push(`
      <section class="preview-surface" id="surface-${slug(surfaceId)}">
        <h2>${escapeHtml(surfaceTitle)}</h2>
        <p class="preview-surface-source"><code>${escapeHtml(surfaceSource)}</code></p>
        ${stateBlocks.join("\n")}
      </section>`);
  }

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Studio surface preview — FIXTURE DATA, not live evidence</title>
<style>${SCAFFOLD_CSS}</style>
</head>
<body>
<header class="preview-banner">
  <strong>FIXTURE DATA. STATIC RENDER. NOT LIVE EVIDENCE.</strong>
  <p>Every screen below is a real Studio React component, mounted once against hand-built fixture wire
  data (copied from that component's own test file wherever one exists), with <code>fetch</code> stubbed
  to answer with that fixture. Effects run and settle once, at generation time — this is NOT the
  <code>renderToString</code> SSR smoke check, which never runs effects at all. But nothing here is
  interactive: no button click after this page loaded does anything, there is no backend, no model, and
  no guarantee the fixtures resemble what real evidence looks like at scale (a real session with 4,000
  turns, a claim list with 300 rows). This page proves what these surfaces look like in these exact
  states with this exact fixture shape — nothing about whether they work end-to-end against the real
  gateway. Open the app itself for that.</p>
  <p class="preview-meta">Generated ${escapeHtml(generatedAtIso)} by
  <code>studio-frontend/scripts/generate-preview.mjs</code>. CSS below is the real, unmodified
  <code>npm run build</code> output (${(cssText.length / 1024).toFixed(1)} KB from
  ${cssFiles.map((f) => escapeHtml(f)).join(", ") || "(none found)"}) — not approximated, not
  hand-written. This tool's own scaffolding (banner, table of contents, two-column theme layout, iframe
  borders) is hand-written CSS namespaced under <code>preview-*</code> classes so it is never mistaken
  for product styling.</p>
</header>

<nav class="preview-toc">
  <strong>Contents</strong>
  <ul>${tocItems.join("")}</ul>
</nav>

<main>
${sections.join("\n")}
</main>

${skippedNote ? `<div class="preview-skip">${escapeHtml(skippedNote)}</div>` : ""}

<script id="app-css-b64" type="application/octet-stream">${cssB64}</script>
<script>
(function () {
  var holder = document.getElementById("app-css-b64");
  var b64 = holder.textContent.trim();
  var bytes = Uint8Array.from(atob(b64), function (c) { return c.charCodeAt(0); });
  var cssText = new TextDecoder("utf-8").decode(bytes);
  var fullCss = cssText + ${JSON.stringify(OVERRIDE_CSS)};
  var frames = document.querySelectorAll("iframe[data-preview-frame]");
  frames.forEach(function (frame) {
    frame.addEventListener("load", function () {
      try {
        var doc = frame.contentDocument;
        var style = doc.createElement("style");
        style.textContent = fullCss;
        doc.head.appendChild(style);
      } catch (err) {
        // Same-origin srcdoc access should never throw; if a browser disagrees, the iframe is left
        // showing unstyled markup rather than nothing.
      }
    });
  });
})();
</script>
</body>
</html>
`;
}
