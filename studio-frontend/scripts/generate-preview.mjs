#!/usr/bin/env node
/**
 * Entry point for the Studio surface visual preview. Produces ONE self-contained HTML file at
 * `.preview/surfaces.html` showing every covered surface, in every covered state, in both themes,
 * rendered against fixture data with no server, no backend, and no model.
 *
 * See `scripts/preview/capture-surfaces.spec.tsx` for what is captured and why, and `scripts/preview/
 * assemble.ts` for how the fragments become one file. This script only does two mechanical things:
 *
 *   1. Makes sure real, current compiled CSS exists (`studio/next/assets/*.css`) by running `vite build`
 *      if it does not -- the preview's whole point is to use the REAL stylesheet, not a hand-approximated
 *      one, so a stale or absent build would silently produce a wrong preview.
 *   2. Runs the capture spec through vitest's own jsdom environment (`vitest.preview.config.ts`), which
 *      writes the final HTML itself once every state is captured.
 *
 * Usage: `node scripts/generate-preview.mjs` from `studio-frontend/`.
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const studioFrontendRoot = path.resolve(here, "..");
const repoRoot = path.resolve(studioFrontendRoot, "..");
const cssAssetsDir = path.join(repoRoot, "studio", "next", "assets");

const isWin = process.platform === "win32";
const bin = (name) => path.join(studioFrontendRoot, "node_modules", ".bin", isWin ? `${name}.cmd` : name);

function hasCompiledCss() {
  return fs.existsSync(cssAssetsDir) && fs.readdirSync(cssAssetsDir).some((f) => f.endsWith(".css"));
}

if (!hasCompiledCss()) {
  console.log("[generate-preview] no compiled CSS under studio/next/assets/ -- running `vite build` first...");
  execFileSync(bin("vite"), ["build"], { cwd: studioFrontendRoot, stdio: "inherit" });
} else {
  console.log("[generate-preview] using existing studio/next/assets/*.css -- run `npm run build` first if it is stale.");
}

console.log("[generate-preview] capturing surface states (vitest + jsdom, real fetch stubbing, real effects)...");
execFileSync(bin("vitest"), ["run", "--config", "vitest.preview.config.ts"], {
  cwd: studioFrontendRoot,
  stdio: "inherit",
});

console.log("[generate-preview] done -- open studio-frontend/.preview/surfaces.html directly (file:// is fine).");
