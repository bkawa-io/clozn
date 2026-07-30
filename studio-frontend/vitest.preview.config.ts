import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * A SEPARATE vitest config for the surface-preview generator only (`scripts/preview/`).
 *
 * Deliberately not folded into `vitest.config.ts`: that config's `include` glob
 * (`src/**\/*.test.{ts,tsx}`) is what `npm test`/CI run, and its result (76 passed / 13 files at the
 * time this generator was added) is a gate other tooling asserts on. Adding the preview-capture spec to
 * that glob would change the count and couple an unrelated dev tool to the product test gate. This
 * config's `include` only ever matches files under `scripts/preview/`, so `npx vitest run` (no args,
 * the CI/gate invocation) never discovers or runs it -- it only runs via
 * `node scripts/generate-preview.mjs`, which points at this config explicitly.
 */
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["scripts/preview/**/*.spec.tsx"],
    clearMocks: true,
    restoreMocks: true,
    // One worker: the capture spec accumulates captured fragments into a module-level array and writes
    // them once at the end. Parallel workers would each hold their own copy and the last one to finish
    // would silently win, dropping every fragment captured by the others.
    fileParallelism: false,
  },
});
