import type { RunIdentity } from "./types";

/**
 * A raw field comparison over `clozn/runs/identity.py`'s block -- never a causal claim. Feature 05
 * (`clozn.triage.v1`, the evidence ladder) owns interpreting *why* an identity difference might matter;
 * this only shows *that* one exists, per the roadmap plan's scoping of this drill-down.
 *
 * Walks generically rather than listing known fields, so a future `identity_providers/*.py` facet
 * landing inside `ext` is diffed automatically instead of needing a matching frontend edit.
 */
export interface IdentityField {
  path: string;
  base: string | undefined;
  candidate: string | undefined;
  status: "same" | "differs" | "base-only" | "candidate-only";
}

function flatten(value: unknown, prefix: string, out: Record<string, string>): void {
  if (value == null) return;
  if (typeof value === "object" && !Array.isArray(value)) {
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      flatten(child, prefix ? `${prefix}.${key}` : key, out);
    }
    return;
  }
  out[prefix] = Array.isArray(value) ? JSON.stringify(value) : String(value);
}

export function diffIdentity(base: RunIdentity | undefined, candidate: RunIdentity | undefined): IdentityField[] {
  const baseFlat: Record<string, string> = {};
  const candidateFlat: Record<string, string> = {};
  if (base) flatten(base, "", baseFlat);
  if (candidate) flatten(candidate, "", candidateFlat);

  const paths = [...new Set([...Object.keys(baseFlat), ...Object.keys(candidateFlat)])].sort();
  return paths.map((path) => {
    const b = baseFlat[path];
    const c = candidateFlat[path];
    const status: IdentityField["status"] =
      b !== undefined && c !== undefined ? (b === c ? "same" : "differs")
      : b !== undefined ? "base-only"
      : "candidate-only";
    return { path, base: b, candidate: c, status };
  });
}
