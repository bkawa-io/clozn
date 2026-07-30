import { describe, expect, test } from "vitest";
import {
  parseScopeUrl,
  scopeRouteParams,
  scopeStateFromParams,
  serializeScopeUrl,
} from "./urlState";

describe("Scope URL state", () => {
  test("parses the full contract in any query order and serializes one canonical order", () => {
    const route = parseScopeUrl(
      "#/runs/run%2Fcurrent/scope?layer=11&reference=run%2Fbefore&token=7&view=layers",
    );

    expect(route).toEqual({
      runId: "run/current",
      state: {
        view: "layers",
        token: 7,
        reference: "run/before",
        layer: 11,
      },
    });
    expect(serializeScopeUrl(route!.runId, {
      view: "layers",
      token: 7,
      reference: "run/before",
      layer: 11,
    })).toBe(
      "#/runs/run%2Fcurrent/scope?view=layers&token=7&reference=run%2Fbefore&layer=11",
    );
  });

  test("preserves token-only deep links and the legacy tokenIndex panel param", () => {
    const route = parseScopeUrl("#/runs/run_1/scope?token=7");
    expect(route).toEqual({
      runId: "run_1",
      state: { view: undefined, token: 7, reference: undefined, layer: undefined },
    });

    const params = scopeRouteParams(route!);
    expect(params).toMatchObject({ runId: "run_1", token: "7", tokenIndex: "7" });
    expect(scopeStateFromParams(params)).toEqual({
      view: undefined,
      token: 7,
      reference: undefined,
      layer: undefined,
    });
  });

  test.each([
    "#/runs/run_1/scope?view=unknown&token=-1&layer=2.5",
    "#/runs/run_1/scope?view=&token=NaN&layer=",
    "#/runs/run_1/scope?view=TRACE&token=9007199254740992&layer=-4",
  ])("fails closed on invalid enum and numeric values: %s", (hash) => {
    expect(parseScopeUrl(hash)).toEqual({
      runId: "run_1",
      state: {
        view: undefined,
        token: undefined,
        reference: undefined,
        layer: undefined,
      },
    });
  });

  test("rejects malformed or empty run identities", () => {
    expect(parseScopeUrl("#/runs/%E0%A4%A/scope?token=1")).toBeNull();
    expect(parseScopeUrl("#/runs//scope?token=1")).toBeNull();
    expect(parseScopeUrl("#/runs/run_1")).toBeNull();
  });
});
