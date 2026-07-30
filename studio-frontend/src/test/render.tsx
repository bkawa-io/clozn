import type { ReactNode } from "react";
import {
  render as testingLibraryRender,
  type RenderOptions,
  type RenderResult,
} from "@testing-library/react";

function TestProviders({ children }: { children: ReactNode }) {
  return children;
}

export function render(
  ui: ReactNode,
  options?: Omit<RenderOptions, "wrapper">,
): RenderResult {
  return testingLibraryRender(ui, {
    wrapper: TestProviders,
    ...options,
  });
}

export * from "@testing-library/react";
