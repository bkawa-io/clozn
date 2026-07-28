import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

/**
 * Seam B: additive sub-panels inside an existing surface.
 *
 * Seam A (`src/panels/`) is for whole new top-level destinations. This is for the commoner case --
 * adding a card to a page someone else owns, which several features need: a context-receipt view inside
 * Lens, a performance timeline inside Scope. Without a seam each of those is an edit to the host page,
 * which puts every feature back in the same file.
 *
 * A host renders `<SlotHost slot="lens.evidence" data={...} />`. Anything under
 * `src/slots/<slot>/*.tsx` with a matching `slot` field appears there, ordered by `order`.
 *
 * The error boundary is the point. A slot panel is written by someone who does not own the host page,
 * so a throw in one card must cost that card and nothing else -- otherwise adding a sub-panel is a way
 * to break a surface you do not own, and hosts would rightly refuse to expose slots at all.
 *
 * Slot names and their `data` shapes are a contract the HOST documents in `docs/SURFACES.md`. This
 * component owns the mechanism, never the vocabulary.
 */
export interface SlotPanel<TData = unknown> {
  id: string;
  /** Host-defined slot name, e.g. "lens.evidence". Documented by the host in docs/SURFACES.md. */
  slot: string;
  title: string;
  /** Render order within the slot; default 100, ties broken by id. */
  order?: number;
  Component: React.ComponentType<{ data: TData }>;
}

interface BoundaryProps {
  id: string;
  children: ReactNode;
}

interface BoundaryState {
  error: Error | null;
}

class SlotBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): BoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Loud in the console, contained in the UI -- never silently blank (roadmap rule 3).
    console.error(`slot panel "${this.props.id}" failed to render:`, error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="slot-panel is-failed" role="note">
          <b>{this.props.id}</b> failed to render: {this.state.error.message}
        </div>
      );
    }
    return this.props.children;
  }
}

const modules = import.meta.glob("../slots/**/*.tsx", { eager: true });

/** Every registered slot panel, keyed by slot name. Built once at module load. */
export function slotPanelsFor(slot: string): SlotPanel[] {
  const found: SlotPanel[] = [];
  for (const path of Object.keys(modules).sort()) {
    const candidate = (modules[path] as { default?: unknown } | undefined)?.default as
      | SlotPanel
      | undefined;
    if (!candidate || typeof candidate !== "object") continue;
    if (candidate.slot !== slot) continue;
    if (typeof candidate.id !== "string" || candidate.Component == null) continue;
    found.push(candidate);
  }
  found.sort((a, b) => (a.order ?? 100) - (b.order ?? 100) || a.id.localeCompare(b.id));
  return found;
}

export function SlotHost<TData>({ slot, data }: { slot: string; data: TData }) {
  const panels = slotPanelsFor(slot);
  if (!panels.length) return null;
  return (
    <>
      {panels.map((panel) => {
        const Body = panel.Component as React.ComponentType<{ data: TData }>;
        return (
          <SlotBoundary id={panel.id} key={panel.id}>
            <section className="slot-panel" aria-label={panel.title}>
              <Body data={data} />
            </section>
          </SlotBoundary>
        );
      })}
    </>
  );
}
