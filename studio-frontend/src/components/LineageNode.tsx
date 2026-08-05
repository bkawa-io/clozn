/**
 * A recursive parent/child run tree.
 *
 * Extracted from `features/runs/Runs.tsx` when the Runs index stopped drawing a lineage canvas. That
 * page reserved a fixed 214px row for this and, on a journal of mostly original runs, spent it
 * rendering a single node labelled "1 RUN" -- height taken directly from the list, which is the only
 * thing that page exists to show. Derivation now rides on the run card itself, where it is one line
 * next to the run it describes (see STUDIO_UI_AUDIT 4.1 / 5.4).
 *
 * The component is kept rather than deleted because a real tree is genuinely the right shape once
 * there IS branching to show -- a session with retries and forks, or a Compare view tracing where a
 * child came from. It is unused today; that is deliberate, not an oversight.
 *
 * Styling lives in `styles/runs.css` under `.runs-tree*`, kept alongside for the same reason.
 */
import type { RunSummary } from "../data/types";

function shortId(id: string) {
  const tail = id.split("_").pop() ?? id;
  return tail.slice(0, 6);
}

export function LineageNode({
  run,
  byParent,
  selectedId,
  onSelect,
}: {
  run: RunSummary;
  /** run id -> its direct children. Built by the caller so one pass covers the whole family. */
  byParent: Map<string, RunSummary[]>;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const children = byParent.get(run.id) ?? [];
  const source = (run.source || "").trim();
  return (
    <div className="runs-tree-node">
      <button
        type="button"
        className={run.id === selectedId ? "is-selected" : ""}
        onClick={() => onSelect(run.id)}
        aria-current={run.id === selectedId ? "true" : undefined}
      >
        <i />
        <span>{shortId(run.id)}</span>
        <small>{source ? source.toUpperCase() : "ENTRY NOT RECORDED"}</small>
      </button>
      {children.length > 0 && (
        <div className="runs-tree-children">
          {children.map((child) => (
            <LineageNode
              key={child.id}
              run={child}
              byParent={byParent}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}
