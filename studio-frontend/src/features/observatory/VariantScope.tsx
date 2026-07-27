import { Fragment, useMemo, useState, type CSSProperties } from "react";
import type { ObservatoryData, RunSummary } from "../../data/types";
import type { TokenAlignment } from "../compare/alignment";
import type { VariantRelation } from "./variant";

interface VariantScopeProps {
  current: ObservatoryData;
  reference: ObservatoryData | null;
  referenceId: string;
  referenceStatus: "idle" | "loading" | "error";
  referenceOptions: RunSummary[];
  alignment: TokenAlignment;
  relation?: VariantRelation;
  selectedToken: number;
  onSelectToken: (index: number) => void;
  onSelectReference: (runId: string) => void;
}

type OriginFilter = "reference" | "variant" | null;

function visibleToken(text: string) {
  if (!text) return "∅";
  if (!text.trim()) return text.includes("\n") ? "↵" : "\u00a0";
  return text
    .replace(/\r\n|\r|\n/g, "")
    .replace(/\t/g, "⇥")
    .replaceAll(" ", "\u00a0");
}

function shortId(value: string) {
  return value.slice(-6);
}

export function VariantScope({
  current,
  reference,
  referenceId,
  referenceStatus,
  referenceOptions,
  alignment,
  relation,
  selectedToken,
  onSelectToken,
  onSelectReference,
}: VariantScopeProps) {
  const [filter, setFilter] = useState<OriginFilter>(null);
  const origins = useMemo(() => current.tokens.map((_, tokenIndex) => {
    const columnIndex = alignment.columnByB.get(tokenIndex);
    return columnIndex != null && alignment.columns[columnIndex]?.kind === "same"
      ? "reference" as const
      : "variant" as const;
  }), [alignment, current.tokens]);
  const inheritedCount = origins.filter((origin) => origin === "reference").length;
  const variantCount = origins.filter((origin) => origin === "variant").length;
  const omittedCount = alignment.columns.filter((column) => column.kind === "a-only").length;

  return (
    <div className={`variant-scope ${filter ? `has-${filter}-filter` : ""}`} aria-label="Run variant provenance">
      <div className="variant-toolbar">
        <span><b>EVIDENCE</b>STRUCTURAL ALIGNMENT</span>
        <label>
          <span>REFERENCE RUN</span>
          <select
            value={referenceId}
            onChange={(event) => onSelectReference(event.target.value)}
            disabled={referenceStatus === "loading"}
          >
            <option value="">SELECT REFERENCE</option>
            {referenceOptions.map((run) => <option value={run.id} key={run.id}>{run.label}</option>)}
          </select>
        </label>
      </div>

      {!reference ? (
        <div className="variant-empty">
          <strong>
            {referenceStatus === "loading"
              ? "LOADING REFERENCE"
              : referenceStatus === "error"
                ? "REFERENCE LOAD FAILED"
                : "SELECT A REFERENCE RUN"}
          </strong>
        </div>
      ) : (
        <>
          <svg className="variant-flow-field" viewBox="0 0 1000 240" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <linearGradient id="variant-reference-flow" x1="0" y1="0" x2="1" y2="1">
                <stop stopColor="var(--signal-cyan)" stopOpacity=".7" />
                <stop offset="1" stopColor="var(--signal-mint)" stopOpacity=".1" />
              </linearGradient>
              <linearGradient id="variant-current-flow" x1="1" y1="0" x2="0" y2="1">
                <stop stopColor="var(--signal-pink)" stopOpacity=".7" />
                <stop offset="1" stopColor="var(--signal-violet)" stopOpacity=".1" />
              </linearGradient>
            </defs>
            {Array.from({ length: 5 }, (_, index) => (
              <Fragment key={index}>
                <path className="is-reference" d={`M 170 28 C ${260 + index * 20} ${74 + index * 12}, ${420 + index * 34} ${112 + index * 8}, ${500 + index * 12} 218`} />
                <path className="is-variant" d={`M 830 28 C ${740 - index * 20} ${74 + index * 12}, ${580 - index * 34} ${112 + index * 8}, ${500 - index * 12} 218`} />
              </Fragment>
            ))}
          </svg>
          <div className="variant-origins">
            <button
              type="button"
              className={`is-reference ${filter === "reference" ? "is-selected" : ""}`}
              aria-pressed={filter === "reference"}
              onClick={() => setFilter((currentFilter) => currentFilter === "reference" ? null : "reference")}
            >
              <i />
              <span>{relation?.referenceLabel ?? "REFERENCE RUN"}</span>
              <strong>{shortId(reference.id)}</strong>
              <output>{inheritedCount} MATCHED TOKENS</output>
            </button>
            <div className="variant-relation">
              <span>{relation?.kind.toUpperCase() ?? "RUN"}</span>
              <b>{relation?.evidence ?? "STRUCTURAL TOKEN ALIGNMENT"}</b>
            </div>
            <button
              type="button"
              className={`is-variant ${filter === "variant" ? "is-selected" : ""}`}
              aria-pressed={filter === "variant"}
              onClick={() => setFilter((currentFilter) => currentFilter === "variant" ? null : "variant")}
            >
              <i />
              <span>{relation?.currentLabel ?? "CURRENT RUN"}</span>
              <strong>{shortId(current.id)}</strong>
              <output>{variantCount} VARIANT TOKENS</output>
            </button>
          </div>

          <div className="variant-token-field">
            <header>
              <span>CURRENT OUTPUT</span>
              <b>{inheritedCount} MATCHED · {variantCount} VARIANT · {omittedCount} REFERENCE-ONLY</b>
            </header>
            <div role="listbox" aria-label="Current output token identity origins">
              {current.tokens.map((token, tokenIndex) => {
                const origin = origins[tokenIndex];
                const muted = Boolean(filter && filter !== origin);
                const hasBreak = token.text.includes("\n");
                return (
                  <Fragment key={`${tokenIndex}-${token.text}`}>
                    <button
                      type="button"
                      role="option"
                      className={`is-${origin} ${selectedToken === tokenIndex ? "is-selected" : ""} ${muted ? "is-muted" : ""}`}
                      aria-selected={selectedToken === tokenIndex}
                      aria-label={`Token ${tokenIndex + 1}: ${token.text || "blank"}, ${origin === "reference" ? "matched reference identity" : "variant identity"}`}
                      onClick={() => onSelectToken(tokenIndex)}
                      style={{ "--variant-confidence": token.confidence ?? 0 } as CSSProperties}
                    >
                      {visibleToken(token.text)}
                    </button>
                    {hasBreak && <span className="variant-token-break" aria-hidden="true" />}
                  </Fragment>
                );
              })}
            </div>
          </div>

          <div className="variant-evidence-note">
            <span><i className="is-reference" />MATCHED REFERENCE IDENTITY</span>
            <span><i className="is-variant" />CHANGED OR INSERTED IDENTITY</span>
            <b>IDENTITY ALIGNMENT DOES NOT ESTABLISH CAUSATION</b>
          </div>
        </>
      )}
    </div>
  );
}
