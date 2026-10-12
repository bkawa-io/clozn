import { useEffect, useRef, useState } from "react";
import { routeHref, type InspectRoute } from "../../app/router";
import { describeError, isContractError, studioApi } from "../../data/client";
import type {
  AnswerClaim,
  AnswerSegmentation,
  ClaimCategory,
  ClaimCategoryReason,
  ClaimSupportEvidence,
  ClaimSupportResult,
  ClaimSupportStatus,
  InfluenceGate,
} from "../../data/claimSupport";
import "./claimSupport.css";

type Resource =
  | { phase: "loading" }
  | { phase: "ready"; value: ClaimSupportEvidence }
  | { phase: "unavailable" | "malformed"; error: string };

type StatusFilter = "all" | ClaimSupportStatus;

interface StatusPresentation {
  label: string;
  glyph: string;
  summary: string;
}

const STATUS_ORDER: readonly ClaimSupportStatus[] = [
  "supported",
  "weakly_supported",
  "contradicted",
  "unsupported_by_supplied_materials",
  "unverifiable_from_available_evidence",
  "measurement_unavailable",
];

const STATUS_PRESENTATION: Readonly<Record<ClaimSupportStatus, StatusPresentation>> = {
  supported: {
    label: "Supported",
    glyph: "✓",
    summary: "Supported by measured supplied-source evidence.",
  },
  weakly_supported: {
    label: "Weakly supported",
    glyph: "≈",
    summary: "Related supplied text overlaps this claim, but that is not measured causal support.",
  },
  contradicted: {
    label: "Contradicted",
    glyph: "!",
    summary: "A cited supplied-source span directly conflicts with this claim.",
  },
  unsupported_by_supplied_materials: {
    label: "Unsupported by supplied materials",
    glyph: "∅",
    summary: "No match was found in the supplied materials after measurement. This does not mean the claim is false.",
  },
  unverifiable_from_available_evidence: {
    label: "Unverifiable from available evidence",
    glyph: "?",
    summary: "This claim category cannot be verified from the available supplied evidence.",
  },
  measurement_unavailable: {
    label: "Measurement unavailable",
    glyph: "—",
    summary: "Support was not measured or could not be resolved. This is not an unsupported finding.",
  },
};

const CATEGORY_LABELS: Readonly<Record<ClaimCategory, string>> = {
  factual_claim: "Factual claim",
  recommendation: "Recommendation",
  uncertainty_statement: "Uncertainty statement",
  instruction_procedure: "Instruction or procedure",
  non_verifiable_prose: "Non-verifiable prose",
};

const CATEGORY_REASON_LABELS: Readonly<Record<ClaimCategoryReason, string>> = {
  code_fence_block: "Code fence block",
  interrogative_sentence: "Interrogative sentence",
  hedge_marker: "Hedge marker",
  recommendation_marker: "Recommendation marker",
  list_item_imperative: "List-item imperative",
  imperative_lead: "Imperative lead",
  factual_declarative: "Factual declarative",
  no_deterministic_category_match: "No deterministic category match",
};

const GATE_COPY: Readonly<Record<InfluenceGate, { label: string; explanation: string }>> = {
  ok: {
    label: "Influence evidence available",
    explanation: "Factual claim results may use the recorded influence measurement and supplied-source comparison.",
  },
  no_influence_map: {
    label: "No influence map",
    explanation: "No influence map was recorded. Factual claims are measurement unavailable, not unsupported.",
  },
  influence_measurement_unavailable: {
    label: "Influence measurement unavailable",
    explanation: "The selected runtime could not provide influence evidence. This does not show absence of support.",
  },
  influence_measurement_error: {
    label: "Influence measurement failed",
    explanation: "The influence measurement ended in an error. No support conclusion can be drawn from that failure.",
  },
  answer_text_mismatch: {
    label: "Answer identity mismatch",
    explanation: "The measurement does not bind to this exact recorded answer, so support results are unavailable.",
  },
  no_resolvable_answer_spans: {
    label: "No resolvable answer spans",
    explanation: "Claim ranges could not be joined to measured answer spans. This is not an unsupported finding.",
  },
};

function shortId(value: string): string {
  return value.length > 22 ? `${value.slice(0, 10)}…${value.slice(-7)}` : value;
}

function methodLabel(result: ClaimSupportResult): string {
  switch (result.method.name) {
    case "forced_score_intervention": return `Forced-score intervention · ${result.method.maxAbsDeltaNats.toFixed(3)} max |Δ| nats`;
    case "textual_overlap": return `Textual overlap · ${(result.method.overlapFraction * 100).toFixed(1)}%`;
    case "numeric_or_date_mismatch": return "Numeric or date mismatch";
    case "direct_negation": return "Direct negation";
    case "measured_comparison_no_match": return "Measured comparison · no match";
    case "category_rule": return "Deterministic category rule";
    case "no_influence_map": return "No influence map";
    case "influence_measurement_unavailable": return "Influence measurement unavailable";
    case "influence_measurement_error": return "Influence measurement error";
    case "answer_text_mismatch": return "Answer text mismatch";
    case "no_resolvable_answer_spans": return "No resolvable answer spans";
  }
}

function segmentationCopy(segmentation: AnswerSegmentation): { title: string; detail: string } | undefined {
  switch (segmentation.state) {
    case "ok": return segmentation.claimCount === 0
      ? { title: "No claim ranges recorded", detail: "Segmentation completed but produced no claims for this recorded answer." }
      : undefined;
    case "empty": return {
      title: "Recorded answer is empty",
      detail: "Segmentation completed against the exact empty answer; there are no claims to evaluate.",
    };
    case "segmentation_limited": return {
      title: "Claim segmentation limited",
      detail: "The recorded answer has an unsupported-script density, so no claim ranges or support verdicts were produced.",
    };
    case "unavailable": return {
      title: "Claim segmentation unavailable",
      detail: segmentation.reason === "answer_text_redacted"
        ? "The recorded answer was redacted, so claim text and ranges are not available."
        : "No recorded answer text is available for segmentation.",
    };
  }
}

function sourceSpanIds(result: ClaimSupportResult): readonly string[] {
  return "sourceSpanIds" in result ? result.sourceSpanIds : [];
}

function RecordedAnswer({ evidence }: { evidence: ClaimSupportEvidence }) {
  if (evidence.recordedResponse === null) {
    return <p className="claim-recorded-unavailable">Recorded answer text is unavailable.</p>;
  }
  if (evidence.recordedResponse === "") {
    return <p className="claim-recorded-unavailable">The recorded answer is empty.</p>;
  }

  const codePoints = Array.from(evidence.recordedResponse);
  const fragments: React.ReactNode[] = [];
  let cursor = 0;
  evidence.claims.claims.forEach((claim, index) => {
    const range = claim.textSpan.resolution.canonical!;
    const result = evidence.support.results[index];
    const presentation = STATUS_PRESENTATION[result.status];
    if (range.start > cursor) fragments.push(codePoints.slice(cursor, range.start).join(""));
    fragments.push(
      <mark
        className={`claim-answer-mark state-${result.status}`}
        data-claim-index={claim.index}
        title={`Claim ${claim.index + 1}: ${presentation.label}`}
        key={`${claim.textSpan.addressId}:${range.start}`}
      >
        {evidence.claimTexts[index]}
      </mark>,
    );
    cursor = range.end;
  });
  if (cursor < codePoints.length) fragments.push(codePoints.slice(cursor).join(""));

  return <p className="claim-recorded-answer">{fragments}</p>;
}

function SpanLink({ runId, spanId, label }: { runId: string; spanId: string; label?: string }) {
  return (
    <a
      href={`/runs/${encodeURIComponent(runId)}/span-addresses#${spanId}`}
      title={spanId}
      aria-label={`${label ?? "Stable source span"} ${spanId}`}
    >
      {shortId(spanId)}
    </a>
  );
}

function ClaimCard({
  runId,
  claim,
  claimText,
  result,
}: {
  runId: string;
  claim: AnswerClaim;
  claimText: string;
  result: ClaimSupportResult;
}) {
  const presentation = STATUS_PRESENTATION[result.status];
  const sourceIds = sourceSpanIds(result);
  return (
    <article className={`claim-support-card state-${result.status}`} data-support-status={result.status}>
      <header>
        <div>
          <span className="claim-status-glyph" aria-hidden="true">{presentation.glyph}</span>
          <div><span>Claim {claim.index + 1}</span><strong>{presentation.label}</strong></div>
        </div>
        <SpanLink runId={runId} spanId={claim.textSpan.addressId} label="Stable claim span" />
      </header>
      <blockquote>{claimText}</blockquote>
      <p className="claim-status-summary">{presentation.summary}</p>
      <dl className="claim-classification">
        <div><dt>Category</dt><dd>{CATEGORY_LABELS[claim.category]}</dd></div>
        <div><dt>Classification basis</dt><dd>{CATEGORY_REASON_LABELS[claim.categoryReason]}</dd></div>
        <div><dt>Method</dt><dd>{methodLabel(result)}</dd></div>
      </dl>
      {sourceIds.length > 0 && (
        <div className="claim-source-spans">
          <span>Cited supplied-source spans</span>
          <ul>
            {sourceIds.map((sourceId) => (
              <li key={sourceId}><SpanLink runId={runId} spanId={sourceId} /></li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}

export function ClaimSupportSurface({ route }: { route: InspectRoute }) {
  const [resource, setResource] = useState<Resource>({ phase: "loading" });
  const [filter, setFilter] = useState<StatusFilter>("all");
  const requestGeneration = useRef(0);

  useEffect(() => {
    const generation = ++requestGeneration.current;
    const controller = new AbortController();
    setResource({ phase: "loading" });
    setFilter("all");
    void studioApi.claimSupportEvidence(route.runId, controller.signal).then((value) => {
      if (!controller.signal.aborted && requestGeneration.current === generation) {
        setResource({ phase: "ready", value });
      }
    }).catch((error: unknown) => {
      if (controller.signal.aborted || requestGeneration.current !== generation) return;
      setResource({
        phase: isContractError(error) ? "malformed" : "unavailable",
        error: describeError(error),
      });
    });
    return () => controller.abort();
  }, [route.runId]);

  const sessionHref = route.sessionId
    ? routeHref({ surface: "session", sessionId: route.sessionId, runId: route.runId })
    : routeHref({ surface: "sessions" });
  const contextHref = routeHref({
    surface: "inspect",
    runId: route.runId,
    view: "context",
    ...(route.sessionId ? { sessionId: route.sessionId } : {}),
  });
  const actionsHref = routeHref({
    surface: "inspect",
    runId: route.runId,
    view: "actions",
    ...(route.sessionId ? { sessionId: route.sessionId } : {}),
  });
  const evidence = resource.phase === "ready" ? resource.value : undefined;
  const segmentation = evidence ? segmentationCopy(evidence.claims.segmentation) : undefined;
  const counts = Object.fromEntries(STATUS_ORDER.map((status) => [
    status,
    evidence?.support.results.filter((result) => result.status === status).length ?? 0,
  ])) as Record<ClaimSupportStatus, number>;
  const visibleResults = evidence?.support.results.filter((result) => filter === "all" || result.status === filter) ?? [];

  return (
    <div className="inspect-surface claim-support-surface">
      <header className="inspect-heading">
        <div>
          <a className="back-link" href={sessionHref}>← {route.sessionId ? "Session" : "Sessions"}</a>
          <h1>Inspect</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {evidence && <span className="claim-support-total">{evidence.support.results.length} claims evaluated</span>}
      </header>

      <section className="inspect-instrument claim-support-instrument" aria-labelledby="claim-support-title">
        <header>
          <div><span>RECORDED CLAIM EVIDENCE</span><h2 id="claim-support-title">Claim support</h2></div>
          <code>GET only · exact answer identity</code>
        </header>
        <div className="claim-support-boundary">
          <p>These labels describe support from the supplied materials. Unsupported is not the same as false, and unavailable is not the same as unsupported.</p>
          <a href={contextHref}>Open Received Context</a>
        </div>

        {resource.phase === "loading" && <p className="claim-support-primary-state" role="status">Reading recorded claim evidence…</p>}
        {resource.phase === "unavailable" && (
          <p className="claim-support-primary-state is-error" role="alert">
            Claim evidence unavailable.<span>{resource.error}</span>
          </p>
        )}
        {resource.phase === "malformed" && (
          <p className="claim-support-primary-state is-error" role="alert">
            Claim evidence response malformed.<span>{resource.error}</span>
          </p>
        )}

        {evidence && (
          <>
            <section className="claim-answer-section" aria-labelledby="claim-answer-title">
              <div>
                <span>Exact recorded response</span>
                <h3 id="claim-answer-title">Answer with verified claim ranges</h3>
              </div>
              <RecordedAnswer evidence={evidence} />
            </section>

            <p className={`claim-influence-gate state-${evidence.support.influenceGate}`} role="note">
              <strong>{GATE_COPY[evidence.support.influenceGate].label}</strong>
              <span>{GATE_COPY[evidence.support.influenceGate].explanation}</span>
            </p>

            {segmentation ? (
              <p className={`claim-segmentation-state state-${evidence.claims.segmentation.state}`} role="status">
                <strong>{segmentation.title}</strong><span>{segmentation.detail}</span>
              </p>
            ) : (
              <>
                <nav className="claim-status-filters" aria-label="Filter claims by support status">
                  <button type="button" className={filter === "all" ? "is-active" : undefined} aria-pressed={filter === "all"} onClick={() => setFilter("all")}>All <b>{evidence.support.results.length}</b></button>
                  {STATUS_ORDER.map((status) => (
                    <button
                      type="button"
                      className={`state-${status}${filter === status ? " is-active" : ""}`}
                      aria-pressed={filter === status}
                      onClick={() => setFilter(status)}
                      key={status}
                    >
                      <i aria-hidden="true">{STATUS_PRESENTATION[status].glyph}</i>
                      {STATUS_PRESENTATION[status].label} <b>{counts[status]}</b>
                    </button>
                  ))}
                </nav>
                {visibleResults.length > 0 ? (
                  <div className="claim-support-results" aria-live="polite">
                    {visibleResults.map((result) => (
                      <ClaimCard
                        runId={route.runId}
                        claim={evidence.claims.claims[result.claimIndex]}
                        claimText={evidence.claimTexts[result.claimIndex]}
                        result={result}
                        key={result.claimAddressId}
                      />
                    ))}
                  </div>
                ) : (
                  <p className="claim-filter-empty" role="status">No claims have this recorded support status.</p>
                )}
              </>
            )}

            <section className="claim-retry-link" aria-labelledby="claim-retry-title">
              <div>
                <span>Separate workflow</span>
                <h3 id="claim-retry-title">Whole-answer corrective retries</h3>
                <p>This evidence view is read-only. Existing whole-answer retries require a separate preview and explicit confirmation.</p>
              </div>
              <a href={actionsHref}>Open corrective actions</a>
            </section>
          </>
        )}
      </section>

      <aside className="claim-read-boundary">
        <strong>Read-only boundary</strong>
        <p>This view reads stored claim support and the already-recorded answer. It does not score, retry, or modify the Run.</p>
      </aside>
    </div>
  );
}
