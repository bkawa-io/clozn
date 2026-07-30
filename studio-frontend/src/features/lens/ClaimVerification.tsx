import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  cancelCorrectivePreview,
  confirmCorrectivePreview,
  correctiveIdempotencyKey as idempotencyKey,
  describeCorrectiveFlowError as describeError,
  loadCorrectiveActions,
  previewCorrectiveAction,
  type CorrectivePreviewReceipt,
  type CorrectiveRegistry,
  type CorrectiveResult,
} from "../../data/correctiveFlow";
import {
  loadClaimVerification,
  loadRunAnswerText,
  type AnswerClaimsDocument,
  type Claim,
  type ClaimCategory,
  type ClaimSupportDocument,
  type ClaimSupportMethod,
  type ClaimSupportResult,
  type ClaimSupportStatus,
  type InfluenceGate,
} from "../../data/claimSupport";
import { VirtualList } from "../experiments/VirtualList";

/**
 * E3 -- the answer verification panel. Renders E1's deterministic claim segmentation
 * (`clozn.answer-claims.v1`) inline over the run's recorded answer text, each claim marked with E2's
 * per-claim verification status (`clozn.claim-support.v1`), fetched together from
 * `GET /runs/<id>/claim-support` (clozn/server/routes/claim_support.py). The recorded answer text itself
 * is fetched separately (`loadRunAnswerText`, the same untouched `GET /runs/<id>` every other Lens view
 * already reads `response` from) -- claim-support stays metadata-only, on purpose, matching every other
 * derived-artifact route in this codebase; see data/claimSupport.ts's own docstring.
 *
 * THE LANGUAGE DISCIPLINE IS THE POINT OF THIS PANEL, NOT A DETAIL OF IT
 * ------------------------------------------------------------------------
 * `unsupported_by_supplied_materials` renders as "not supported by supplied materials" -- never "false",
 * "wrong", or "incorrect" anywhere in this file (see claimStatusMeta below; tests scan rendered output
 * for exactly those three words). `measurement_unavailable` ("we could not measure this") gets its own
 * glyph, colour, and copy, deliberately never adjacent in meaning to "not supported" -- a reader must
 * never come away thinking a claim the measurement never reached was checked and found wanting.
 * `contradicted` always shows the contradicting source span and its deterministic basis (a numeric/date
 * mismatch or a negation) inline in the drawer, because a user challenged on this status deserves the
 * receipt, not just the verdict. Non-factual categories (`unverifiable_from_available_evidence`, always
 * and only the four non-`factual_claim` categories -- see clozn/runs/claim_support.py's own "status and
 * category, one rule each") render subdued: present, explained, never alarming.
 *
 * CROSS-LINKING, BOTH DIRECTIONS -- SAME DISCIPLINE AS WhatMattered.tsx
 * -------------------------------------------------------------------------
 * Selecting an evidence-bearing claim (supported/weakly_supported/contradicted) opens the source drawer
 * showing its cited source spans and the method that produced its status; selecting one of those sources
 * from the drawer highlights every claim that cites it, including claims other than the one that opened
 * the drawer. One `Selection` union drives both directions -- see `linkedClaimIndices`/`linkedSourceIds`.
 *
 * Rendering this panel fires up to three GETs (claim-support, the run's answer text, the corrective
 * actions registry for the retry button's availability) -- never a measurement, never a mutation. The
 * retry button's own preview/confirm calls only ever fire from an explicit click, reusing D3's real
 * preview -> confirm mechanics (`data/correctiveFlow.ts`) exactly as D5's guided-repair panel does.
 */

export interface ClaimVerificationProps {
  runId: string;
}

type Resource<T> =
  | { status: "idle" | "loading" }
  | { status: "failed"; message: string }
  | { status: "ready"; value: T };

// ============================================================================================ presentation

export const CLAIM_STATUS_ORDER: ClaimSupportStatus[] = [
  "supported",
  "weakly_supported",
  "contradicted",
  "unsupported_by_supplied_materials",
  "measurement_unavailable",
  "unverifiable_from_available_evidence",
];

export interface ClaimStatusMeta {
  glyph: string;
  label: string;
  description: string;
}

/** One rule each, mirroring clozn.claim_support's own docstring exactly in wording -- this function is
 * the ONE place claim-status copy lives, so the never-"false" language rule has exactly one call site to
 * audit. */
export function claimStatusMeta(status: ClaimSupportStatus): ClaimStatusMeta {
  switch (status) {
    case "supported":
      return { glyph: "SP", label: "SUPPORTED", description: "a measured, causal effect from a source" };
    case "weakly_supported":
      return { glyph: "WS", label: "WEAKLY SUPPORTED", description: "word overlap, not a measured effect" };
    case "contradicted":
      return { glyph: "CN", label: "CONTRADICTED", description: "a source disagrees on a number/date, or negates this claim" };
    case "unsupported_by_supplied_materials":
      return {
        glyph: "NS", label: "NOT SUPPORTED BY SUPPLIED MATERIALS",
        description: "not a judgment on whether this claim is true",
      };
    case "measurement_unavailable":
      return {
        glyph: "MU", label: "WE COULD NOT MEASURE THIS",
        description: "distinct from 'not supported' -- the measurement could not run",
      };
    case "unverifiable_from_available_evidence":
      return { glyph: "UV", label: "UNVERIFIABLE FROM AVAILABLE EVIDENCE", description: "not a checkable factual assertion" };
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

export function claimCategoryLabel(category: ClaimCategory): string {
  switch (category) {
    case "factual_claim": return "FACTUAL CLAIM";
    case "recommendation": return "RECOMMENDATION";
    case "uncertainty_statement": return "UNCERTAINTY STATEMENT";
    case "instruction_procedure": return "INSTRUCTION / PROCEDURE";
    case "non_verifiable_prose": return "PROSE";
    default: {
      const exhaustive: never = category;
      return exhaustive;
    }
  }
}

/** The five reasons a WHOLE-RUN measurement gate can fail -- shared verbatim between `methodLabel`'s own
 * five matching `ClaimSupportMethodName` branches and `gateNotice` below, one string each, never
 * duplicated: `InfluenceGate`'s non-"ok" values are exactly this same five-name subset of
 * `ClaimSupportMethodName` (clozn.claim-support.v1's own `source.influence_map.gate` enum). */
const UNAVAILABLE_REASON: Record<Exclude<InfluenceGate, "ok">, string> = {
  no_influence_map: "no source-influence measurement exists yet",
  influence_measurement_unavailable: "the source-influence measurement is unavailable",
  influence_measurement_error: "the source-influence measurement did not complete cleanly",
  answer_text_mismatch: "the measurement no longer matches this answer's text",
  no_resolvable_answer_spans: "the measurement could not be matched to this answer's spans",
};

/** One line describing HOW a status was reached -- shown in the claim list and, in full, in the source
 * drawer. Every branch is worded to survive the same never-"false"/"wrong"/"incorrect" scan the six
 * statuses themselves are held to. */
export function methodLabel(method: ClaimSupportMethod): string {
  switch (method.name) {
    case "forced_score_intervention":
      return method.maxAbsDeltaNats != null
        ? `measured effect, up to ${method.maxAbsDeltaNats.toFixed(4)} nats`
        : "measured effect";
    case "textual_overlap":
      return method.overlapFraction != null
        ? `${Math.round(method.overlapFraction * 100)}% word overlap with a source`
        : "word overlap with a source";
    case "numeric_or_date_mismatch":
      return "numeric/date mismatch against a source";
    case "direct_negation":
      return "direct negation in a source";
    case "measured_comparison_no_match":
      return "measured -- no matching source found";
    case "category_rule":
      return "category rule -- not a checkable factual assertion";
    case "no_influence_map":
    case "influence_measurement_unavailable":
    case "influence_measurement_error":
    case "answer_text_mismatch":
    case "no_resolvable_answer_spans":
      return UNAVAILABLE_REASON[method.name];
    default: {
      const exhaustive: never = method.name;
      return exhaustive;
    }
  }
}

function gateNotice(gate: InfluenceGate): string | undefined {
  return gate === "ok" ? undefined : UNAVAILABLE_REASON[gate];
}

// ================================================================================================ offsets

/** Unicode CODE POINT array -- `Array.from` iterates a string by code point, correctly handling
 * surrogate pairs, unlike `string.slice`'s UTF-16 code-unit indices. Every offset this panel receives
 * (`textSpan.resolution.canonical.start`/`end`) is a code-point offset per `offsetContract`; slicing with
 * plain `.slice()` would silently misalign on any answer containing a character outside the BMP. */
export function toCodePoints(text: string): string[] {
  return Array.from(text);
}

function sliceCodePoints(points: string[], start: number, end: number): string {
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return "";
  return points.slice(Math.max(0, start), Math.max(0, end)).join("");
}

// ================================================================================================== rows

interface ClaimRow {
  claim: Claim;
  result?: ClaimSupportResult;
  start?: number;
  end?: number;
  text?: string;
}

function buildRows(claims: Claim[], resultsByIndex: Map<number, ClaimSupportResult>, codePoints: string[] | null): ClaimRow[] {
  return claims.map((claim) => {
    const canonical = claim.textSpan.resolution.canonical;
    const resolvable = (
      claim.textSpan.resolution.state === "metadata_only" || claim.textSpan.resolution.state === "exact"
    ) && canonical?.start != null && canonical?.end != null;
    const start = resolvable ? canonical!.start : undefined;
    const end = resolvable ? canonical!.end : undefined;
    const text = resolvable && codePoints ? sliceCodePoints(codePoints, start!, end!) : undefined;
    return { claim, result: resultsByIndex.get(claim.index), start, end, text };
  });
}

interface InlineSegment {
  key: string;
  kind: "plain" | "claim";
  text: string;
  row?: ClaimRow;
}

/** Every code point of `codePoints` appears in exactly one segment, in order -- claimed spans as `"claim"`
 * segments, everything else (whitespace, punctuation, prose the segmentation pass left unclaimed) as
 * `"plain"` segments, so the rendered answer is always the complete recorded text, never a lossy subset. */
export function buildInlineSegments(rows: ClaimRow[], codePoints: string[]): InlineSegment[] {
  const resolved = rows
    .filter((row): row is ClaimRow & { start: number; end: number } => row.start != null && row.end != null)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const segments: InlineSegment[] = [];
  let cursor = 0;
  for (const row of resolved) {
    if (row.start < cursor) continue; // overlap guard -- never render the same text twice
    if (row.start > cursor) {
      segments.push({ key: `plain-${cursor}`, kind: "plain", text: sliceCodePoints(codePoints, cursor, row.start) });
    }
    segments.push({ key: `claim-${row.claim.index}`, kind: "claim", text: sliceCodePoints(codePoints, row.start, row.end), row });
    cursor = row.end;
  }
  if (cursor < codePoints.length) {
    segments.push({ key: `plain-${cursor}`, kind: "plain", text: sliceCodePoints(codePoints, cursor, codePoints.length) });
  }
  return segments;
}

// =============================================================================================== selection

export type Selection = { kind: "claim"; index: number } | { kind: "source"; addressId: string } | null;

function spanHref(runId: string, addressId: string): string {
  return `#/runs/${encodeURIComponent(runId)}/span-addresses#${addressId}`;
}

/** The three request-failure/data-integrity notices this panel can show (verification request failed,
 * answer-text request failed, answer text out of sync) share one shape -- rendered once here. */
function Notice({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="claim-verify-notice is-failed" role="alert">
      <strong>{title}</strong>
      <span>{children}</span>
    </div>
  );
}

// =================================================================================================== misc

const RETRY_ACTION_ID = "use-context";
const VIRTUALIZE_THRESHOLD = 200;
const CLAIM_ROW_HEIGHT = 108;

// =============================================================================================== component

export function ClaimVerification({ runId }: ClaimVerificationProps) {
  const [verification, setVerification] = useState<Resource<{ claims: AnswerClaimsDocument; support: ClaimSupportDocument }>>({ status: "idle" });
  const [answerText, setAnswerText] = useState<Resource<string>>({ status: "idle" });
  const [registry, setRegistry] = useState<Resource<CorrectiveRegistry>>({ status: "idle" });

  const [visibleStatuses, setVisibleStatuses] = useState<Set<ClaimSupportStatus>>(new Set(CLAIM_STATUS_ORDER));
  const [selection, setSelection] = useState<Selection>(null);

  const [retryPreview, setRetryPreview] = useState<CorrectivePreviewReceipt>();
  const [retryResult, setRetryResult] = useState<CorrectiveResult>();
  const [retryPhase, setRetryPhase] = useState<"idle" | "previewing" | "preview_ready" | "confirming" | "result_ready">("idle");
  const [retryError, setRetryError] = useState("");

  // One monotonic id per RUN SELECTION, the same guard every other Lens evidence slot panel uses
  // (WhatMattered.tsx, DiagnosisRepair.tsx): every async response below checks this before writing state,
  // so a response for a run this panel has since navigated away from can never land on top of the run now
  // showing.
  const requestIdRef = useRef(0);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    setSelection(null);
    setVisibleStatuses(new Set(CLAIM_STATUS_ORDER));
    setRetryPreview(undefined);
    setRetryResult(undefined);
    setRetryPhase("idle");
    setRetryError("");
    if (!runId) {
      setVerification({ status: "idle" });
      setAnswerText({ status: "idle" });
      setRegistry({ status: "idle" });
      return;
    }
    setVerification({ status: "loading" });
    setAnswerText({ status: "loading" });
    setRegistry({ status: "loading" });
    const controller = new AbortController();
    void loadClaimVerification(runId, controller.signal).then((value) => {
      if (requestIdRef.current !== requestId) return;
      setVerification({ status: "ready", value });
    }).catch((error) => {
      if (requestIdRef.current !== requestId) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      setVerification({ status: "failed", message: describeError(error) });
    });
    void loadRunAnswerText(runId, controller.signal).then((value) => {
      if (requestIdRef.current !== requestId) return;
      setAnswerText({ status: "ready", value });
    }).catch((error) => {
      if (requestIdRef.current !== requestId) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      setAnswerText({ status: "failed", message: describeError(error) });
    });
    void loadCorrectiveActions(runId, controller.signal).then((value) => {
      if (requestIdRef.current !== requestId) return;
      setRegistry({ status: "ready", value });
    }).catch((error) => {
      if (requestIdRef.current !== requestId) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      setRegistry({ status: "failed", message: describeError(error) });
    });
    return () => controller.abort();
  }, [runId]);

  const claimsDoc = verification.status === "ready" ? verification.value.claims : undefined;
  const supportDoc = verification.status === "ready" ? verification.value.support : undefined;
  const text = answerText.status === "ready" ? answerText.value : undefined;

  const codePoints = useMemo(() => text != null ? toCodePoints(text) : null, [text]);

  // The claim-support route and the answer-text route are two independent fetches; if the answer this
  // run now records has changed between them (a race, or a corrective retry that landed on this run id)
  // the two code-point counts diverge -- treat that as untrustworthy rather than silently mis-slicing.
  const answerTextTrusted = useMemo(() => {
    if (!claimsDoc || codePoints == null) return false;
    if (claimsDoc.segmentation.state === "unavailable") return true; // nothing to cross-check
    return claimsDoc.answerSource.basisCodePoints === codePoints.length;
  }, [claimsDoc, codePoints]);

  const resultsByIndex = useMemo(() => {
    const index = new Map<number, ClaimSupportResult>();
    for (const result of supportDoc?.results ?? []) index.set(result.claimIndex, result);
    return index;
  }, [supportDoc]);

  const rows = useMemo(
    () => buildRows(claimsDoc?.claims ?? [], resultsByIndex, answerTextTrusted ? codePoints : null),
    [claimsDoc, resultsByIndex, answerTextTrusted, codePoints],
  );

  const inlineSegments = useMemo(
    () => answerTextTrusted && codePoints ? buildInlineSegments(rows, codePoints) : null,
    [rows, codePoints, answerTextTrusted],
  );

  // Reverse index: source address id -> every claim index that cites it -- the "selecting a source
  // highlights its claims" half of the cross-linking discipline.
  const claimsBySource = useMemo(() => {
    const index = new Map<string, number[]>();
    for (const row of rows) {
      for (const sourceId of row.result?.sourceSpanIds ?? []) {
        const list = index.get(sourceId) ?? [];
        list.push(row.claim.index);
        index.set(sourceId, list);
      }
    }
    return index;
  }, [rows]);

  const linkedClaimIndices = useMemo(() => {
    if (!selection) return new Set<number>();
    if (selection.kind === "claim") return new Set([selection.index]);
    return new Set(claimsBySource.get(selection.addressId) ?? []);
  }, [selection, claimsBySource]);

  const linkedSourceIds = useMemo(() => {
    if (!selection) return new Set<string>();
    if (selection.kind === "source") return new Set([selection.addressId]);
    const row = rows.find((item) => item.claim.index === selection.index);
    return new Set(row?.result?.sourceSpanIds ?? []);
  }, [selection, rows]);

  const selectedRow = selection?.kind === "claim" ? rows.find((row) => row.claim.index === selection.index) : undefined;

  const statusCounts = useMemo(() => {
    const counts = new Map<ClaimSupportStatus, number>();
    for (const row of rows) if (row.result) counts.set(row.result.status, (counts.get(row.result.status) ?? 0) + 1);
    return counts;
  }, [rows]);

  const filteredRows = useMemo(
    () => rows.filter((row) => !row.result || visibleStatuses.has(row.result.status)),
    [rows, visibleStatuses],
  );

  function toggleStatus(status: ClaimSupportStatus) {
    setVisibleStatuses((current) => {
      const next = new Set(current);
      if (next.has(status)) next.delete(status); else next.add(status);
      return next;
    });
  }

  function resetFilter() {
    setVisibleStatuses(new Set(CLAIM_STATUS_ORDER));
  }

  function selectClaim(index: number) {
    setSelection((current) => current?.kind === "claim" && current.index === index ? null : { kind: "claim", index });
  }

  function selectSource(addressId: string) {
    setSelection((current) => current?.kind === "source" && current.addressId === addressId ? null : { kind: "source", addressId });
  }

  const unsupportedCount = supportDoc?.results.filter((r) => r.status === "unsupported_by_supplied_materials").length ?? 0;
  const registryHasRetryAction = registry.status === "ready" && registry.value.actions.some((a) => a.id === RETRY_ACTION_ID);
  const retryBusy = retryPhase === "previewing" || retryPhase === "confirming";

  function retryDisabledReason(): string | undefined {
    if (registry.status === "loading" || registry.status === "idle") return "loading corrective actions";
    if (registry.status === "failed") return "the corrective actions registry failed to load";
    if (!registryHasRetryAction) return "no corrective action maps to this yet (D1/D3 vocabulary gap)";
    if (unsupportedCount === 0) return "there are no unsupported claims to retry";
    return undefined;
  }

  async function startRetryPreview() {
    if (!runId) return;
    const generation = requestIdRef.current;
    setRetryPreview(undefined);
    setRetryResult(undefined);
    setRetryError("");
    setRetryPhase("previewing");
    try {
      const next = await previewCorrectiveAction(runId, RETRY_ACTION_ID, "prompt_policy");
      if (requestIdRef.current !== generation) return;
      setRetryPreview(next);
      setRetryPhase("preview_ready");
    } catch (error) {
      if (requestIdRef.current !== generation) return;
      setRetryPhase("idle");
      setRetryError(`${describeError(error)} -- no run was created; the original run is unchanged.`);
    }
  }

  async function cancelRetryPreview() {
    if (!retryPreview) return;
    const generation = requestIdRef.current;
    try {
      await cancelCorrectivePreview(retryPreview.preview_id);
    } catch {
      // Best effort -- an unconfirmed preview also simply expires server-side.
    }
    if (requestIdRef.current !== generation) return;
    setRetryPreview(undefined);
    setRetryPhase("idle");
    setRetryError("");
  }

  async function confirmRetryPreview() {
    if (!retryPreview) return;
    const generation = requestIdRef.current;
    setRetryPhase("confirming");
    setRetryError("");
    try {
      const next = await confirmCorrectivePreview(retryPreview.preview_id, idempotencyKey("claim-verify-retry"));
      if (requestIdRef.current !== generation) return;
      setRetryResult(next);
      setRetryPhase("result_ready");
      if (next.outcome.status !== "succeeded") {
        setRetryError(`${next.outcome.note ?? "the retry did not complete"} -- the original run is unchanged.`);
      }
    } catch (error) {
      if (requestIdRef.current !== generation) return;
      setRetryPhase("preview_ready");
      setRetryError(`${describeError(error)} -- the original run is unchanged.`);
    }
  }

  return (
    <section className="claim-verify" aria-labelledby="claim-verify-title">
      <header className="claim-verify-head">
        <div>
          <span className="eyebrow">ANSWER VERIFICATION</span>
          <h3 id="claim-verify-title">Are the claims supported?</h3>
        </div>
        {claimsDoc && (
          <span className="claim-verify-count">
            {claimsDoc.claims.length} CLAIM{claimsDoc.claims.length === 1 ? "" : "S"}
          </span>
        )}
      </header>

      <p className="claim-verify-boundary">
        Each claim below is marked with what the supplied materials show for it -- never whether it is true.
      </p>

      {verification.status === "idle" || verification.status === "loading" || answerText.status === "loading" ? (
        <div className="claim-verify-empty">LOADING CLAIM VERIFICATION</div>
      ) : verification.status === "failed" ? (
        <Notice title="CLAIM VERIFICATION REQUEST FAILED">{verification.message}</Notice>
      ) : (
        <>
          {answerText.status === "failed" && (
            <Notice title="ANSWER TEXT REQUEST FAILED">
              Claim statuses remain visible below; inline highlighting over the answer text is not.
            </Notice>
          )}

          {claimsDoc!.segmentation.state !== "ok" && (
            <div className={`claim-verify-banner is-${claimsDoc!.segmentation.state}`}>
              <strong>{claimsDoc!.segmentation.state.replaceAll("_", " ").toUpperCase()}</strong>
              <span>{segmentationNote(claimsDoc!.segmentation.state, claimsDoc!.segmentation.reason)}</span>
            </div>
          )}

          {!answerTextTrusted && claimsDoc!.segmentation.state === "ok" && (
            <Notice title="ANSWER TEXT OUT OF SYNC">
              The displayed answer no longer matches the text these claims were measured against, so
              inline highlighting is withheld instead of shown against mismatched text.
            </Notice>
          )}

          {supportDoc && gateNotice(supportDoc.influenceGate) && (
            <p className="claim-verify-gate-notice">Whole run: {gateNotice(supportDoc.influenceGate)}.</p>
          )}

          <ul className="claim-verify-legend" aria-label="Claim status filter">
            {CLAIM_STATUS_ORDER.map((status) => {
              const meta = claimStatusMeta(status);
              const active = visibleStatuses.has(status);
              return (
                <li key={status} className={`is-${status}`}>
                  <button
                    type="button"
                    aria-pressed={active}
                    className={active ? "is-active" : "is-inactive"}
                    onClick={() => toggleStatus(status)}
                  >
                    <b>{meta.glyph}</b>
                    <span>{meta.label}</span>
                    <em>{statusCounts.get(status) ?? 0}</em>
                  </button>
                </li>
              );
            })}
            <li className="claim-verify-reset">
              <button type="button" onClick={resetFilter}>SHOW ALL</button>
            </li>
          </ul>

          {inlineSegments && (
            <div className="claim-verify-passage" aria-label="Answer text with claim markers">
              {inlineSegments.map((segment) => {
                if (segment.kind === "plain") return <span key={segment.key}>{segment.text}</span>;
                const row = segment.row!;
                const status = row.result?.status;
                const meta = status ? claimStatusMeta(status) : undefined;
                const subdued = status === "unverifiable_from_available_evidence";
                const isLinked = linkedClaimIndices.has(row.claim.index);
                const isSelected = selection?.kind === "claim" && selection.index === row.claim.index;
                return (
                  <mark
                    key={segment.key}
                    data-claim-index={row.claim.index}
                    data-claim-status={status}
                    className={[
                      "claim-mark",
                      status ? `is-${status}` : "",
                      subdued ? "is-subdued" : "",
                      isLinked ? "is-linked" : "",
                      isSelected ? "is-selected" : "",
                    ].filter(Boolean).join(" ")}
                    onClick={() => selectClaim(row.claim.index)}
                  >
                    {segment.text}
                    {meta && <sup className="claim-mark-glyph">{meta.glyph}</sup>}
                  </mark>
                );
              })}
            </div>
          )}

          <ClaimList
            rows={filteredRows}
            totalCount={rows.length}
            selection={selection}
            linkedClaimIndices={linkedClaimIndices}
            onSelect={selectClaim}
          />

          {selection && (
            <SourceDrawer
              runId={runId}
              selection={selection}
              rows={rows}
              selectedRow={selectedRow}
              linkedClaimIndices={linkedClaimIndices}
              linkedSourceIds={linkedSourceIds}
              onSelectClaim={selectClaim}
              onSelectSource={selectSource}
              onClose={() => setSelection(null)}
            />
          )}

          <section className="claim-verify-retry" aria-label="Retry unsupported claims">
            <header className="section-title">
              <h3>Retry unsupported claims</h3>
              <span>{unsupportedCount}</span>
            </header>
            <p className="claim-verify-retry-note">
              Asks the model to ground its answer in the supplied materials and say what is missing.
            </p>
            <button
              type="button"
              disabled={Boolean(retryDisabledReason()) || retryBusy}
              onClick={() => void startRetryPreview()}
            >
              {retryBusy ? "WORKING…" : "RETRY UNSUPPORTED CLAIMS"}
            </button>
            {retryDisabledReason() && (
              <p className="claim-verify-retry-reason">{retryDisabledReason()}</p>
            )}
            {retryError && <p className="claim-verify-retry-error" role="alert">{retryError}</p>}

            {retryPreview && !retryResult && (
              <p className="claim-verify-retry-preview">
                WILL INJECT: {retryPreview.action.description}
                <button type="button" disabled={retryPhase === "confirming"} onClick={() => void cancelRetryPreview()}>
                  CANCEL
                </button>
                <button
                  type="button"
                  className="is-primary"
                  disabled={retryPhase === "confirming"}
                  onClick={() => void confirmRetryPreview()}
                >
                  {retryPhase === "confirming" ? "RUNNING…" : "CONFIRM -- RUN MATCHED RETRY"}
                </button>
              </p>
            )}

            {retryResult && (
              <p className={`claim-verify-retry-result is-${retryResult.outcome.status}`}>
                <b>{retryResult.outcome.status.toUpperCase().replaceAll("_", " ")}</b>: {retryResult.comparison.note}
                {retryResult.outcome.status === "succeeded" && retryResult.children.corrected.run_id && (
                  <a href={`#/compare/${encodeURIComponent(retryResult.parent_run_id)}/${encodeURIComponent(retryResult.children.corrected.run_id)}`}>
                    OPEN PAIRED COMPARISON
                  </a>
                )}
                <small>To keep or discard, use Corrective retries in Diagnosis &amp; Repair below.</small>
              </p>
            )}
          </section>
        </>
      )}
    </section>
  );
}

function segmentationNote(state: string, reason?: string): string {
  switch (reason) {
    case "answer_text_empty": return "the recorded answer text is empty -- nothing to verify.";
    case "answer_text_redacted": return "this run's text is redacted -- claim segmentation could not run.";
    case "no_answer_text": return "no answer text was recorded for this run.";
    case "unsupported_script_density":
      return "this answer is dense in a script these heuristics do not reliably tokenize -- zero claims "
        + "were produced rather than guessing at sentence boundaries.";
    default: return `segmentation state: ${state}.`;
  }
}

// ================================================================================================ list

function ClaimList({
  rows, totalCount, selection, linkedClaimIndices, onSelect,
}: {
  rows: ClaimRow[];
  totalCount: number;
  selection: Selection;
  linkedClaimIndices: Set<number>;
  onSelect: (index: number) => void;
}) {
  const renderRow = (row: ClaimRow) => (
    <ClaimListRow
      row={row}
      isSelected={selection?.kind === "claim" && selection.index === row.claim.index}
      isLinked={linkedClaimIndices.has(row.claim.index)}
      onSelect={onSelect}
    />
  );

  return (
    <section className="claim-verify-list" aria-label="Claims">
      <header>
        <span>CLAIM LIST</span>
        <b>{rows.length}{rows.length !== totalCount ? ` OF ${totalCount}` : ""} SHOWN</b>
      </header>
      {rows.length === 0 ? (
        <p className="claim-verify-list-empty">
          {totalCount === 0 ? "No claims were extracted from this answer." : "No claims match the current filter."}
        </p>
      ) : rows.length > VIRTUALIZE_THRESHOLD ? (
        <VirtualList
          items={rows}
          rowHeight={CLAIM_ROW_HEIGHT}
          ariaLabel="Claim list, virtualized"
          emptyLabel="No claims match the current filter."
          keyFor={(row) => `claim-row-${row.claim.index}`}
          renderRow={renderRow}
          className="claim-verify-rows-virtual"
        />
      ) : (
        <ol className="claim-verify-rows">
          {rows.map((row) => <li key={row.claim.index}>{renderRow(row)}</li>)}
        </ol>
      )}
    </section>
  );
}

function ClaimListRow({
  row, isSelected, isLinked, onSelect,
}: {
  row: ClaimRow;
  isSelected: boolean;
  isLinked: boolean;
  onSelect: (index: number) => void;
}) {
  const status = row.result?.status;
  const meta = status ? claimStatusMeta(status) : undefined;
  const subdued = status === "unverifiable_from_available_evidence";
  return (
    <button
      type="button"
      className={[
        "claim-verify-row",
        status ? `is-${status}` : "",
        subdued ? "is-subdued" : "",
        isSelected ? "is-selected" : "",
        isLinked && !isSelected ? "is-linked" : "",
      ].filter(Boolean).join(" ")}
      data-claim-index={row.claim.index}
      data-claim-status={status}
      onClick={() => onSelect(row.claim.index)}
    >
      <span className="claim-verify-row-index">#{row.claim.index + 1}</span>
      <span className="claim-verify-row-category">{claimCategoryLabel(row.claim.category)}</span>
      {meta && (
        <span className={`claim-verify-row-status is-${status}`}>
          <b>{meta.glyph}</b>{meta.label}
        </span>
      )}
      <span className="claim-verify-row-text">{row.text || "(text unavailable)"}</span>
      {row.result && <span className="claim-verify-row-method">{methodLabel(row.result.method)}</span>}
    </button>
  );
}

// =============================================================================================== drawer

function SourceDrawer({
  runId, selection, rows, selectedRow, linkedClaimIndices, linkedSourceIds, onSelectClaim, onSelectSource, onClose,
}: {
  runId: string;
  selection: NonNullable<Selection>;
  rows: ClaimRow[];
  selectedRow?: ClaimRow;
  linkedClaimIndices: Set<number>;
  linkedSourceIds: Set<string>;
  onSelectClaim: (index: number) => void;
  onSelectSource: (addressId: string) => void;
  onClose: () => void;
}) {
  return (
    <aside className="claim-verify-drawer" aria-label="Source drawer">
      <header>
        <span>{selection.kind === "claim" ? "CLAIM" : "SOURCE"}</span>
        <button type="button" onClick={onClose}>CLOSE</button>
      </header>

      {selection.kind === "claim" && selectedRow && (
        <ClaimDrawerBody runId={runId} row={selectedRow} linkedSourceIds={linkedSourceIds} onSelectSource={onSelectSource} />
      )}
      {selection.kind === "claim" && !selectedRow && (
        <p className="claim-verify-drawer-empty">This claim is no longer available.</p>
      )}
      {selection.kind === "source" && (
        <SourceDrawerBody
          runId={runId}
          addressId={selection.addressId}
          rows={rows}
          linkedClaimIndices={linkedClaimIndices}
          onSelectClaim={onSelectClaim}
        />
      )}
    </aside>
  );
}

/** Shared by both drawer bodies below -- a claim's cited sources and a source's citing claims are the
 * SAME cross-linked-list shape (a label, an emptiness fallback, N clickable rows), rendered once here
 * rather than twice. */
function CrossLinkList({
  label, emptyText, items,
}: {
  label: string;
  emptyText: string;
  items: { key: string; selected: boolean; content: ReactNode; trailing?: ReactNode; onClick: () => void }[];
}) {
  return (
    <div className="claim-verify-drawer-sources">
      <span>{label}</span>
      {items.length === 0 ? (
        <p className="claim-verify-drawer-no-sources">{emptyText}</p>
      ) : (
        <ul>
          {items.map((item) => (
            <li key={item.key}>
              <button type="button" className={item.selected ? "is-selected" : ""} onClick={item.onClick}>
                {item.content}
              </button>
              {item.trailing}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ClaimDrawerBody({
  runId, row, linkedSourceIds, onSelectSource,
}: {
  runId: string;
  row: ClaimRow;
  linkedSourceIds: Set<string>;
  onSelectSource: (addressId: string) => void;
}) {
  const status = row.result?.status;
  const meta = status ? claimStatusMeta(status) : undefined;
  const sourceIds = row.result?.sourceSpanIds ?? [];
  return (
    <div className="claim-verify-drawer-body">
      <p className="claim-verify-drawer-text">{row.text || "(text unavailable)"}</p>
      {meta && (
        <p className={`claim-verify-drawer-status is-${status}`}>
          <b>{meta.glyph}</b> {meta.label}
          <small>{meta.description}</small>
        </p>
      )}
      {row.result && <p className="claim-verify-drawer-method">METHOD: {methodLabel(row.result.method)}</p>}
      {status === "contradicted" && (
        <p className="claim-verify-drawer-contradiction">
          {row.result?.method.name === "numeric_or_date_mismatch"
            ? "A cited source states a different number or date."
            : "A cited source negates this claim directly."}
        </p>
      )}
      <CrossLinkList
        label={`CITED SOURCE${sourceIds.length === 1 ? "" : "S"}`}
        emptyText="No cited sources for this status."
        items={sourceIds.map((addressId) => ({
          key: addressId,
          selected: linkedSourceIds.has(addressId),
          content: addressId.slice(-10),
          trailing: <a href={spanHref(runId, addressId)}>view span</a>,
          onClick: () => onSelectSource(addressId),
        }))}
      />
    </div>
  );
}

function SourceDrawerBody({
  runId, addressId, rows, linkedClaimIndices, onSelectClaim,
}: {
  runId: string;
  addressId: string;
  rows: ClaimRow[];
  linkedClaimIndices: Set<number>;
  onSelectClaim: (index: number) => void;
}) {
  const citingClaims = rows.filter((row) => linkedClaimIndices.has(row.claim.index));
  return (
    <div className="claim-verify-drawer-body">
      <p className="claim-verify-drawer-source-id">{addressId}</p>
      <a href={spanHref(runId, addressId)}>view span</a>
      <CrossLinkList
        label={`CLAIM${citingClaims.length === 1 ? "" : "S"} CITING THIS SOURCE`}
        emptyText="No claims cite this source."
        items={citingClaims.map((row) => {
          const meta = row.result ? claimStatusMeta(row.result.status) : undefined;
          return {
            key: String(row.claim.index),
            selected: false,
            content: <>{meta && <b>{meta.glyph}</b>} {row.text || `claim #${row.claim.index + 1}`}</>,
            onClick: () => onSelectClaim(row.claim.index),
          };
        })}
      />
    </div>
  );
}
