import { useEffect, useState } from "react";
import {
  loadContextReceipt,
  type LegacyReceipt,
  type NewReceipt,
  type ReceiptSegment,
  type ReceiptView,
} from "../../data/context-receipt";

interface ContextReceiptProps {
  runId: string;
  /** Start with recorded facts expanded when the parent interaction is already an explicit reveal. */
  defaultDetailedOpen?: boolean;
  /** Reveal retained rendered text immediately after an explicit parent disclosure. */
  defaultAdvancedOpen?: boolean;
}

type LoadState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error" }
  | { status: "ready"; view: ReceiptView };

// ---------------------------------------------------------------------------------------------------
// reason-code taxonomy -- never collapse these into one word. Each code names WHO is responsible: the
// calling client, clozn's own assembly step, or the model/backend during generation. Mirrors the
// enums in clozn/schemas/defs/clozn.context-receipt.v1.json.
// ---------------------------------------------------------------------------------------------------

type ReasonCategory = "client" | "clozn" | "model" | "infra";

const CATEGORY_LABEL: Record<ReasonCategory, string> = {
  client: "CLIENT BEHAVIOR",
  clozn: "CLOZN ASSEMBLY",
  model: "MODEL",
  infra: "INFRASTRUCTURE",
};

const SEGMENT_REASON_INFO: Record<string, { label: string; category: ReasonCategory; live: boolean }> = {
  context_budget: { label: "Context budget", category: "clozn", live: false },
  client_not_delivered: { label: "Not delivered by the client", category: "client", live: false },
  history_policy: { label: "History-trimming policy", category: "clozn", live: false },
  attachment_limit: { label: "Attachment/document limit", category: "clozn", live: false },
  deduplicated: { label: "Deduplicated", category: "clozn", live: false },
  empty_content: { label: "Empty content", category: "clozn", live: false },
  unsupported_role: { label: "Unsupported role", category: "clozn", live: false },
  tool_result_pruned: { label: "Tool result pruned", category: "clozn", live: false },
  template_transformed: { label: "Rendered through the chat template", category: "clozn", live: true },
  system_message_merged: { label: "System message merged", category: "clozn", live: false },
  encoding_error: { label: "Encoding error", category: "infra", live: false },
};

const TERMINATION_REASON_INFO: Record<string, { label: string; category: ReasonCategory; live: boolean }> = {
  eos: { label: "Model reached a natural end of sequence", category: "model", live: true },
  stop_sequence: { label: "Stop sequence matched", category: "model", live: false },
  max_tokens: { label: "Requested output token cap reached", category: "model", live: true },
  context_limit: { label: "Context window reached", category: "model", live: true },
  client_cancelled: { label: "Client disconnected mid-stream", category: "client", live: true },
  timeout: { label: "Timed out before generation started", category: "infra", live: false },
  worker_error: { label: "Worker/engine error", category: "infra", live: true },
  tool_call: { label: "Model issued a tool call", category: "model", live: true },
  content_filter: { label: "Content filter", category: "infra", live: false },
  unknown: { label: "Unclassified", category: "infra", live: true },
};

function reasonInfo(reason: string | undefined, table: typeof SEGMENT_REASON_INFO) {
  if (!reason) return undefined;
  return table[reason] ?? { label: reason.replaceAll("_", " "), category: "infra" as ReasonCategory, live: true };
}

function reasonLabel(reason: string | undefined, table: typeof SEGMENT_REASON_INFO) {
  const info = reasonInfo(reason, table);
  return info ? info.label : "reason not recorded";
}

// ---------------------------------------------------------------------------------------------------
// formatting
// ---------------------------------------------------------------------------------------------------

function shortId(id: string | undefined) {
  if (!id) return "—";
  return id.length > 18 ? `${id.slice(0, 10)}…${id.slice(-4)}` : id;
}

function count(value: number | undefined) {
  return value == null ? "—" : value.toLocaleString();
}

function bytesText(value: number | undefined) {
  return value == null ? "—" : `${value.toLocaleString()} B`;
}

function boolText(value: boolean | undefined, whenTrue: string, whenFalse: string) {
  if (value == null) return "—";
  return value ? whenTrue : whenFalse;
}

// ---------------------------------------------------------------------------------------------------
// compact facts -- the five high-value facts near the response (spec: latest message included,
// system/project instructions included-or-omitted, document chunks, history omitted, termination).
// ---------------------------------------------------------------------------------------------------

interface CompactFact {
  id: string;
  label: string;
  value: string;
  detail?: string;
  state: "ok" | "warn" | "unavailable" | "na";
}

function latestSegment(segments: ReceiptSegment[]) {
  if (!segments.length) return undefined;
  return [...segments].sort((a, b) => (a.originalOrder ?? 0) - (b.originalOrder ?? 0)).at(-1);
}

function newCompactFacts(receipt: NewReceipt): CompactFact[] {
  const offTier = receipt.privacy === "off";
  const unavailableForPrivacy = (id: string, label: string): CompactFact => ({
    id,
    label,
    value: "UNAVAILABLE",
    detail: "receipt-privacy tier is off -- no content or metadata was retained for this run",
    state: "unavailable",
  });

  const latest = latestSegment(receipt.delivered);
  const latestFact: CompactFact = offTier
    ? unavailableForPrivacy("latest", "LATEST MESSAGE")
    : !latest
      ? { id: "latest", label: "LATEST MESSAGE", value: "NONE RECORDED", state: "unavailable" }
      : latest.included == null
        ? { id: "latest", label: "LATEST MESSAGE", value: "UNAVAILABLE", detail: "inclusion was not recorded for this segment", state: "unavailable" }
        : latest.included
          ? { id: "latest", label: "LATEST MESSAGE", value: "INCLUDED", detail: latest.sourceLabel ? `role: ${latest.sourceLabel}` : undefined, state: "ok" }
          : { id: "latest", label: "LATEST MESSAGE", value: "OMITTED", detail: reasonLabel(latest.reason, SEGMENT_REASON_INFO), state: "warn" };

  const systemSegments = receipt.delivered.filter((segment) => segment.sourceLabel === "system");
  const systemFact: CompactFact = offTier
    ? unavailableForPrivacy("system", "SYSTEM / PROJECT INSTRUCTIONS")
    : !systemSegments.length
      ? { id: "system", label: "SYSTEM / PROJECT INSTRUCTIONS", value: "NONE SENT", detail: "no system-role message was delivered for this run", state: "na" }
      : systemSegments.some((segment) => segment.included == null)
        ? { id: "system", label: "SYSTEM / PROJECT INSTRUCTIONS", value: "UNAVAILABLE", detail: "inclusion was not recorded", state: "unavailable" }
        : systemSegments.every((segment) => segment.included)
          ? { id: "system", label: "SYSTEM / PROJECT INSTRUCTIONS", value: "INCLUDED", state: "ok" }
          : systemSegments.every((segment) => !segment.included)
            ? { id: "system", label: "SYSTEM / PROJECT INSTRUCTIONS", value: "OMITTED", detail: reasonLabel(systemSegments[0]?.reason, SEGMENT_REASON_INFO), state: "warn" }
            : { id: "system", label: "SYSTEM / PROJECT INSTRUCTIONS", value: "PARTIALLY OMITTED", state: "warn" };

  const allSegments = [...receipt.delivered, ...receipt.assembled];
  const nonMessageSegments = allSegments.filter((segment) => segment.sourceType && segment.sourceType !== "message");
  const chunkFact: CompactFact = offTier
    ? unavailableForPrivacy("chunks", "DOCUMENT CHUNKS")
    : nonMessageSegments.length
      ? { id: "chunks", label: "DOCUMENT CHUNKS", value: String(nonMessageSegments.length), state: "ok" }
      : { id: "chunks", label: "DOCUMENT CHUNKS", value: "N/A", detail: "clozn has no document/attachment chunking -- every segment is a whole chat message", state: "na" };

  const ordered = [...receipt.delivered].sort((a, b) => (a.originalOrder ?? 0) - (b.originalOrder ?? 0));
  const history = ordered.slice(0, -1);
  const historyFact: CompactFact = offTier
    ? unavailableForPrivacy("history", "HISTORY OMITTED")
    : !history.length
      ? { id: "history", label: "HISTORY OMITTED", value: "N/A", detail: "only one delivered message -- no history to omit", state: "na" }
      : history.some((segment) => segment.included == null)
        ? { id: "history", label: "HISTORY OMITTED", value: "UNAVAILABLE", detail: "inclusion was not recorded", state: "unavailable" }
        : (() => {
            const omitted = history.filter((segment) => segment.included === false);
            return omitted.length
              ? {
                  id: "history",
                  label: "HISTORY OMITTED",
                  value: `${omitted.length} OF ${history.length}`,
                  detail: [...new Set(omitted.map((segment) => reasonLabel(segment.reason, SEGMENT_REASON_INFO)))].join("; "),
                  state: "warn" as const,
                }
              : { id: "history", label: "HISTORY OMITTED", value: `0 OF ${history.length}`, state: "ok" as const };
          })();

  const terminationInfo = reasonInfo(receipt.termination?.reason, TERMINATION_REASON_INFO);
  const terminationFact: CompactFact = !receipt.termination?.reason
    ? { id: "termination", label: "TERMINATION REASON", value: "NOT RECORDED", detail: "no finish signal was captured for this run", state: "unavailable" }
    : {
        id: "termination",
        label: "TERMINATION REASON",
        value: (terminationInfo?.label ?? receipt.termination.reason).toUpperCase(),
        detail: `${CATEGORY_LABEL[terminationInfo?.category ?? "infra"]} · raw: ${receipt.termination.reasonRaw ?? "—"}`,
        state: receipt.termination.reason === "eos" ? "ok" : "warn",
      };

  return [latestFact, systemFact, chunkFact, historyFact, terminationFact];
}

function legacyCompactFacts(receipt: LegacyReceipt): CompactFact[] {
  const unavailableLegacy = (id: string, label: string): CompactFact => ({
    id,
    label,
    value: "UNAVAILABLE",
    detail: "legacy receipt (pre-2026-07-27 schema) -- segment IDs and per-segment inclusion were never captured",
    state: "unavailable",
  });
  const chunkFact: CompactFact = {
    id: "chunks",
    label: "DOCUMENT CHUNKS",
    value: "N/A",
    detail: "clozn has no document/attachment chunking -- every segment is a whole chat message",
    state: "na",
  };
  const terminationFact: CompactFact = receipt.outputCutOff
    ? {
        id: "termination",
        label: "TERMINATION REASON",
        value: "OUTPUT TRUNCATED",
        detail: "legacy signal only -- the current termination vocabulary was not captured for this run",
        state: "warn",
      }
    : {
        id: "termination",
        label: "TERMINATION REASON",
        value: "NOT FLAGGED",
        detail: "legacy receipts only ever detect an output/context cutoff; no other reason is captured",
        state: "unavailable",
      };
  return [
    unavailableLegacy("latest", "LATEST MESSAGE"),
    unavailableLegacy("system", "SYSTEM / PROJECT INSTRUCTIONS"),
    chunkFact,
    unavailableLegacy("history", "HISTORY OMITTED"),
    terminationFact,
  ];
}

function CompactRow({ fact }: { fact: CompactFact }) {
  return (
    <div className={`context-receipt-fact is-${fact.state}`} title={fact.detail}>
      <span>{fact.label}</span>
      <strong>{fact.value}</strong>
      {fact.detail && <small>{fact.detail}</small>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------
// detailed sections
// ---------------------------------------------------------------------------------------------------

function SegmentTable({ title, segments }: { title: string; segments: ReceiptSegment[] }) {
  return (
    <section className="context-receipt-section">
      <header><span>{title}</span><b>{segments.length}</b></header>
      {segments.length ? (
        <div className="context-receipt-table" role="table">
          <div className="context-receipt-row is-head" role="row">
            <span>#</span><span>ROLE</span><span>SEGMENT</span><span>BYTES</span><span>INCLUDED</span><span>REASON</span><span>STATE</span>
          </div>
          {[...segments].sort((a, b) => (a.originalOrder ?? 0) - (b.originalOrder ?? 0)).map((segment, index) => (
            <div className="context-receipt-row" role="row" key={segment.segmentId ?? index}>
              <span>{segment.originalOrder ?? index}</span>
              <span>{segment.sourceLabel ?? "—"}</span>
              <span title={segment.clientSourceId ?? segment.segmentId}>
                {shortId(segment.clientSourceId ?? segment.segmentId)}
              </span>
              <span>{bytesText(segment.deliveredBytes)}</span>
              <span>{boolText(segment.included, "YES", "NO")}</span>
              <span title={reasonInfo(segment.reason, SEGMENT_REASON_INFO)?.label}>
                {segment.reason ? reasonLabel(segment.reason, SEGMENT_REASON_INFO) : "—"}
              </span>
              <span>{segment.redactionState ?? "—"}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="context-receipt-empty">NO SEGMENTS RECORDED</div>
      )}
    </section>
  );
}

function NewDetail({ receipt, defaultAdvancedOpen = false }: { receipt: NewReceipt; defaultAdvancedOpen?: boolean }) {
  const [advancedOpen, setAdvancedOpen] = useState(defaultAdvancedOpen);
  const estimatedBadge = receipt.rendered?.estimated == null
    ? undefined
    : receipt.rendered.estimated ? "ESTIMATED" : "EXACT";

  return (
    <div className="context-receipt-detail">
      {receipt.templateFingerprint && (
        <section className="context-receipt-section">
          <header><span>TEMPLATE FINGERPRINT</span></header>
          <p className="context-receipt-note">
            <code>{receipt.templateFingerprint}</code>
            {receipt.tokenizerConflatedWithTemplate && (
              <span className="context-receipt-caveat">
                fingerprints template AND tokenizer rendering jointly -- a change here cannot be attributed to
                one or the other alone
              </span>
            )}
          </p>
        </section>
      )}

      <SegmentTable title="REQUEST AS DELIVERED" segments={receipt.delivered} />
      <SegmentTable title="CONTEXT AS ASSEMBLED" segments={receipt.assembled} />

      <section className="context-receipt-section">
        <header><span>RENDERED-PROMPT METADATA</span></header>
        {receipt.rendered ? (
          <dl className="context-receipt-facts">
            <div><dt>SHA-256</dt><dd title={receipt.rendered.sha256}>{shortId(receipt.rendered.sha256)}</dd></div>
            <div><dt>BYTES</dt><dd>{count(receipt.rendered.bytes)}</dd></div>
            <div>
              <dt>TOKEN COUNT</dt>
              <dd>
                {count(receipt.rendered.tokens ?? receipt.rendered.tokenCount)}
                {estimatedBadge && <span className={`context-receipt-badge is-${estimatedBadge.toLowerCase()}`}>{estimatedBadge}</span>}
              </dd>
            </div>
            <div>
              <dt>CONTENT AVAILABLE</dt>
              <dd>{boolText(receipt.rendered.contentAvailable, "YES", "NO")}</dd>
            </div>
            <div>
              <dt>BOUND TEMPLATE</dt>
              <dd title={receipt.rendered.templateFingerprint}>{shortId(receipt.rendered.templateFingerprint)}</dd>
            </div>
            <div>
              <dt>SPECIAL TOKENS</dt>
              <dd>{receipt.rendered.specialTokens?.length ? receipt.rendered.specialTokens.join(", ") : "not captured in v1"}</dd>
            </div>
          </dl>
        ) : (
          <div className="context-receipt-empty">NO RENDERED-PROMPT METADATA RECORDED</div>
        )}
      </section>

      <section className="context-receipt-section">
        <header><span>TRANSFORMATIONS</span><b>{receipt.transformations.length}</b></header>
        {receipt.transformations.length ? (
          <ul className="context-receipt-list">
            {receipt.transformations.map((transformation, index) => {
              const info = reasonInfo(transformation.reason, SEGMENT_REASON_INFO);
              return (
                <li key={`${transformation.reason}-${index}`}>
                  <strong>{info?.label ?? transformation.reason ?? "—"}</strong>
                  <span className="context-receipt-category">{CATEGORY_LABEL[info?.category ?? "infra"]}</span>
                  <span>{transformation.segmentIds.length} segment(s)</span>
                  {transformation.detail && <small>{transformation.detail}</small>}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="context-receipt-empty">NO TRANSFORMATIONS RECORDED</div>
        )}
      </section>

      <section className="context-receipt-section">
        <header><span>OMITTED ITEMS</span><b>{receipt.omissions.length}</b></header>
        {receipt.omissions.length ? (
          <ul className="context-receipt-list">
            {receipt.omissions.map((omission, index) => {
              const info = reasonInfo(omission.reason, SEGMENT_REASON_INFO);
              return (
                <li key={`${omission.segmentId}-${index}`}>
                  <strong title={omission.segmentId}>{shortId(omission.segmentId)}</strong>
                  <span className="context-receipt-category">{CATEGORY_LABEL[info?.category ?? "infra"]}</span>
                  <span>{info?.label ?? omission.reason ?? "reason not recorded"}</span>
                  {omission.detail && <small>{omission.detail}</small>}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="context-receipt-empty">NO OMISSIONS WERE RECORDED FOR THIS RUN</div>
        )}
      </section>

      <section className="context-receipt-section">
        <header><span>TOKEN BUDGET</span></header>
        <dl className="context-receipt-facts">
          <div><dt>CONTEXT WINDOW</dt><dd>{count(receipt.contextWindowTokens ?? receipt.limits.contextWindowTokens)}</dd></div>
          <div><dt>RESERVED OUTPUT</dt><dd>{count(receipt.reservedOutputTokens ?? receipt.limits.requestedMaxTokens)}</dd></div>
          <div><dt>PROMPT TOKENS</dt><dd>{count(receipt.limits.promptTokens)}</dd></div>
          <div><dt>GENERATED TOKENS</dt><dd>{count(receipt.termination?.generatedTokens ?? receipt.limits.generatedTokens)}</dd></div>
        </dl>
      </section>

      <section className="context-receipt-section">
        <header><span>TERMINATION DETAILS</span></header>
        {receipt.termination?.reason ? (
          <dl className="context-receipt-facts">
            <div><dt>NORMALIZED REASON</dt><dd>{reasonInfo(receipt.termination.reason, TERMINATION_REASON_INFO)?.label ?? receipt.termination.reason}</dd></div>
            <div><dt>CATEGORY</dt><dd>{CATEGORY_LABEL[reasonInfo(receipt.termination.reason, TERMINATION_REASON_INFO)?.category ?? "infra"]}</dd></div>
            <div><dt>RAW BACKEND VALUE</dt><dd>{receipt.termination.reasonRaw ?? "—"}</dd></div>
            <div><dt>RAW VALUE SOURCE</dt><dd>{receipt.termination.source ?? "—"}</dd></div>
            <div><dt>GENERATED TOKENS</dt><dd>{count(receipt.termination.generatedTokens)}</dd></div>
          </dl>
        ) : (
          <div className="context-receipt-empty">NO TERMINATION SIGNAL WAS CAPTURED FOR THIS RUN</div>
        )}
      </section>

      {receipt.privacy && (
        <section className="context-receipt-section">
          <header><span>RECEIPT PRIVACY TIER</span></header>
          <p className="context-receipt-note">{receipt.privacy.toUpperCase().replaceAll("_", " ")}</p>
        </section>
      )}

      <section className="context-receipt-section is-advanced">
        <header>
          <button type="button" onClick={() => setAdvancedOpen((open) => !open)} aria-expanded={advancedOpen}>
            <span>ADVANCED</span>
            <small>{advancedOpen ? "HIDE" : "SHOW"} RAW RENDERED PROMPT</small>
          </button>
        </header>
        {advancedOpen && (
          <div className="context-receipt-advanced-body">
            {receipt.legacyFinalPrompt ? (
              <>
                <p className="context-receipt-caveat">
                  contains the literal rendered template, including any special-token syntax the model's chat
                  template inserts
                </p>
                <pre className="context-receipt-raw">{receipt.legacyFinalPrompt}</pre>
              </>
            ) : (
              <div className="context-receipt-empty">
                {receipt.contentWithheldByRequest || receipt.contentWithheldByPrivacyTier
                  ? "RENDERED PROMPT TEXT WAS WITHHELD (privacy tier or read-time request)"
                  : "NO RENDERED PROMPT TEXT WAS RETAINED FOR THIS RECEIPT"}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function MessageList({ title, messages }: { title: string; messages: Array<{ role: string; content?: string }> }) {
  return (
    <section className="context-receipt-section">
      <header><span>{title}</span><b>{messages.length}</b></header>
      {messages.length ? (
        <ul className="context-receipt-list is-messages">
          {messages.map((message, index) => (
            <li key={index}>
              <strong>{message.role}</strong>
              <p>{message.content ?? "(content withheld)"}</p>
            </li>
          ))}
        </ul>
      ) : (
        <div className="context-receipt-empty">NONE RECORDED</div>
      )}
    </section>
  );
}

function LegacyDetail({ receipt, defaultAdvancedOpen = false }: { receipt: LegacyReceipt; defaultAdvancedOpen?: boolean }) {
  const [advancedOpen, setAdvancedOpen] = useState(defaultAdvancedOpen);
  return (
    <div className="context-receipt-detail">
      <p className="context-receipt-caveat context-receipt-legacy-banner">
        legacy receipt shape (pre-2026-07-27 schema) -- segment IDs and normalized termination reasons are
        unavailable for this run; the sections below show only what this shape actually captured
      </p>

      <MessageList title="REQUEST AS DELIVERED" messages={receipt.deliveredMessages} />
      <MessageList title="CONTEXT AS ASSEMBLED" messages={receipt.assembledMessages} />

      <section className="context-receipt-section">
        <header><span>OMITTED ITEMS</span></header>
        <div className="context-receipt-empty">NOT CAPTURED BY THIS RECEIPT SHAPE</div>
      </section>

      <section className="context-receipt-section">
        <header><span>TOKEN BUDGET</span></header>
        <dl className="context-receipt-facts">
          <div><dt>CONTEXT WINDOW</dt><dd>{count(receipt.limits.contextWindowTokens)}</dd></div>
          <div><dt>REQUESTED MAX</dt><dd>{count(receipt.limits.requestedMaxTokens)}</dd></div>
          <div><dt>PROMPT TOKENS</dt><dd>{count(receipt.limits.promptTokens)}</dd></div>
          <div><dt>GENERATED TOKENS</dt><dd>{count(receipt.limits.generatedTokens)}</dd></div>
        </dl>
      </section>

      <section className="context-receipt-section">
        <header><span>TERMINATION DETAILS (LEGACY)</span></header>
        <dl className="context-receipt-facts">
          <div><dt>OUTPUT CUT OFF</dt><dd>{boolText(receipt.outputCutOff, "YES", "NO")}</dd></div>
          <div><dt>INPUT TRUNCATED</dt><dd>{boolText(receipt.inputTruncated, "YES", "NO")}</dd></div>
          <div><dt>INPUT POLICY</dt><dd>{receipt.inputPolicy ?? "—"}</dd></div>
        </dl>
        {receipt.warnings.length > 0 && (
          <ul className="context-receipt-list">
            {receipt.warnings.map((warning, index) => (
              <li key={index}>
                <strong>{warning.code}</strong>
                {warning.severity && <span className="context-receipt-category">{warning.severity.toUpperCase()}</span>}
                {warning.message && <small>{warning.message}</small>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="context-receipt-section is-advanced">
        <header>
          <button type="button" onClick={() => setAdvancedOpen((open) => !open)} aria-expanded={advancedOpen}>
            <span>ADVANCED</span>
            <small>{advancedOpen ? "HIDE" : "SHOW"} RAW RENDERED PROMPT</small>
          </button>
        </header>
        {advancedOpen && (
          receipt.finalPrompt ? (
            <>
              <p className="context-receipt-caveat">
                contains the literal rendered template, including any special-token syntax the model's chat
                template inserts
              </p>
              <pre className="context-receipt-raw">{receipt.finalPrompt}</pre>
            </>
          ) : (
            <div className="context-receipt-empty">
              {receipt.contentWithheldByRequest ? "RENDERED PROMPT TEXT WAS WITHHELD (read-time request)" : "NO RENDERED PROMPT TEXT WAS RETAINED"}
            </div>
          )
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------
// top-level card
// ---------------------------------------------------------------------------------------------------

export function ContextReceipt({
  runId,
  defaultDetailedOpen = false,
  defaultAdvancedOpen = false,
}: ContextReceiptProps) {
  const [state, setState] = useState<LoadState>({ status: "idle" });
  const [detailedOpen, setDetailedOpen] = useState(defaultDetailedOpen);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setState({ status: "loading" });
    setDetailedOpen(defaultDetailedOpen);
    void loadContextReceipt(runId, controller.signal).then((view) => {
      if (!controller.signal.aborted) setState({ status: "ready", view });
    }).catch(() => {
      if (!controller.signal.aborted) setState({ status: "error" });
    });
    return () => controller.abort();
  }, [defaultDetailedOpen, runId]);

  const view = state.status === "ready" ? state.view : null;

  return (
    <section className="context-receipt" aria-labelledby="context-receipt-title">
      <header className="context-receipt-head">
        <div>
          <span className="eyebrow">EVIDENCE</span>
          <h3 id="context-receipt-title">Context receipt</h3>
        </div>
        {view && view.shape !== "absent" && (
          <span className={`context-receipt-shape is-${view.shape}`}>{view.shape.toUpperCase()}</span>
        )}
      </header>

      {state.status === "idle" || state.status === "loading" ? (
        <div className="context-receipt-empty">LOADING CONTEXT RECEIPT</div>
      ) : state.status === "error" || !view ? (
        <div className="context-receipt-empty is-error">CONTEXT RECEIPT REQUEST FAILED</div>
      ) : view.shape === "absent" ? (
        <div className="context-receipt-empty">NO CONTEXT RECEIPT WAS RECORDED FOR THIS RUN</div>
      ) : view.shape === "unrecognized" ? (
        <div className="context-receipt-empty is-error">
          CONTEXT RECEIPT HAS AN UNRECOGNIZED SHAPE
          <small>
            schema field: {view.schemaVersionRaw ?? "(none)"} · keys: {view.rawKeys.join(", ") || "(none)"}
          </small>
        </div>
      ) : (
        <>
          <div className="context-receipt-facts-strip">
            {(view.shape === "new" ? newCompactFacts(view.receipt) : legacyCompactFacts(view.receipt))
              .map((fact) => <CompactRow fact={fact} key={fact.id} />)}
          </div>
          <button
            type="button"
            className="context-receipt-toggle"
            aria-expanded={detailedOpen}
            onClick={() => setDetailedOpen((open) => !open)}
          >{detailedOpen ? "HIDE DETAILED VIEW" : "SHOW DETAILED VIEW"}</button>
          {detailedOpen && (
            view.shape === "new"
              ? <NewDetail receipt={view.receipt} defaultAdvancedOpen={defaultAdvancedOpen} />
              : <LegacyDetail receipt={view.receipt} defaultAdvancedOpen={defaultAdvancedOpen} />
          )}
        </>
      )}
    </section>
  );
}
