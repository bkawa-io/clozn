import { useEffect, useState } from "react";
import { routeHref, type InspectRoute } from "../../app/router";
import { studioApi } from "../../data/client";
import { ContractError } from "../../data/contracts";
import type {
  ContextReceiptV1,
  ContextReceiptView,
} from "../../data/contextReceipt";
import "./contextReceipt.css";

type LoadState<T> =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "ready"; value: T }
  | { state: "unavailable"; message: string }
  | { state: "malformed"; message: string };

function failure(error: unknown): LoadState<never> {
  if (error instanceof ContractError) return { state: "malformed", message: error.message };
  return { state: "unavailable", message: error instanceof Error ? error.message : "Request failed." };
}

function shortId(value: string): string {
  return value.length > 24 ? `${value.slice(0, 11)}…${value.slice(-8)}` : value;
}

function countLabel(value: readonly unknown[] | undefined): string {
  return value === undefined ? "Not recorded" : value.length.toLocaleString();
}

function tokenLabel(receipt: ContextReceiptV1): string {
  const count = receipt.rendered?.tokens;
  if (count === undefined) return "Not recorded";
  return `${count.toLocaleString()}${receipt.rendered?.estimated ? " estimated" : " exact"}`;
}

function recordedCount(value: number | undefined): string {
  return value === undefined ? "Not recorded" : value.toLocaleString();
}

function ReceiptCapacityView({ receipt }: { receipt: ContextReceiptV1 }) {
  return (
    <dl className="receipt-limits">
      <div><dt>Prompt tokens</dt><dd>{recordedCount(receipt.rendered?.tokens)}</dd></div>
      <div><dt>Context window</dt><dd>{recordedCount(receipt.contextWindowTokens)}</dd></div>
      <div><dt>Reserved output</dt><dd>{recordedCount(receipt.reservedOutputTokens)}</dd></div>
      <div><dt>Generated</dt><dd>{recordedCount(receipt.termination?.generatedTokens)}</dd></div>
    </dl>
  );
}

function V1ReceiptView({ receipt }: { receipt: ContextReceiptV1 }) {
  return (
    <>
      <dl className="receipt-summary-grid">
        <div><dt>Stored privacy</dt><dd>{receipt.privacy?.replaceAll("_", " ") ?? "Not recorded"}</dd></div>
        <div><dt>Delivered</dt><dd>{countLabel(receipt.delivered)}</dd></div>
        <div><dt>Assembled</dt><dd>{countLabel(receipt.assembled)}</dd></div>
        <div><dt>Rendered tokens</dt><dd>{tokenLabel(receipt)}</dd></div>
        <div><dt>Termination</dt><dd>{receipt.termination?.reason.replaceAll("_", " ") ?? "Not recorded"}</dd></div>
        <div><dt>Rendered SHA256</dt><dd><code>{receipt.rendered?.sha256 ? shortId(receipt.rendered.sha256) : "Not recorded"}</code></dd></div>
      </dl>
      <ReceiptCapacityView receipt={receipt} />

      <section className="receipt-subsection" aria-labelledby="receipt-delivery-title">
        <div className="receipt-subsection-heading">
          <h3 id="receipt-delivery-title">Delivery ledger</h3>
          <span>{receipt.delivered === undefined ? "not recorded" : `${receipt.delivered.length} recorded`}</span>
        </div>
        {receipt.delivered === undefined ? (
          <p>Segment delivery was not recorded.</p>
        ) : receipt.delivered.length === 0 ? (
          <p>Recorded empty delivery list.</p>
        ) : (
          <ol className="receipt-segments">
            {receipt.delivered.map((segment) => (
              <li key={segment.segmentId}>
                <div>
                  <strong>{segment.sourceLabel ?? segment.clientSourceId ?? `Message ${segment.originalOrder + 1}`}</strong>
                  <span className={`state-${segment.included === true ? "included" : segment.included === false ? "omitted" : "unknown"}`}>
                    {segment.included === true ? "Included" : segment.included === false ? "Omitted" : "Inclusion not recorded"}
                  </span>
                </div>
                <code>{segment.segmentId}</code>
                <dl>
                  <div><dt>Order</dt><dd>{segment.originalOrder}</dd></div>
                  <div><dt>Bytes</dt><dd>{segment.deliveredBytes?.toLocaleString() ?? "Not recorded"}</dd></div>
                  <div><dt>Hash</dt><dd>{segment.contentHash ?? "Not recorded"}</dd></div>
                  <div><dt>Sources</dt><dd>{segment.sources === undefined ? "Not recorded" : segment.sources.length}</dd></div>
                </dl>
              </li>
            ))}
          </ol>
        )}
      </section>

      <div className="receipt-evidence-grid">
        <section className="receipt-subsection" aria-labelledby="receipt-transformations-title">
          <div className="receipt-subsection-heading">
            <h3 id="receipt-transformations-title">Transformations</h3>
            <span>{receipt.transformations === undefined ? "not recorded" : `${receipt.transformations.length} recorded`}</span>
          </div>
          {receipt.transformations === undefined ? <p>Not recorded.</p> : receipt.transformations.length === 0 ? <p>Recorded empty list.</p> : (
            <ul>{receipt.transformations.map((item, index) => <li key={`${item.reason}:${index}`}><code>{item.reason}</code>{item.detail && <span>{item.detail}</span>}</li>)}</ul>
          )}
        </section>
        <section className="receipt-subsection" aria-labelledby="receipt-omissions-title">
          <div className="receipt-subsection-heading">
            <h3 id="receipt-omissions-title">Omissions</h3>
            <span>{receipt.omissions === undefined ? "not recorded" : `${receipt.omissions.length} recorded`}</span>
          </div>
          {receipt.omissions === undefined ? <p>Not recorded.</p> : receipt.omissions.length === 0 ? <p>Recorded empty list.</p> : (
            <ul>{receipt.omissions.map((item, index) => <li key={`${item.segmentId}:${index}`}><code>{item.reason}</code><span>{item.segmentId}</span></li>)}</ul>
          )}
        </section>
      </div>
    </>
  );
}

function ReceiptBody({ view }: { view: ContextReceiptView }) {
  if (view.shape === "absent") {
    return <p className="receipt-primary-state" role="status">No Context Receipt was recorded for this Run.<span>{view.note}</span></p>;
  }
  if (view.shape === "unrecognized") {
    return (
      <p className="receipt-primary-state is-error" role="alert">
        Recorded receipt has an unrecognized shape.
        <span>{view.schemaVersionRaw ?? "No schema identity"} · {view.rawKeys.join(", ")}</span>
      </p>
    );
  }
  if (view.receipt.kind === "unsupported") {
    return (
      <p className="receipt-primary-state is-error" role="alert">
        Recorded receipt uses an unsupported schema.
        <span>{view.receipt.schemaVersion} · {view.receipt.rawKeys.join(", ")}</span>
      </p>
    );
  }
  return <V1ReceiptView receipt={view.receipt} />;
}

export function ContextReceiptSurface({ route }: { route: InspectRoute }) {
  const [receipt, setReceipt] = useState<LoadState<ContextReceiptView>>({ state: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setReceipt({ state: "loading" });
    void studioApi.contextReceipt(route.runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setReceipt({ state: "ready", value });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setReceipt(failure(error));
    });
    return () => controller.abort();
  }, [route.runId]);

  const metadata = receipt.state === "ready" ? receipt.value : undefined;
  const sessionHref = route.sessionId
    ? routeHref({ surface: "session", sessionId: route.sessionId, runId: route.runId })
    : routeHref({ surface: "sessions" });

  return (
    <div className="inspect-surface receipt-surface">
      <header className="inspect-heading">
        <div>
          <a className="back-link" href={sessionHref}>← {route.sessionId ? "Session" : "Sessions"}</a>
          <h1>Inspect</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {metadata && <span className={`receipt-shape state-${metadata.shape}`}>{metadata.shape}</span>}
      </header>

      <section className="inspect-instrument receipt-instrument" aria-labelledby="receipt-title">
        <header>
          <div><span>RECORDED INPUT</span><h2 id="receipt-title">Context receipt</h2></div>
          <code>GET only</code>
        </header>
        {receipt.state === "loading" && <p className="receipt-primary-state" role="status">Reading Context Receipt…</p>}
        {receipt.state === "malformed" && <p className="receipt-primary-state is-error" role="alert">Malformed Context Receipt contract.<span>{receipt.message}</span></p>}
        {receipt.state === "unavailable" && <p className="receipt-primary-state is-error" role="alert">Context Receipt unavailable.<span>{receipt.message}</span></p>}
        {receipt.state === "ready" && <ReceiptBody view={receipt.value} />}

      </section>

      <aside className="receipt-boundary">
        <strong>Read-only boundary</strong>
        <p>Navigation reads stored evidence only. It does not reconstruct a prompt, rerun a model, or change the Run.</p>
      </aside>
    </div>
  );
}
