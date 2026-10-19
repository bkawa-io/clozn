import { useEffect, useRef, useState } from "react";
import { describeError, isContractError, studioApi } from "../../data/client";
import type { RunDetails } from "../../data/contracts";
import type { ContextReceiptView, ContextReceiptV1 } from "../../data/contextReceipt";
import "./contextReceipt.css";

type DetailState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "ready"; run: RunDetails; receipt: ContextReceiptView }
  | { phase: "unavailable" | "malformed"; message: string };

function text(value: unknown): string {
  if (typeof value === "string" && value.trim()) return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return "Not recorded";
}

function v1Receipt(view: ContextReceiptView): ContextReceiptV1 | undefined {
  return view.shape === "recorded" && view.receipt.kind === "v1" ? view.receipt : undefined;
}

export function RunDetailsDisclosure({ runId }: { runId: string }) {
  const [state, setState] = useState<DetailState>({ phase: "idle" });
  const controllerRef = useRef<AbortController | undefined>(undefined);
  useEffect(() => {
    controllerRef.current?.abort();
    controllerRef.current = undefined;
    setState({ phase: "idle" });
    return () => controllerRef.current?.abort();
  }, [runId]);

  const load = () => {
    if (state.phase !== "idle") return;
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ phase: "loading" });
    void Promise.all([
      studioApi.run(runId, controller.signal),
      studioApi.contextReceipt(runId, controller.signal),
    ]).then(([run, receipt]) => {
      if (!controller.signal.aborted) setState({ phase: "ready", run, receipt });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setState({ phase: isContractError(error) ? "malformed" : "unavailable", message: describeError(error) });
      }
    });
  };

  const receipt = state.phase === "ready" ? v1Receipt(state.receipt) : undefined;
  const identity = state.phase === "ready" ? state.run.identity : {};
  return (
    <details className="run-details-disclosure" onToggle={(event) => {
      if (event.currentTarget.open) load();
    }}>
      <summary>Context, provenance, and runtime details</summary>
      {state.phase === "idle" && <p>Open to read stored details.</p>}
      {state.phase === "loading" && <p role="status">Reading stored details…</p>}
      {(state.phase === "unavailable" || state.phase === "malformed") && (
        <p role="alert">Stored details unavailable.<span>{state.message}</span></p>
      )}
      {state.phase === "ready" && (
        <div className="run-details-body">
          <dl>
            <div><dt>Model</dt><dd>{state.run.model ?? "Not recorded"}</dd></div>
            <div><dt>Runtime</dt><dd>{state.run.substrate ?? "Not recorded"}</dd></div>
            <div><dt>Engine build</dt><dd><code>{text(identity.engine_build)}</code></dd></div>
            <div><dt>Model digest</dt><dd><code>{text(identity.model_sha256)}</code></dd></div>
            <div><dt>Template</dt><dd><code>{receipt?.templateFingerprint ?? text(identity.template_fingerprint)}</code></dd></div>
            <div><dt>Prompt tokens</dt><dd>{receipt?.rendered?.tokens ?? "Not recorded"}</dd></div>
            <div><dt>Context window</dt><dd>{receipt?.contextWindowTokens ?? "Not recorded"}</dd></div>
            <div><dt>Termination</dt><dd>{receipt?.termination?.reason.replaceAll("_", " ") ?? "Not recorded"}</dd></div>
          </dl>
          <section aria-labelledby="recorded-context-title">
            <h3 id="recorded-context-title">Received context</h3>
            {!receipt?.delivered ? <p>Delivery metadata was not recorded.</p> : (
              <ol>
                {receipt.delivered.map((segment) => (
                  <li key={segment.segmentId}>
                    <strong>{segment.sourceLabel ?? `Message ${segment.originalOrder + 1}`}</strong>
                    <span>{segment.sources?.length ?? 0} source{segment.sources?.length === 1 ? "" : "s"} · {segment.deliveredBytes ?? "?"} bytes</span>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </div>
      )}
    </details>
  );
}
