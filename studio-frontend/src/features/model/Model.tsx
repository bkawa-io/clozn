import { useEffect, useMemo, useState } from "react";
import { EvidenceMark } from "../../components/EvidenceMark";
import {
  StateBoard,
  type StateBoardCapabilityFlag,
  type StateBoardCard,
} from "../../components/StateBoard";
import {
  TypedActionOffer,
  type TypedActionOfferProps,
} from "../../components/TypedActionOffer";
import type { RuntimeState } from "../../data/types";
import { SnapshotsPanel } from "../../panels/snapshots";
import {
  loadModelWorkspace,
  type EngineModel,
  type LocalModel,
  type ModelWorkspaceData,
} from "./api";

interface ModelProps {
  runtime: RuntimeState;
  inspectorOpen: boolean;
}

type RuntimeLoadState = "loading" | "reported" | "partial" | "unavailable";
type AbsenceState = "not_measured" | "unavailable";

const capabilityLabels: Record<string, string> = {
  attn_knockout: "Attention knockout",
  infill: "Infill",
  jlens: "J-lens",
  revise: "Revision",
  sae: "Sparse features",
  sampling: "Sampling",
  score_arms: "Arm scoring",
  state_stream: "State stream",
  steering: "Activation steering",
  streaming: "Token streaming",
};

const loadStateLabels: Record<RuntimeLoadState, string> = {
  loading: "READING",
  reported: "REPORTED",
  partial: "PARTIAL",
  unavailable: "UNAVAILABLE",
};

/**
 * These are deliberately blocked offers rather than dormant controls. The current Model workspace
 * routes do not expose the backing records or descriptors, so Runtime can say exactly what is absent
 * without pretending it knows a transaction the gateway has not described.
 */
const unavailableOffers: readonly TypedActionOfferProps[] = [
  {
    title: "Ollama adoption",
    absence: {
      state: "unavailable",
      label: "Adoption workflow unavailable",
      reason: "This Runtime surface receives no resolve, fidelity, dry-run, commit, or undo record for adoption.",
    },
    cost: "Unknown; no adoption plan or local installation transaction was reported.",
    preconditions: [
      "A local adoption plan must be exposed.",
      "A backed dry-run and commit descriptor must be supplied.",
    ],
    action: {
      availability: "blocked",
      label: "Adoption controls unavailable",
      blockerReason: "Studio cannot infer an adoption action from model inventory or engine health alone.",
    },
  },
  {
    title: "Privacy controls",
    absence: {
      state: "unavailable",
      label: "Privacy controls unavailable",
      reason: "No installation-level privacy tier or control record is supplied by the current Runtime routes.",
    },
    cost: "Unknown until a privacy policy and control descriptor are recorded.",
    preconditions: [
      "A privacy policy record must be exposed for this installation.",
      "A backed control descriptor must identify an allowed change.",
    ],
    action: {
      availability: "blocked",
      label: "Privacy controls unavailable",
      blockerReason: "The frontend has no privacy control route or action descriptor to call.",
    },
  },
];

function titleCase(value: string): string {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function capabilityLabel(id: string): string {
  return capabilityLabels[id] ?? titleCase(id);
}

function shortHash(value?: string): string {
  return value ? `${value.slice(0, 12)}…` : "Not reported";
}

function formatBytes(value?: number): string {
  if (value == null || !Number.isFinite(value)) return "Size not reported";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = Math.max(0, value);
  let unit = 0;
  while (amount >= 1000 && unit < units.length - 1) {
    amount /= 1000;
    unit += 1;
  }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : "Runtime workspace could not be loaded.";
}

function capabilityFlags(engine?: EngineModel): StateBoardCapabilityFlag[] {
  return Object.entries(engine?.capabilities ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([id, available]) => {
      const label = capabilityLabel(id);
      return available
        ? {
            id,
            label,
            available: true,
            note: `Reported available by /engine/health; this is an availability flag, not evidence that the capability ran.`,
          }
        : {
            id,
            label,
            available: false,
            note: "Reported unavailable by /engine/health.",
            reason: `${label} is reported unavailable by /engine/health; the response did not identify a more specific precondition.`,
          };
    });
}

function stateBoardCards(
  data: ModelWorkspaceData,
  flags: readonly StateBoardCapabilityFlag[],
  loading: boolean,
  engineReason: string,
): StateBoardCard[] {
  const cards: StateBoardCard[] = [{
    id: "engine-health",
    kind: "engine_health",
    title: "Engine health",
    description: "The existing health route names a serving engine, but it does not expose doctor output or a readiness receipt.",
    health: data.engine
      ? {
          status: "healthy",
          note: "A successful /engine/health response returned a serving-engine record. Deeper readiness is not reported here.",
        }
      : loading
        ? {
            status: "checking",
            note: "Reading the existing /engine/health route; no installation-health conclusion is available yet.",
          }
        : {
            status: "unavailable",
            reason: engineReason,
          },
  }];

  if (flags.length > 0) {
    cards.push({
      id: "capability-flags",
      kind: "capabilities",
      title: "Capability flags",
      description: "Other screens can cite these reported preconditions instead of failing silently at click time.",
      flags,
    });
  }

  return cards;
}

function RuntimeAbsence({
  label,
  reason,
  state = "unavailable",
  className,
}: {
  label: string;
  reason: string;
  state?: AbsenceState;
  className?: string;
}) {
  // EvidenceMark owns the four-state grammar. A Runtime-specific empty tile would make these absences
  // look unlike the same absence on a run or comparison surface.
  return (
    <div className={["runtime-absence", className].filter(Boolean).join(" ")} data-evidence-state={state}>
      {state === "not_measured" ? (
        <EvidenceMark variant="chip" state="not_measured" label={label} reason={reason} />
      ) : (
        <EvidenceMark variant="chip" state="unavailable" label={label} reason={reason} />
      )}
    </div>
  );
}

function ModelRecord({ model, engine }: { model: LocalModel; engine?: EngineModel }) {
  const serving = Boolean(engine?.model && model.path === engine.model);

  return (
    <article className="runtime-model-record" data-model-path={model.path}>
      <header>
        <span>{serving ? "Serving engine match" : "Local model record"}</span>
        <strong>{model.filename}</strong>
      </header>
      <dl>
        <div><dt>Quant</dt><dd>{model.quant ?? "Not reported"}</dd></div>
        <div><dt>Size</dt><dd>{formatBytes(model.sizeBytes)}</dd></div>
        <div><dt>SHA256</dt><dd>{shortHash(model.sha256)}</dd></div>
      </dl>
    </article>
  );
}

function SourceRow({
  label,
  endpoint,
  detail,
  absence,
}: {
  label: string;
  endpoint: string;
  detail?: string;
  absence?: { state: AbsenceState; reason: string };
}) {
  return (
    <li className="runtime-source-row">
      <div>
        <strong>{label}</strong>
        <code>{endpoint}</code>
      </div>
      {detail ? <p>{detail}</p> : absence && <RuntimeAbsence label={`${label} unavailable`} {...absence} />}
    </li>
  );
}

export function Model({ runtime, inspectorOpen }: ModelProps) {
  const [data, setData] = useState<ModelWorkspaceData>({ axes: [], errors: {} });
  const [loadState, setLoadState] = useState<RuntimeLoadState>("loading");
  const [fatalError, setFatalError] = useState<string>();

  useEffect(() => {
    const controller = new AbortController();
    void loadModelWorkspace(controller.signal)
      .then((next) => {
        if (controller.signal.aborted) return;
        setData(next);
        setFatalError(undefined);
        setLoadState(next.engine
          ? (Object.keys(next.errors).length > 0 ? "partial" : "reported")
          : "unavailable");
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setFatalError(errorText(error));
        setLoadState("unavailable");
      });
    return () => controller.abort();
  }, []);

  const loading = loadState === "loading";
  const engineReason = fatalError
    ?? data.errors.engine
    ?? "No engine record was received from /engine/health.";
  const flags = useMemo(() => capabilityFlags(data.engine), [data.engine]);
  const cards = useMemo(
    () => stateBoardCards(data, flags, loading, engineReason),
    [data, engineReason, flags, loading],
  );
  const activeAxes = data.axes.filter((axis) => Math.abs(axis.value) > 0.0001);
  const capabilityReason = loading
    ? "The engine-health request is still running, so no capability flags are available yet."
    : data.errors.engine
      ? data.errors.engine
      : "The /engine/health response did not include any capability flags.";
  const inventoryReason = data.errors.inventory
    ?? "The /models/local response did not provide a model inventory record.";
  const axesReason = data.errors.axes
    ?? "The /steer/axes response did not include steering-axis records.";

  return (
    <>
      <aside className="instrument runtime-map" aria-labelledby="runtime-map-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">INSTALLATION</span>
            <h2 id="runtime-map-title">Runtime</h2>
          </div>
          <strong data-runtime-load-state={loadState}>{loadStateLabels[loadState]}</strong>
        </header>

        <div className="runtime-map-copy">
          <p>The machine-level record: what the current Studio routes can establish about this installation.</p>
          <dl>
            <div><dt>Engine source</dt><dd>/engine/health</dd></div>
            <div><dt>Inventory source</dt><dd>/models/local</dd></div>
            <div><dt>Configuration</dt><dd>/steer/axes</dd></div>
          </dl>
        </div>

        <div className="runtime-map-boundary">
          <span>Truthfulness boundary</span>
          <p>These routes do not report resident capacity, qualification, adoption, or privacy controls.</p>
        </div>
      </aside>

      <section className="instrument runtime-main" aria-labelledby="runtime-title">
        <header className="instrument-head runtime-main-head">
          <div>
            <span className="eyebrow">THE MACHINE</span>
            <h1 id="runtime-title">Runtime</h1>
            <p>Installation-level state, with unavailable systems left visibly unavailable.</p>
          </div>
          <div className="runtime-main-status" data-runtime-load-state={loadState}>
            <span>WORKSPACE</span>
            <strong>{loadStateLabels[loadState]}</strong>
          </div>
        </header>

        <div className="runtime-scroll">
          <StateBoard cards={cards} title="Installation state" />

          {flags.length === 0 && (
            <RuntimeAbsence
              label="Capability flags unavailable"
              state={loading ? "not_measured" : "unavailable"}
              reason={capabilityReason}
              className="runtime-wide-absence"
            />
          )}

          <section className="runtime-section runtime-engine-record" aria-labelledby="runtime-engine-record-title">
            <header className="runtime-section-head">
              <div>
                <span>Reported engine</span>
                <h2 id="runtime-engine-record-title">Serving model record</h2>
              </div>
              <code>/engine/health</code>
            </header>
            {data.engine ? (
              <dl className="runtime-fact-grid">
                <div><dt>Model</dt><dd>{data.engine.modelName}</dd></div>
                <div><dt>Architecture</dt><dd>{data.engine.architecture}</dd></div>
                <div><dt>Quant</dt><dd>{data.engine.quant ?? "Not reported"}</dd></div>
                <div><dt>Device</dt><dd>{data.engine.device ?? "Not reported"}</dd></div>
                <div><dt>Context</dt><dd>{data.engine.context?.toLocaleString() ?? "Not reported"}</dd></div>
                <div><dt>GPU layers</dt><dd>{data.engine.gpuLayers?.toLocaleString() ?? "Not reported"}</dd></div>
                <div><dt>Protocol</dt><dd>{data.engine.protocolVersion ?? "Not reported"}</dd></div>
                <div><dt>Model SHA256</dt><dd>{shortHash(data.engine.sha256)}</dd></div>
              </dl>
            ) : (
              <RuntimeAbsence
                label="Serving engine unavailable"
                state={loading ? "not_measured" : "unavailable"}
                reason={loading ? "The /engine/health request has not completed." : engineReason}
              />
            )}
          </section>

          <section className="runtime-section" aria-labelledby="runtime-inventory-title">
            <header className="runtime-section-head">
              <div>
                <span>Reported files</span>
                <h2 id="runtime-inventory-title">Local model inventory</h2>
              </div>
              <code>/models/local</code>
            </header>
            {data.localModels === undefined ? (
              <RuntimeAbsence
                label="Model inventory unavailable"
                state={loading ? "not_measured" : "unavailable"}
                reason={loading ? "The /models/local request has not completed." : inventoryReason}
              />
            ) : data.localModels.length > 0 ? (
              <div className="runtime-model-grid">
                {data.localModels.map((model) => <ModelRecord model={model} engine={data.engine} key={model.path || model.filename} />)}
              </div>
            ) : (
              <RuntimeAbsence
                label="Model inventory not measured"
                state="not_measured"
                reason="The /models/local response contained no model records; Runtime cannot infer an empty installation or a resident-model cap."
              />
            )}
            <RuntimeAbsence
              label="Resident capacity unavailable"
              reason="The current inventory route lists model files but does not report worker residency or a hard capacity. No N-of-M meter is shown."
              className="runtime-capacity-boundary"
            />
          </section>

          <section className="runtime-section" aria-labelledby="runtime-configuration-title">
            <header className="runtime-section-head">
              <div>
                <span>Configuration evidence</span>
                <h2 id="runtime-configuration-title">Axes and omitted installation records</h2>
              </div>
              <a href="#/behavior">OPEN BEHAVIOR</a>
            </header>
            <div className="runtime-configuration-grid">
              <article className="runtime-configuration-record">
                <span>Steering axes</span>
                {loading ? (
                  <RuntimeAbsence label="Steering axes pending" state="not_measured" reason="The /steer/axes request has not completed." />
                ) : data.errors.axes ? (
                  <RuntimeAbsence label="Steering axes unavailable" reason={axesReason} />
                ) : data.axes.length === 0 ? (
                  <RuntimeAbsence label="Steering axes not measured" state="not_measured" reason={axesReason} />
                ) : (
                  <>
                    <p>{activeAxes.length} active of {data.axes.length} reported axes.</p>
                    {activeAxes.length > 0 && (
                      <ul className="runtime-axis-list" aria-label="Active steering axes">
                        {activeAxes.map((axis) => (
                          <li key={axis.name}>
                            <strong>{axis.name}</strong>
                            <span>{axis.value >= 0 ? "+" : ""}{axis.value.toFixed(2)} · {axis.calibrated ? "calibrated" : "not calibrated"}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </>
                )}
              </article>

              <article className="runtime-configuration-record">
                <span>Qualification</span>
                <RuntimeAbsence
                  label="Qualification record unavailable"
                  reason="The current Runtime routes do not supply qualification steps, product/lab boundaries, or a qualification receipt."
                />
              </article>

              <article className="runtime-configuration-record">
                <span>Correction scopes</span>
                <RuntimeAbsence
                  label="Correction scopes unavailable"
                  reason="The current Runtime routes do not supply correction-resolution or correction-scope records."
                />
              </article>
            </div>
          </section>

          <SnapshotsPanel runtime={runtime} embedded />

          <section className="runtime-section runtime-offers" aria-labelledby="runtime-offers-title">
            <header className="runtime-section-head">
              <div>
                <span>Not exposed by these routes</span>
                <h2 id="runtime-offers-title">Installation controls</h2>
              </div>
            </header>
            <div className="runtime-offer-grid">
              {unavailableOffers.map((offer) => <TypedActionOffer {...offer} key={offer.title} />)}
            </div>
          </section>
        </div>
      </section>

      {inspectorOpen && (
        <aside className="instrument runtime-inspector" aria-labelledby="runtime-inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">EVIDENCE BOUNDARY</span>
              <h2 id="runtime-inspector-title">Source ledger</h2>
            </div>
            <strong>READ ONLY</strong>
          </header>
          <ul className="runtime-source-list">
            <SourceRow
              label="Engine record"
              endpoint="/engine/health"
              detail={data.engine ? "Returned a serving-engine record." : undefined}
              absence={data.engine ? undefined : {
                state: loading ? "not_measured" : "unavailable",
                reason: loading ? "The request has not completed." : engineReason,
              }}
            />
            <SourceRow
              label="Local inventory"
              endpoint="/models/local"
              detail={data.localModels ? `${data.localModels.length} model record${data.localModels.length === 1 ? "" : "s"} returned.` : undefined}
              absence={data.localModels ? undefined : {
                state: loading ? "not_measured" : "unavailable",
                reason: loading ? "The request has not completed." : inventoryReason,
              }}
            />
            <SourceRow
              label="Steering axes"
              endpoint="/steer/axes"
              detail={!loading && !data.errors.axes && data.axes.length > 0 ? `${data.axes.length} axis record${data.axes.length === 1 ? "" : "s"} returned.` : undefined}
              absence={!loading && !data.errors.axes && data.axes.length > 0 ? undefined : {
                state: loading || !data.errors.axes ? "not_measured" : "unavailable",
                reason: loading ? "The request has not completed." : axesReason,
              }}
            />
          </ul>
        </aside>
      )}
    </>
  );
}
