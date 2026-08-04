import { EvidenceMark } from "./EvidenceMark";
import "./StateBoard.css";

/**
 * C7 -- cards answer one deliberately narrow question: what exists, and what condition is it in?
 * These are presentation records rather than runtime-route records. A runtime adapter can preserve its
 * own transport schema at the boundary while this component remains usable for managed workers,
 * qualification receipts, and correction eligibility without inheriting any one of their lifecycles.
 */

export interface StateBoardHardCap {
  /** Recorded numerator. It is shown verbatim even if it exceeds `limit`; only the visual ratio clamps. */
  used: number;
  /** Recorded hard maximum, not a soft completion target. */
  limit: number;
  /** The thing constrained by this cap, for example "resident workers". */
  label?: string;
}

interface StateBoardAvailable {
  available: true;
  /** A citeable explanation of the available condition, not merely a yes/no flag. */
  note: string;
  reason?: never;
}

interface StateBoardUnavailable {
  available: false;
  /** An unavailable capability or action is an explained absence, never an implicit false. */
  note: string;
  reason: string;
}

export type StateBoardAvailability = StateBoardAvailable | StateBoardUnavailable;

interface StateBoardWorkerBase {
  id: string;
  label: string;
}

export type StateBoardWorker =
  | (StateBoardWorkerBase & {
      status: "resident" | "busy" | "starting";
      /** Present workers still carry a visible condition for an operator to cite. */
      note: string;
      reason?: never;
    })
  | (StateBoardWorkerBase & {
      status: "unavailable";
      reason: string;
      note?: never;
    });

export type StateBoardEngineHealth =
  | { status: "healthy"; note: string; reason?: never }
  | { status: "degraded"; note: string; reason?: never }
  | { status: "checking"; note: string; reason?: never }
  | { status: "unavailable"; reason: string; note?: never };

interface StateBoardQualificationStepBase {
  id: string;
  label: string;
}

/**
 * Qualification is a checklist, not a completion percentage. In particular, `partial` records work
 * that happened but did not clear the whole gate, while `blocked` and `not_run` record different
 * absences and therefore require their own reasons at the type boundary.
 */
export type StateBoardQualificationStep =
  | (StateBoardQualificationStepBase & {
      status: "passed" | "failed" | "partial";
      note: string;
      reason?: never;
    })
  | (StateBoardQualificationStepBase & {
      status: "blocked" | "not_run";
      reason: string;
      note?: never;
    });

export type StateBoardCapabilityFlag = StateBoardAvailability & {
  id: string;
  label: string;
};

export type StateBoardCorrectionScope = StateBoardAvailability & {
  id: string;
  label: string;
};

interface StateBoardCardBase {
  id: string;
  /** Card headings are optional because the kind has a stable, readable fallback. */
  title?: string;
  /** Optional scope/context; individual conditions keep their own required evidence where needed. */
  description?: string;
}

export interface StateBoardResidentWorkersCard extends StateBoardCardBase {
  kind: "resident_workers";
  capacity: StateBoardHardCap;
  workers: readonly StateBoardWorker[];
}

export interface StateBoardEngineHealthCard extends StateBoardCardBase {
  kind: "engine_health";
  health: StateBoardEngineHealth;
  /** Engine-local flags can sit beside health without coupling either to `/engine/health`. */
  capabilities?: readonly StateBoardCapabilityFlag[];
}

export interface StateBoardQualificationCard extends StateBoardCardBase {
  kind: "qualification";
  steps: readonly StateBoardQualificationStep[];
}

export interface StateBoardCorrectionScopesCard extends StateBoardCardBase {
  kind: "correction_scopes";
  scopes: readonly StateBoardCorrectionScope[];
}

export interface StateBoardCapabilitiesCard extends StateBoardCardBase {
  kind: "capabilities";
  flags: readonly StateBoardCapabilityFlag[];
}

export type StateBoardCard =
  | StateBoardResidentWorkersCard
  | StateBoardEngineHealthCard
  | StateBoardQualificationCard
  | StateBoardCorrectionScopesCard
  | StateBoardCapabilitiesCard;

export interface StateBoardProps {
  cards: readonly StateBoardCard[];
  title?: string;
  className?: string;
}

const MAX_VISUAL_CAPACITY_SLOTS = 12;

function nonNegativeWhole(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;
}

function readableKind(kind: StateBoardCard["kind"]): string {
  switch (kind) {
    case "resident_workers": return "Resident workers";
    case "engine_health": return "Engine health";
    case "qualification": return "Qualification";
    case "correction_scopes": return "Correction scopes";
    case "capabilities": return "Capabilities";
    default: {
      const exhaustive: never = kind;
      return exhaustive;
    }
  }
}

function workerStatusLabel(status: StateBoardWorker["status"]): string {
  switch (status) {
    case "resident": return "Resident";
    case "busy": return "Busy";
    case "starting": return "Starting";
    case "unavailable": return "Unavailable";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

function engineStatusLabel(status: StateBoardEngineHealth["status"]): string {
  switch (status) {
    case "healthy": return "Healthy";
    case "degraded": return "Degraded";
    case "checking": return "Checking";
    case "unavailable": return "Unavailable";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

function qualificationStatusLabel(status: StateBoardQualificationStep["status"]): string {
  switch (status) {
    case "passed": return "Passed";
    case "failed": return "Failed";
    case "partial": return "Partial";
    case "blocked": return "Blocked";
    case "not_run": return "Not run";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

function stateClassName(status: string): string {
  return `is-${status.replaceAll("_", "-")}`;
}

function HardCapMeter({ capacity }: { capacity: StateBoardHardCap }) {
  const used = nonNegativeWhole(capacity.used);
  const limit = nonNegativeWhole(capacity.limit);
  const boundedUsed = Math.min(used, limit);
  const slotCount = Math.min(limit, MAX_VISUAL_CAPACITY_SLOTS);
  // The slots are a bounded ratio for large caps; the explicit numerator and denominator below remain
  // authoritative, so a compact meter can never hide an over-cap condition or pretend to be a task.
  const filledSlotCount = limit > 0 && boundedUsed > 0
    ? Math.max(1, Math.ceil((boundedUsed / limit) * slotCount))
    : 0;
  const overCap = used > limit;
  const label = capacity.label ?? "Resident workers";
  const capacityText = `${used.toLocaleString()} of ${limit.toLocaleString()}`;

  return (
    <div className="state-board-hard-cap" data-over-cap={String(overCap)}>
      <div
        className="state-board-capacity-meter"
        data-filled-slots={filledSlotCount}
        data-slot-count={slotCount}
        role="img"
        aria-label={`${label}: ${capacityText}; hard cap${overCap ? " exceeded" : ""}.`}
      >
        {slotCount > 0 ? Array.from({ length: slotCount }, (_, index) => (
          <span
            className={[
              "state-board-capacity-slot",
              index < filledSlotCount ? "is-filled" : "is-empty",
            ].join(" ")}
            aria-hidden="true"
            key={index}
          />
        )) : <span className="state-board-capacity-none">No capacity slots configured</span>}
      </div>
      <div className="state-board-capacity-copy">
        <strong>{capacityText}</strong>
        <span>{label} · hard cap</span>
        {overCap && <b>Hard cap exceeded</b>}
      </div>
    </div>
  );
}

function WorkerRow({ worker }: { worker: StateBoardWorker }) {
  const statusLabel = workerStatusLabel(worker.status);

  return (
    <li
      className={`state-board-worker ${stateClassName(worker.status)}`}
      data-worker-id={worker.id}
      data-status={worker.status}
    >
      <span className={`state-board-worker-form ${stateClassName(worker.status)}`} aria-hidden="true" />
      <div className="state-board-item-copy">
        <strong>{worker.label}</strong>
        {worker.status === "unavailable" ? (
          <EvidenceMark variant="chip" state="unavailable" label={statusLabel} reason={worker.reason} />
        ) : (
          <>
            <span className={`state-board-status-label ${stateClassName(worker.status)}`}>{statusLabel}</span>
            <p>{worker.note}</p>
          </>
        )}
      </div>
    </li>
  );
}

function EngineHealth({ health }: { health: StateBoardEngineHealth }) {
  const statusLabel = engineStatusLabel(health.status);

  if (health.status === "unavailable") {
    return (
      <div className="state-board-engine-condition is-unavailable" data-status={health.status}>
        <EvidenceMark variant="chip" state="unavailable" label="Engine unavailable" reason={health.reason} />
      </div>
    );
  }

  return (
    <div className={`state-board-engine-condition ${stateClassName(health.status)}`} data-status={health.status}>
      <span className={`state-board-condition-form ${stateClassName(health.status)}`} aria-hidden="true" />
      <div className="state-board-item-copy">
        <span className={`state-board-status-label ${stateClassName(health.status)}`}>{statusLabel}</span>
        <p>{health.note}</p>
      </div>
    </div>
  );
}

function QualificationStepRow({ step }: { step: StateBoardQualificationStep }) {
  const statusLabel = qualificationStatusLabel(step.status);
  const absenceState = step.status === "not_run" ? "not_measured" : "unavailable";

  return (
    <li
      className={`state-board-qualification-step ${stateClassName(step.status)}`}
      data-qualification-step={step.id}
      data-status={step.status}
    >
      <div className="state-board-qualification-step-head">
        <strong>{step.label}</strong>
        {step.status === "blocked" || step.status === "not_run" ? (
          <EvidenceMark variant="chip" state={absenceState} label={statusLabel} reason={step.reason} />
        ) : (
          <span className={`state-board-qualification-status ${stateClassName(step.status)}`}>
            <span className={`state-board-qualification-form ${stateClassName(step.status)}`} aria-hidden="true" />
            {statusLabel}
          </span>
        )}
      </div>
      {step.status === "passed" || step.status === "failed" || step.status === "partial"
        ? <p>{step.note}</p>
        : null}
    </li>
  );
}

function AvailabilityIndicator({ availability }: { availability: StateBoardAvailability }) {
  if (!availability.available) {
    return <EvidenceMark variant="chip" state="unavailable" label="Unavailable" reason={availability.reason} />;
  }

  return (
    <span className="state-board-availability is-available">
      <span className="state-board-availability-form" aria-hidden="true" />
      Available
    </span>
  );
}

function CapabilityList({ flags }: { flags: readonly StateBoardCapabilityFlag[] }) {
  if (flags.length === 0) return <p className="state-board-empty">No capability flags were supplied.</p>;

  return (
    <ul className="state-board-availability-list" aria-label="Capability flags">
      {flags.map((flag) => (
        <li
          className={`state-board-availability-row ${flag.available ? "is-available" : "is-unavailable"}`}
          data-capability-id={flag.id}
          data-available={String(flag.available)}
          key={flag.id}
        >
          <div className="state-board-availability-row-head">
            <strong>{flag.label}</strong>
            <AvailabilityIndicator availability={flag} />
          </div>
          <p>{flag.note}</p>
        </li>
      ))}
    </ul>
  );
}

function CorrectionScopeList({ scopes }: { scopes: readonly StateBoardCorrectionScope[] }) {
  if (scopes.length === 0) return <p className="state-board-empty">No correction scopes were supplied.</p>;

  return (
    <ul className="state-board-availability-list" aria-label="Correction scopes">
      {scopes.map((scope) => (
        <li
          className={`state-board-availability-row ${scope.available ? "is-available" : "is-unavailable"}`}
          data-correction-scope={scope.id}
          data-available={String(scope.available)}
          key={scope.id}
        >
          <div className="state-board-availability-row-head">
            <strong>{scope.label}</strong>
            <AvailabilityIndicator availability={scope} />
          </div>
          <p>{scope.note}</p>
        </li>
      ))}
    </ul>
  );
}

function CardContent({ card }: { card: StateBoardCard }) {
  switch (card.kind) {
    case "resident_workers":
      return (
        <>
          <HardCapMeter capacity={card.capacity} />
          {card.workers.length > 0 ? (
            <ul className="state-board-worker-list" aria-label="Resident worker conditions">
              {card.workers.map((worker) => <WorkerRow worker={worker} key={worker.id} />)}
            </ul>
          ) : <p className="state-board-empty">No resident workers are recorded.</p>}
        </>
      );
    case "engine_health":
      return (
        <>
          <EngineHealth health={card.health} />
          {card.capabilities && <CapabilityList flags={card.capabilities} />}
        </>
      );
    case "qualification":
      return (
        <ol className="state-board-qualification-list" aria-label="Qualification checklist">
          {card.steps.map((step) => <QualificationStepRow step={step} key={step.id} />)}
        </ol>
      );
    case "correction_scopes":
      return <CorrectionScopeList scopes={card.scopes} />;
    case "capabilities":
      return <CapabilityList flags={card.flags} />;
    default: {
      const exhaustive: never = card;
      return exhaustive;
    }
  }
}

function StateBoardCardView({ card }: { card: StateBoardCard }) {
  const title = card.title ?? readableKind(card.kind);

  return (
    <article className={`state-board-card is-${card.kind.replaceAll("_", "-")}`} data-state-board-card={card.id} aria-label={title}>
      <header className="state-board-card-header">
        <span>{readableKind(card.kind)}</span>
        <h3>{title}</h3>
      </header>
      {card.description && <p className="state-board-card-description">{card.description}</p>}
      <CardContent card={card} />
    </article>
  );
}

export function StateBoard({ cards, title = "State board", className }: StateBoardProps) {
  return (
    <section className={["state-board", className].filter(Boolean).join(" ")} aria-label={title}>
      <header className="state-board-header">
        <span>Recorded state</span>
        <h2>{title}</h2>
      </header>
      {cards.length > 0 ? (
        <div className="state-board-cards">
          {cards.map((card) => <StateBoardCardView card={card} key={card.id} />)}
        </div>
      ) : <p className="state-board-empty">No state cards were supplied.</p>}
    </section>
  );
}
