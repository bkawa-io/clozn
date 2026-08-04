import { useId } from "react";
import { EvidenceMark } from "./EvidenceMark";
import "./TypedActionOffer.css";

/**
 * C8 -- one explicit next step for evidence that is absent. This accepts a presentation model rather
 * than a route response: adapters decide how transport data becomes an absence, cost, preconditions,
 * and action before this component receives it. The component never fetches, polls, mutates, or
 * infers whether a precondition has been met.
 */

/**
 * An offer can only describe an actual absence. Both members require `reason`, matching EvidenceMark's
 * invariant that missing evidence is explained rather than rendered as a zero or a blank treatment.
 */
export type TypedActionOfferAbsence =
  | { state: "not_measured"; reason: string; label?: string }
  | { state: "unavailable"; reason: string; label?: string };

/**
 * There is exactly one action descriptor per offer. Only an available action can have a callback;
 * a blocked one must instead say which blocker prevents it. This keeps a disabled button from becoming
 * a silent grey control, even for JavaScript callers that do not receive TypeScript's checks.
 */
export type TypedActionOfferAction =
  | {
      availability: "available";
      label: string;
      onAction: () => void;
      blockerReason?: never;
    }
  | {
      availability: "blocked";
      label: string;
      blockerReason: string;
      onAction?: never;
    };

export interface TypedActionOfferProps {
  /** Names the one evidence-gathering offer without coupling it to a route or capability identifier. */
  title: string;
  absence: TypedActionOfferAbsence;
  /** What running the action consumes or changes, stated before the user decides to run it. */
  cost: string;
  /** Conditions the caller knows must hold. An empty list is permitted only when there are none to state. */
  preconditions: readonly string[];
  action: TypedActionOfferAction;
  className?: string;
}

function AbsenceMark({ absence }: { absence: TypedActionOfferAbsence }) {
  // The switch keeps the state/reason pairing intact at the EvidenceMark boundary; no generic absence
  // badge is introduced here, so all absence treatments stay in the shared primitive.
  switch (absence.state) {
    case "not_measured":
      return <EvidenceMark variant="chip" state="not_measured" reason={absence.reason} label={absence.label} />;
    case "unavailable":
      return <EvidenceMark variant="chip" state="unavailable" reason={absence.reason} label={absence.label} />;
    default: {
      const exhaustive: never = absence;
      return exhaustive;
    }
  }
}

export function TypedActionOffer({
  title,
  absence,
  cost,
  preconditions,
  action,
  className,
}: TypedActionOfferProps) {
  const titleId = useId();
  const blockerId = useId();
  const isBlocked = action.availability === "blocked";

  return (
    <section
      className={["typed-action-offer", className].filter(Boolean).join(" ")}
      aria-labelledby={titleId}
    >
      <header className="typed-action-offer-header">
        <span className="typed-action-offer-eyebrow">Evidence action</span>
        <h3 id={titleId}>{title}</h3>
      </header>

      <div className="typed-action-offer-reason">
        <span className="typed-action-offer-label">Why evidence is absent</span>
        <AbsenceMark absence={absence} />
      </div>

      <dl className="typed-action-offer-details">
        <div>
          <dt>Cost</dt>
          <dd>{cost}</dd>
        </div>
        <div>
          <dt>Preconditions</dt>
          <dd>
            <ul aria-label="Preconditions">
              {preconditions.map((precondition, index) => (
                <li key={`${precondition}-${index}`}>{precondition}</li>
              ))}
            </ul>
          </dd>
        </div>
      </dl>

      <div className="typed-action-offer-action">
        <button
          type="button"
          disabled={isBlocked}
          aria-describedby={isBlocked ? blockerId : undefined}
          // The callback is selected only from the available union member. Rendering merely creates
          // the control; invoking the action remains an intentional button-click event.
          onClick={action.availability === "available" ? action.onAction : undefined}
        >
          {action.label}
        </button>
        {action.availability === "blocked" && (
          <p className="typed-action-offer-blocker" id={blockerId} role="note">
            <strong>Blocked by:</strong> {action.blockerReason}
          </p>
        )}
      </div>
    </section>
  );
}
