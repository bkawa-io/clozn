import { useId, useState } from "react";
import type { InvestigationLocus } from "../../core/investigation";
import "./investigation.css";

export interface TestThisLauncherProps {
  readonly locus: InvestigationLocus;
  readonly onLaunch: (locus: InvestigationLocus) => void | Promise<void>;
  readonly label?: string;
  readonly disabled?: boolean;
  readonly className?: string;
}

/** Deliberately action-shaped: launching may run a model or create a child run. */
export function TestThisLauncher({ locus, onLaunch, label = "Test this", disabled = false, className }: TestThisLauncherProps) {
  const descriptionId = useId();
  const errorId = useId();
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string>();

  const launch = async () => {
    if (launching || disabled) return;
    setLaunching(true);
    setError(undefined);
    try {
      await onLaunch(locus);
    } catch {
      setError("Could not start this test. Nothing has been changed here.");
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className={["investigation-test-this", className].filter(Boolean).join(" ")}>
      <button aria-describedby={`${descriptionId}${error ? ` ${errorId}` : ""}`} className="investigation-test-this__button" disabled={disabled || launching} onClick={launch} type="button">
        {launching ? "Starting test…" : label}
      </button>
      <span className="investigation-test-this__notice" id={descriptionId}>May run model work or create a child run.</span>
      {error ? <span className="investigation-test-this__error" id={errorId} role="alert">{error}</span> : null}
    </div>
  );
}
