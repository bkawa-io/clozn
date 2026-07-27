import { useEffect, useMemo, useState, type CSSProperties } from "react";
import type { RuntimeState } from "../../data/types";
import {
  applyAxis,
  applyConcept,
  fitAnchoredBag,
  loadBehaviorWorkspace,
  previewAxis,
  previewConcept,
  saveGuard,
  saveProfile,
  saveSampling,
  switchProfile,
  toggleAnchoredBag,
  type AnchoredBag,
  type AxisPreview,
  type BehaviorAxis,
  type BehaviorProfile,
  type ConceptPreview,
  type GuardSettings,
  type MemoryCard,
  type SamplingSettings,
} from "./api";

interface BehaviorProps {
  runtime: RuntimeState;
  inspectorOpen: boolean;
}

type BehaviorView = "dials" | "concepts" | "memory" | "runtime" | "profiles";
type LoadStatus = "loading" | "ready" | "error";
type OperationStatus = "idle" | "draft" | "pending" | "applied" | "failed" | "reverted";

interface OperationState {
  status: OperationStatus;
  action: string;
  detail?: string;
}

const modules: Array<{ id: BehaviorView; label: string }> = [
  { id: "dials", label: "TONE DIALS" },
  { id: "concepts", label: "CONCEPT STEERING" },
  { id: "memory", label: "ANCHORED MEMORY" },
  { id: "runtime", label: "RUNTIME DEFAULTS" },
  { id: "profiles", label: "PROFILES" },
];

const PREVIEW_PROMPT = "Tell me about your day.";

function changed(current: number, draft: number | undefined) {
  return draft != null && Math.abs(current - draft) > 0.0001;
}

function formatValue(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function basename(value?: string) {
  return value?.split(/[\\/]/).pop() || "—";
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error || "Operation failed");
}

function profileCounts(profile: BehaviorProfile) {
  return {
    dials: Object.keys(profile.dials).length,
    cards: profile.cards.length,
  };
}

export function Behavior({ runtime, inspectorOpen }: BehaviorProps) {
  const [view, setView] = useState<BehaviorView>("dials");
  const [loadStatus, setLoadStatus] = useState<LoadStatus>("loading");
  const [axes, setAxes] = useState<BehaviorAxis[]>([]);
  const [drafts, setDrafts] = useState<Record<string, number>>({});
  const [selectedAxisName, setSelectedAxisName] = useState("");
  const [sampling, setSampling] = useState<SamplingSettings>();
  const [samplingDraft, setSamplingDraft] = useState<SamplingSettings>();
  const [guard, setGuard] = useState<GuardSettings>();
  const [guardDraft, setGuardDraft] = useState<GuardSettings>();
  const [cards, setCards] = useState<MemoryCard[]>([]);
  const [bags, setBags] = useState<AnchoredBag[]>([]);
  const [profiles, setProfiles] = useState<BehaviorProfile[]>([]);
  const [activeProfile, setActiveProfile] = useState<string>();
  const [errors, setErrors] = useState<Awaited<ReturnType<typeof loadBehaviorWorkspace>>["errors"]>({});
  const [operation, setOperation] = useState<OperationState>({
    status: "idle",
    action: "NO PENDING CHANGE",
  });
  const [previewPrompt, setPreviewPrompt] = useState(PREVIEW_PROMPT);
  const [axisPreview, setAxisPreview] = useState<AxisPreview>();
  const [conceptPreview, setConceptPreview] = useState<ConceptPreview>();
  const [concept, setConcept] = useState("");
  const [conceptStrength, setConceptStrength] = useState(1);
  const [activeConcepts, setActiveConcepts] = useState<Record<string, number>>({});
  const [profileName, setProfileName] = useState("");
  const [profileDescription, setProfileDescription] = useState("");

  function installWorkspace(next: Awaited<ReturnType<typeof loadBehaviorWorkspace>>) {
    setAxes(next.axes);
    setDrafts(Object.fromEntries(next.axes.map((axis) => [axis.name, axis.value])));
    setSelectedAxisName((current) =>
      next.axes.some((axis) => axis.name === current) ? current : next.axes[0]?.name ?? "");
    setSampling(next.sampling);
    setSamplingDraft(next.sampling);
    setGuard(next.guard);
    setGuardDraft(next.guard);
    setCards(next.cards);
    setBags(next.bags);
    setProfiles(next.profiles);
    setActiveProfile(next.activeProfile);
    setErrors(next.errors);
  }

  useEffect(() => {
    const controller = new AbortController();
    setLoadStatus("loading");
    void loadBehaviorWorkspace(controller.signal).then((next) => {
      if (controller.signal.aborted) return;
      installWorkspace(next);
      setLoadStatus(next.axes.length || next.sampling ? "ready" : "error");
    }).catch(() => {
      if (!controller.signal.aborted) setLoadStatus("error");
    });
    return () => controller.abort();
  }, []);

  const selectedAxis = axes.find((axis) => axis.name === selectedAxisName);
  const dirtyAxes = axes.filter((axis) => changed(axis.value, drafts[axis.name]));
  const activeAxes = axes.filter((axis) => Math.abs(axis.value) > 0.0001);
  const activeCards = cards.filter((card) => card.status === "active");
  const activeBags = bags.filter((bag) => bag.on);
  const samplingDirty = Boolean(
    sampling
    && samplingDraft
    && (
      sampling.sampling !== samplingDraft.sampling
      || sampling.sample_temperature !== samplingDraft.sample_temperature
      || sampling.sample_top_p !== samplingDraft.sample_top_p
      || sampling.sample_top_k !== samplingDraft.sample_top_k
      || sampling.sample_repeat_penalty !== samplingDraft.sample_repeat_penalty
    ),
  );
  const guardDirty = JSON.stringify(guard) !== JSON.stringify(guardDraft);
  const pendingCount = dirtyAxes.length + Number(samplingDirty) + Number(guardDirty);
  const selectedDraft = selectedAxis ? drafts[selectedAxis.name] ?? selectedAxis.value : 0;
  const sortedActiveAxes = useMemo(
    () => [...activeAxes].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    [activeAxes],
  );

  function updateDraft(name: string, value: number) {
    setSelectedAxisName(name);
    setDrafts((current) => ({ ...current, [name]: value }));
    const axis = axes.find((item) => item.name === name);
    setOperation({
      status: axis && changed(axis.value, value) ? "draft" : "idle",
      action: axis && changed(axis.value, value) ? `DRAFT · ${name.toUpperCase()}` : "NO PENDING CHANGE",
      detail: axis ? `${formatValue(axis.value)} → ${formatValue(value)}` : undefined,
    });
  }

  function revertAxisDrafts() {
    setDrafts(Object.fromEntries(axes.map((axis) => [axis.name, axis.value])));
    setOperation({
      status: "reverted",
      action: "DRAFTS REVERTED",
      detail: `${dirtyAxes.length} ${dirtyAxes.length === 1 ? "AXIS" : "AXES"}`,
    });
  }

  function updateSamplingDraft(
    next: SamplingSettings,
    label: string,
    value: string,
  ) {
    setSamplingDraft(next);
    setOperation({
      status: "draft",
      action: `DRAFT · ${label}`,
      detail: value,
    });
  }

  function revertSamplingDraft() {
    setSamplingDraft(sampling);
    setOperation({
      status: "reverted",
      action: "DECODING DRAFT REVERTED",
    });
  }

  function updateGuardDraft(
    next: GuardSettings,
    label: string,
    value: string,
  ) {
    setGuardDraft(next);
    setOperation({
      status: "draft",
      action: `DRAFT · ${label}`,
      detail: value,
    });
  }

  function revertGuardDraft() {
    setGuardDraft(guard);
    setOperation({
      status: "reverted",
      action: "GUARD DRAFT REVERTED",
    });
  }

  async function commitAxis(axis: BehaviorAxis) {
    const value = drafts[axis.name] ?? axis.value;
    setOperation({
      status: "pending",
      action: `APPLYING · ${axis.name.toUpperCase()}`,
      detail: formatValue(value),
    });
    try {
      const result = await applyAxis(axis.name, value);
      setAxes((current) => current.map((item) =>
        item.name === axis.name ? { ...item, value } : item));
      setDrafts((current) => ({ ...current, [axis.name]: value }));
      setOperation({
        status: "applied",
        action: `APPLIED · ${axis.name.toUpperCase()}`,
        detail: result.warning || formatValue(value),
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: `FAILED · ${axis.name.toUpperCase()}`,
        detail: errorMessage(error),
      });
    }
  }

  async function commitAllAxes() {
    if (!dirtyAxes.length) return;
    setOperation({
      status: "pending",
      action: "APPLYING DIAL STACK",
      detail: `${dirtyAxes.length} ${dirtyAxes.length === 1 ? "AXIS" : "AXES"}`,
    });
    const applied: Record<string, number> = {};
    try {
      for (const axis of dirtyAxes) {
        const value = drafts[axis.name] ?? axis.value;
        await applyAxis(axis.name, value);
        applied[axis.name] = value;
      }
      setAxes((current) => current.map((axis) =>
        applied[axis.name] == null ? axis : { ...axis, value: applied[axis.name] }));
      setOperation({
        status: "applied",
        action: "DIAL STACK APPLIED",
        detail: `${dirtyAxes.length} ${dirtyAxes.length === 1 ? "AXIS" : "AXES"}`,
      });
    } catch (error) {
      setAxes((current) => current.map((axis) =>
        applied[axis.name] == null ? axis : { ...axis, value: applied[axis.name] }));
      setOperation({
        status: "failed",
        action: "PARTIAL APPLY",
        detail: errorMessage(error),
      });
    }
  }

  async function runAxisPreview() {
    if (!selectedAxis || !previewPrompt.trim()) return;
    setOperation({
      status: "pending",
      action: `PREVIEWING · ${selectedAxis.name.toUpperCase()}`,
      detail: formatValue(selectedDraft),
    });
    try {
      const result = await previewAxis(selectedAxis.name, selectedDraft, previewPrompt.trim());
      setAxisPreview(result);
      setConceptPreview(undefined);
      setOperation({
        status: "applied",
        action: `PREVIEW READY · ${selectedAxis.name.toUpperCase()}`,
        detail: result.warning,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "PREVIEW FAILED",
        detail: errorMessage(error),
      });
    }
  }

  async function commitConcept(strength = conceptStrength, conceptOverride?: string) {
    const word = (conceptOverride ?? concept).trim();
    if (!word) return;
    setOperation({
      status: "pending",
      action: strength === 0 ? `REMOVING · ${word.toUpperCase()}` : `APPLYING · ${word.toUpperCase()}`,
      detail: formatValue(strength),
    });
    try {
      const active = await applyConcept(word, strength);
      setActiveConcepts(active);
      setOperation({
        status: strength === 0 ? "reverted" : "applied",
        action: strength === 0 ? `REMOVED · ${word.toUpperCase()}` : `APPLIED · ${word.toUpperCase()}`,
        detail: strength === 0 ? undefined : formatValue(strength),
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "CONCEPT APPLY FAILED",
        detail: errorMessage(error),
      });
    }
  }

  async function runConceptPreview() {
    const word = concept.trim();
    if (!word || !previewPrompt.trim()) return;
    setOperation({
      status: "pending",
      action: `PREVIEWING · ${word.toUpperCase()}`,
      detail: formatValue(conceptStrength),
    });
    try {
      const result = await previewConcept(word, conceptStrength, previewPrompt.trim());
      setConceptPreview(result);
      setAxisPreview(undefined);
      setOperation({
        status: "applied",
        action: `PREVIEW READY · ${word.toUpperCase()}`,
        detail: result.note,
      });
    } catch (error) {
      setOperation({
        status: "failed",
        action: "CONCEPT PREVIEW FAILED",
        detail: errorMessage(error),
      });
    }
  }

  async function commitSampling() {
    if (!samplingDraft) return;
    setOperation({ status: "pending", action: "APPLYING RUNTIME DEFAULTS" });
    try {
      const next = await saveSampling(samplingDraft);
      setSampling(next);
      setSamplingDraft(next);
      setOperation({ status: "applied", action: "RUNTIME DEFAULTS APPLIED" });
    } catch (error) {
      setOperation({ status: "failed", action: "RUNTIME APPLY FAILED", detail: errorMessage(error) });
    }
  }

  async function commitGuard() {
    if (!guardDraft) return;
    setOperation({ status: "pending", action: "APPLYING GUARD DEFAULT" });
    try {
      const next = await saveGuard(guardDraft);
      setGuard(next);
      setGuardDraft(next);
      setOperation({ status: "applied", action: "GUARD DEFAULT APPLIED" });
    } catch (error) {
      setOperation({ status: "failed", action: "GUARD APPLY FAILED", detail: errorMessage(error) });
    }
  }

  async function toggleBag(cardId: string, on: boolean) {
    setOperation({
      status: "pending",
      action: `${on ? "ENABLING" : "DISABLING"} ANCHORED BAG`,
      detail: cardId,
    });
    try {
      const next = await toggleAnchoredBag(cardId, on);
      setBags((current) => current.map((bag) => bag.card_id === cardId ? next : bag));
      setOperation({
        status: on ? "applied" : "reverted",
        action: `${on ? "ENABLED" : "DISABLED"} ANCHORED BAG`,
        detail: cardId,
      });
    } catch (error) {
      setOperation({ status: "failed", action: "MEMORY APPLY FAILED", detail: errorMessage(error) });
    }
  }

  async function fitCard(cardId: string) {
    setOperation({ status: "pending", action: "FITTING ANCHORED BAG", detail: cardId });
    try {
      const next = await fitAnchoredBag(cardId);
      setBags((current) => [
        ...current.filter((bag) => bag.card_id !== cardId),
        next,
      ]);
      setOperation({
        status: "applied",
        action: "ANCHORED BAG FIT",
        detail: next.reconstruction_cos == null
          ? cardId
          : `COS ${next.reconstruction_cos.toFixed(3)}`,
      });
    } catch (error) {
      setOperation({ status: "failed", action: "MEMORY FIT FAILED", detail: errorMessage(error) });
    }
  }

  async function createProfile() {
    const name = profileName.trim();
    if (!name) return;
    setOperation({ status: "pending", action: "SAVING PROFILE", detail: name });
    try {
      await saveProfile(name, profileDescription.trim(), axes, cards);
      const next = await loadBehaviorWorkspace();
      installWorkspace(next);
      setProfileName("");
      setProfileDescription("");
      setOperation({ status: "applied", action: "PROFILE SAVED", detail: name });
    } catch (error) {
      setOperation({ status: "failed", action: "PROFILE SAVE FAILED", detail: errorMessage(error) });
    }
  }

  async function activateProfile(name: string) {
    setOperation({ status: "pending", action: "SWITCHING PROFILE", detail: name });
    try {
      await switchProfile(name);
      const next = await loadBehaviorWorkspace();
      installWorkspace(next);
      setOperation({ status: "applied", action: "PROFILE ACTIVE", detail: name });
    } catch (error) {
      setOperation({ status: "failed", action: "PROFILE SWITCH FAILED", detail: errorMessage(error) });
    }
  }

  return (
    <>
      <aside className="instrument behavior-stack" aria-labelledby="behavior-stack-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">INTERVENTION SCOPE</span>
            <h2 id="behavior-stack-title">Behavior</h2>
          </div>
          <strong>{pendingCount} DRAFT</strong>
        </header>
        <nav className="behavior-modules" aria-label="Behavior modules">
          {modules.map((module) => (
            <button
              type="button"
              className={view === module.id ? "is-active" : ""}
              aria-pressed={view === module.id}
              onClick={() => setView(module.id)}
              key={module.id}
            >
              <span>{module.label}</span>
              <b>
                {module.id === "dials"
                  ? activeAxes.length
                  : module.id === "concepts"
                    ? Object.keys(activeConcepts).length
                    : module.id === "memory"
                      ? activeCards.length + activeBags.length
                      : module.id === "profiles" ? profiles.length : sampling ? 1 : 0}
              </b>
            </button>
          ))}
        </nav>
        <section className="behavior-stack-state">
          <header><span>ACTIVE STACK</span><b>{activeAxes.length + activeCards.length + activeBags.length}</b></header>
          <dl>
            <div><dt>Model</dt><dd>{basename(runtime.engine?.model)}</dd></div>
            <div><dt>Tone dials</dt><dd>{activeAxes.length}</dd></div>
            <div><dt>Concept dials</dt><dd>{Object.keys(activeConcepts).length}</dd></div>
            <div><dt>Memory cards</dt><dd>{activeCards.length}</dd></div>
            <div><dt>Anchored bags</dt><dd>{activeBags.length}</dd></div>
            <div><dt>Profile</dt><dd>{activeProfile || "—"}</dd></div>
          </dl>
          <div className="behavior-active-dials">
            {sortedActiveAxes.slice(0, 6).map((axis) => (
              <button type="button" onClick={() => {
                setView("dials");
                setSelectedAxisName(axis.name);
              }} key={axis.name}>
                <span>{axis.name}</span>
                <output>{formatValue(axis.value)}</output>
              </button>
            ))}
          </div>
        </section>
      </aside>

      <section className="instrument behavior-console" aria-labelledby="behavior-console-title">
        {loadStatus === "loading" ? (
          <div className="behavior-load-state">READING BEHAVIOR STATE</div>
        ) : view === "dials" ? (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">MODEL INTERVENTION</span>
                <h1 id="behavior-console-title">Tone dials</h1>
              </div>
              <div className="behavior-head-stats">
                <span><b>AVAILABLE</b>{axes.length}</span>
                <span><b>ACTIVE</b>{activeAxes.length}</span>
                <span><b>DRAFT</b>{dirtyAxes.length}</span>
              </div>
              <div className="behavior-head-actions">
                <button type="button" disabled={!dirtyAxes.length} onClick={revertAxisDrafts}>REVERT</button>
                <button type="button" className="is-primary" disabled={!dirtyAxes.length || operation.status === "pending"} onClick={() => void commitAllAxes()}>
                  APPLY {dirtyAxes.length || ""}
                </button>
              </div>
            </header>
            {errors.axes ? (
              <div className="behavior-unavailable">{errors.axes}</div>
            ) : (
              <div className="behavior-dial-list">
                {axes.map((axis) => {
                  const draft = drafts[axis.name] ?? axis.value;
                  const currentPosition = (axis.value + axis.max) / (axis.max * 2) * 100;
                  return (
                    <div
                      className={[
                        "behavior-dial-row",
                        selectedAxisName === axis.name ? "is-selected" : "",
                        changed(axis.value, draft) ? "is-dirty" : "",
                      ].join(" ")}
                      style={{ "--dial-current": `${currentPosition}%` } as CSSProperties}
                      key={axis.name}
                    >
                      <button type="button" className="behavior-dial-label" onClick={() => setSelectedAxisName(axis.name)}>
                        <strong>{axis.name}</strong>
                        <span>{axis.calibrated ? "CALIBRATED" : axis.custom ? "CUSTOM" : axis.library ? "LIBRARY" : "UNCALIBRATED"}</span>
                      </button>
                      <div className="behavior-dial-control">
                        <div className="behavior-poles">
                          <span>{axis.poles[1]}</span>
                          <b>CURRENT {formatValue(axis.value)}</b>
                          <span>{axis.poles[0]}</span>
                        </div>
                        <input
                          type="range"
                          aria-label={`${axis.name}, ${axis.poles[1]} to ${axis.poles[0]}`}
                          min={-axis.max}
                          max={axis.max}
                          step=".05"
                          value={draft}
                          onChange={(event) => updateDraft(axis.name, Number(event.target.value))}
                        />
                      </div>
                      <output>{formatValue(draft)}</output>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        ) : view === "concepts" ? (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">DIRECTIONAL INTERVENTION</span>
                <h1 id="behavior-console-title">Concept steering</h1>
              </div>
              <span className={`behavior-capability ${runtime.engine?.jlens ? "is-ready" : ""}`}>
                {runtime.engine?.jlens ? "J-LENS READY" : "J-LENS REQUIRED"}
              </span>
            </header>
            <div className="behavior-concept-stage">
              <section className="behavior-concept-form">
                <label>
                  <span>CONCEPT</span>
                  <input value={concept} onChange={(event) => setConcept(event.target.value)} placeholder="word or token" />
                </label>
                <label>
                  <span>STRENGTH</span>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step=".1"
                    value={conceptStrength}
                    onChange={(event) => setConceptStrength(Number(event.target.value))}
                  />
                  <output>{formatValue(conceptStrength)}</output>
                </label>
                <div className="behavior-concept-actions">
                  <button type="button" disabled={!runtime.engine?.jlens || !concept.trim() || operation.status === "pending"} onClick={() => void runConceptPreview()}>PREVIEW</button>
                  <button type="button" className="is-primary" disabled={!runtime.engine?.jlens || !concept.trim() || operation.status === "pending"} onClick={() => void commitConcept()}>APPLY CONCEPT</button>
                </div>
              </section>
              <section className="behavior-session-concepts">
                <header><span>SESSION-OBSERVED CONCEPTS</span><b>{Object.keys(activeConcepts).length}</b></header>
                {Object.entries(activeConcepts).map(([name, strength]) => (
                  <div key={name}>
                    <strong>{name}</strong>
                    <output>{formatValue(strength)}</output>
                    <button type="button" onClick={() => {
                      setConcept(name);
                      void commitConcept(0, name);
                    }}>REMOVE</button>
                  </div>
                ))}
                {!Object.keys(activeConcepts).length && <div className="behavior-empty-row">NONE OBSERVED</div>}
              </section>
            </div>
          </>
        ) : view === "memory" ? (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">MEMORY INTERVENTION</span>
                <h1 id="behavior-console-title">Anchored memory</h1>
              </div>
              <div className="behavior-head-stats">
                <span><b>CARDS</b>{cards.length}</span>
                <span><b>ANCHORED</b>{bags.length}</span>
                <span><b>ACTIVE</b>{activeBags.length}</span>
              </div>
            </header>
            <div className="behavior-memory-stage">
              {errors.memory && <div className="behavior-unavailable">{errors.memory}</div>}
              <section className="behavior-memory-section">
                <header><span>ANCHORED BAGS</span><b>{bags.length}</b></header>
                {bags.map((bag) => (
                  <article className="behavior-bag" key={bag.card_id}>
                    <div>
                      <strong>{bag.card_text || bag.card_id}</strong>
                      <span>
                        {bag.terms.length} TERMS
                        {bag.layer == null ? "" : ` · L${bag.layer}`}
                        {bag.reconstruction_cos == null ? "" : ` · COS ${bag.reconstruction_cos.toFixed(3)}`}
                      </span>
                    </div>
                    <div className="behavior-bag-terms">
                      {bag.terms.map((term) => <span key={term.token}>{term.token} <b>{term.alpha >= 0 ? "+" : ""}{term.alpha.toFixed(3)}</b></span>)}
                    </div>
                    <label>
                      <input type="checkbox" checked={bag.on} onChange={(event) => void toggleBag(bag.card_id, event.target.checked)} />
                      <span>{bag.on ? "ON" : "OFF"}</span>
                    </label>
                  </article>
                ))}
                {!bags.length && <div className="behavior-empty-row">0 ANCHORED BAGS</div>}
              </section>
              <section className="behavior-memory-section">
                <header><span>MEMORY CARDS</span><b>{cards.length}</b></header>
                {cards.map((card) => (
                  <article className="behavior-card" key={card.id}>
                    <div><strong>{card.text}</strong><span>{card.status.toUpperCase()}</span></div>
                    <button type="button" disabled={!runtime.engine?.jlens || operation.status === "pending"} onClick={() => void fitCard(card.id)}>
                      {runtime.engine?.jlens ? "FIT ANCHOR" : "J-LENS REQUIRED"}
                    </button>
                  </article>
                ))}
                {!cards.length && <div className="behavior-empty-row">0 MEMORY CARDS</div>}
              </section>
            </div>
          </>
        ) : view === "runtime" ? (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">SERVER DEFAULTS</span>
                <h1 id="behavior-console-title">Runtime</h1>
              </div>
            </header>
            <div className="behavior-runtime-stage">
              <section className="behavior-runtime-block">
                <header><span>DECODING</span><b>{samplingDirty ? "DRAFT" : "CURRENT"}</b></header>
                {samplingDraft ? (
                  <div className="behavior-runtime-fields">
                    <label className="behavior-toggle">
                      <input
                        type="checkbox"
                        checked={samplingDraft.sampling}
                        onChange={(event) => updateSamplingDraft(
                          { ...samplingDraft, sampling: event.target.checked },
                          "SAMPLING",
                          event.target.checked ? "ON" : "OFF",
                        )}
                      />
                      <span>SAMPLING</span>
                    </label>
                    {([
                      ["sample_temperature", "TEMPERATURE", 0, 2, .05],
                      ["sample_top_p", "TOP P", 0, 1, .01],
                      ["sample_top_k", "TOP K", 0, 200, 1],
                      ["sample_repeat_penalty", "REPEAT PENALTY", .5, 2, .01],
                    ] as const).map(([key, label, min, max, step]) => (
                      <label key={key}>
                        <span>{label}</span>
                        <input
                          type="number"
                          min={min}
                          max={max}
                          step={step}
                          value={samplingDraft[key]}
                          onChange={(event) => updateSamplingDraft(
                            {
                              ...samplingDraft,
                              [key]: Number(event.target.value),
                            },
                            label,
                            event.target.value,
                          )}
                        />
                      </label>
                    ))}
                    <div className="behavior-runtime-actions">
                      <button type="button" disabled={!samplingDirty || operation.status === "pending"} onClick={revertSamplingDraft}>REVERT</button>
                      <button type="button" className="is-primary" disabled={!samplingDirty || operation.status === "pending"} onClick={() => void commitSampling()}>APPLY DECODING</button>
                    </div>
                  </div>
                ) : <div className="behavior-unavailable">{errors.sampling || "SAMPLING ROUTE UNAVAILABLE"}</div>}
              </section>
              <section className="behavior-runtime-block">
                <header><span>DISPOSITION GUARD DEFAULT</span><b>{guardDraft ? guardDirty ? "DRAFT" : guardDraft.enabled ? "ON" : "OFF" : "UNAVAILABLE"}</b></header>
                {guardDraft ? (
                  <div className="behavior-runtime-fields">
                    <label className="behavior-toggle">
                      <input
                        type="checkbox"
                        checked={guardDraft.enabled}
                        onChange={(event) => updateGuardDraft(
                          { ...guardDraft, enabled: event.target.checked },
                          "GUARD",
                          event.target.checked ? "ON" : "OFF",
                        )}
                      />
                      <span>ENABLED</span>
                    </label>
                    <label className="is-wide">
                      <span>CONCEPTS · COMMA SEPARATED</span>
                      <input
                        value={(guardDraft.guard?.concepts ?? []).join(", ")}
                        onChange={(event) => updateGuardDraft(
                          {
                            ...guardDraft,
                            guard: {
                              ...(guardDraft.guard ?? {}),
                              concepts: event.target.value.split(",").map((item) => item.trim()).filter(Boolean),
                            },
                          },
                          "GUARD CONCEPTS",
                          event.target.value,
                        )}
                      />
                    </label>
                    <div className="behavior-runtime-actions">
                      <button type="button" disabled={!guardDirty || operation.status === "pending"} onClick={revertGuardDraft}>REVERT</button>
                      <button type="button" className="is-primary" disabled={!guardDirty || operation.status === "pending"} onClick={() => void commitGuard()}>APPLY GUARD</button>
                    </div>
                  </div>
                ) : <div className="behavior-unavailable">{errors.guard || "GUARD ROUTE UNAVAILABLE"}</div>}
              </section>
            </div>
          </>
        ) : (
          <>
            <header className="instrument-head behavior-console-head">
              <div>
                <span className="eyebrow">PORTABLE BUNDLE</span>
                <h1 id="behavior-console-title">Profiles</h1>
              </div>
              <div className="behavior-head-stats">
                <span><b>SAVED</b>{profiles.length}</span>
                <span><b>ACTIVE</b>{activeProfile ? 1 : 0}</span>
              </div>
            </header>
            <div className="behavior-profile-stage">
              <section className="behavior-profile-create">
                <label><span>NAME</span><input value={profileName} onChange={(event) => setProfileName(event.target.value)} placeholder="profile-name" /></label>
                <label><span>DESCRIPTION</span><input value={profileDescription} onChange={(event) => setProfileDescription(event.target.value)} /></label>
                <div>
                  <span>{axes.length} DIALS · {activeCards.length} ACTIVE CARDS</span>
                  <button type="button" className="is-primary" disabled={!profileName.trim() || operation.status === "pending"} onClick={() => void createProfile()}>SAVE CURRENT</button>
                </div>
              </section>
              <section className="behavior-profile-list">
                {profiles.map((profile) => {
                  const counts = profileCounts(profile);
                  const active = profile.name === activeProfile;
                  return (
                    <article className={active ? "is-active" : ""} key={profile.name}>
                      <div>
                        <strong>{profile.name}</strong>
                        <span>{profile.description || "NO DESCRIPTION"}</span>
                      </div>
                      <dl>
                        <div><dt>DIALS</dt><dd>{counts.dials}</dd></div>
                        <div><dt>CARDS</dt><dd>{counts.cards}</dd></div>
                      </dl>
                      <button type="button" disabled={active || operation.status === "pending"} onClick={() => void activateProfile(profile.name)}>
                        {active ? "ACTIVE" : "SWITCH"}
                      </button>
                    </article>
                  );
                })}
                {!profiles.length && <div className="behavior-empty-row">0 SAVED PROFILES</div>}
              </section>
            </div>
          </>
        )}
      </section>

      {inspectorOpen && (
        <aside className="instrument behavior-inspector" aria-labelledby="behavior-inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">CHANGE INSPECTOR</span>
              <h2 id="behavior-inspector-title">Consequence</h2>
            </div>
            <span className={`behavior-operation-chip is-${operation.status}`}>{operation.status.toUpperCase()}</span>
          </header>
          <section className="behavior-operation">
            <span>LAST OPERATION</span>
            <strong>{operation.action}</strong>
            {operation.detail && <p>{operation.detail}</p>}
          </section>

          {view === "dials" && selectedAxis ? (
            <>
              <section className="behavior-selected-control">
                <header><span>SELECTED AXIS</span><b>{selectedAxis.calibrated ? "CALIBRATED" : "UNCALIBRATED"}</b></header>
                <strong>{selectedAxis.name}</strong>
                <div>
                  <span>{selectedAxis.poles[1]}</span>
                  <output>{formatValue(selectedDraft)}</output>
                  <span>{selectedAxis.poles[0]}</span>
                </div>
                <dl>
                  <div><dt>CURRENT</dt><dd>{formatValue(selectedAxis.value)}</dd></div>
                  <div><dt>DRAFT</dt><dd>{formatValue(selectedDraft)}</dd></div>
                  <div><dt>BOUND</dt><dd>±{selectedAxis.max.toFixed(2)}</dd></div>
                </dl>
                <button type="button" className="is-primary" disabled={!changed(selectedAxis.value, selectedDraft) || operation.status === "pending"} onClick={() => void commitAxis(selectedAxis)}>APPLY AXIS</button>
              </section>
              <section className="behavior-preview-control">
                <label><span>PREVIEW PROMPT</span><textarea value={previewPrompt} onChange={(event) => setPreviewPrompt(event.target.value)} /></label>
                <button type="button" disabled={!previewPrompt.trim() || operation.status === "pending"} onClick={() => void runAxisPreview()}>RUN A/B PREVIEW</button>
              </section>
              <section className="behavior-preview-output">
                <header><span>A/B OUTPUT</span><b>{axisPreview ? axisPreview.axis.toUpperCase() : "NOT RUN"}</b></header>
                {axisPreview ? (
                  <>
                    <div><span>BASELINE</span><p>{axisPreview.baseline}</p></div>
                    <div className="is-steered"><span>DRAFT VALUE</span><p>{axisPreview.steered}</p></div>
                  </>
                ) : <div className="behavior-empty-row">NO PREVIEW RESULT</div>}
              </section>
            </>
          ) : view === "concepts" ? (
            <>
              <section className="behavior-selected-control">
                <header><span>CONCEPT DIRECTION</span><b>{runtime.engine?.jlens ? "READY" : "UNAVAILABLE"}</b></header>
                <strong>{concept || "—"}</strong>
                <div><span>-2.00</span><output>{formatValue(conceptStrength)}</output><span>+2.00</span></div>
              </section>
              <section className="behavior-preview-control">
                <label><span>PREVIEW PROMPT</span><textarea value={previewPrompt} onChange={(event) => setPreviewPrompt(event.target.value)} /></label>
              </section>
              <section className="behavior-preview-output">
                <header><span>A/B OUTPUT</span><b>{conceptPreview ? conceptPreview.concept.toUpperCase() : "NOT RUN"}</b></header>
                {conceptPreview ? (
                  <>
                    <div><span>BASELINE</span><p>{conceptPreview.baseline}</p></div>
                    <div className="is-steered"><span>CONCEPT STEERED</span><p>{conceptPreview.steered}</p></div>
                  </>
                ) : <div className="behavior-empty-row">NO PREVIEW RESULT</div>}
              </section>
            </>
          ) : (
            <section className="behavior-scope-facts">
              <header><span>CURRENT SCOPE</span><b>{view.toUpperCase()}</b></header>
              <dl>
                <div><dt>MODEL</dt><dd>{basename(runtime.engine?.model)}</dd></div>
                <div><dt>PENDING DRAFTS</dt><dd>{pendingCount}</dd></div>
                <div><dt>ACTIVE DIALS</dt><dd>{activeAxes.length}</dd></div>
                <div><dt>ACTIVE CARDS</dt><dd>{activeCards.length}</dd></div>
                <div><dt>ACTIVE PROFILE</dt><dd>{activeProfile || "—"}</dd></div>
              </dl>
            </section>
          )}
        </aside>
      )}
    </>
  );
}
