/* clozn studio -- live-cast assembly for the Observatory hero (THE CASTING). Split out of
   observatory.mjs (build 4) once the real-cast pipeline grew a second live-computed layer (the
   per-position J-lens argument below) -- observatory.mjs owns the shell/panels/interactivity;
   this module owns "given a run, honestly assemble one castable.mjs data-contract object, or say
   why not." Exports assembleRealCast(run, state) -- same contract as before the split: returns
   {cast, notes} or null (never raises; observatory.mjs falls back to a labeled demo cast on null).

   ============================================================================ what's live here
   sky/tokens/provenance: this run's own influence-map + trace (topk_entropy, alternatives) --
     unchanged from the original build; see the notes array assembled below for the exact wording
     shown in the hero note.
   cloud shape (entropyByDepth): POST /jlens per sampled layer, read at this reply's OWN final
     token position -- unchanged.
   candidate cells / lead-change arcs / commit stratum (cells/leadFlips/meta.commitLayer): NEW.
     /jlens already returns per-layer readouts for EVERY position in the text, not just the last
     one (confirmed live against run_0019f994b3c94_5c4642 on the :8131 gateway: a 35-token reply,
     4 fitted J-lens layers [2,14,21,25] -- the SAME per-layer fetch buildJlensTrajectory() already
     makes for the cloud shape). So a single decision position's per-layer top-1 history was never
     an extra round trip away -- just an unread field. buildArgument() below reads it: picks the
     K<=8 highest-(recorded-)entropy answer positions (trace.topk_entropy, restricted to the prefix
     where the J-lens's OWN tokenization is VERIFIED to line up piece-for-piece with the run's
     recorded trace tokens -- alignedPrefixLen() below; an unverified run honestly omits this layer
     entirely rather than guessing position correspondence), reads each position's per-layer top-1
     RAW LENS LOGIT ARGMAX (already-fetched rows, zero extra network calls), and merges the K
     positions' own argmax trajectories into ONE cloud's worth of cells (distinct candidate pieces,
     capped to the MAX_ARGUMENT_CELLS most frequent, share = fraction of the K featured positions
     whose deepest-layer top-1 pick was that piece) and leadFlips (per-position top-1 changes
     between consecutive sampled layers, deduped, capped to MAX_ARGUMENT_FLIPS, dropped whenever
     either endpoint didn't make the cell cap). commitLayer is the first sampled-layer index at or
     after which NONE of the featured positions still flip (0 when none of them ever flip).
     HONESTY: a linear J-lens always emits something, even from noise -- shallow layers here often
     surface unrelated tokens (confirmed live: layer 2's top-1 was frequently punctuation/garbage
     unrelated to the actual reply). Every cell/arc is a real, capped, computed measurement from
     this run's own recorded text -- never a verified argument, never invented. The hero note states
     the exact K, the positions (by token text + index), the layers, and the two cap counts (cells
     dropped, flips dropped) -- no silent caps, per the codebase's honesty rules.
   ghosts (almosts): each recorded alternative is a valid ForceToken candidate. The canonical
     Time Travel path records GeneratedObservation evidence first; any child Run is created only
     by explicit materialization.

   Caching: buildJlensTrajectory's fetches (the expensive part -- one POST /jlens per sampled layer)
   are cached per run id at module scope (JLENS_CACHE) so re-picking the same run, or forking then
   returning to a parent already visited this page-load, never refetches. buildArgument() itself is
   cheap/pure and always recomputed from whatever's cached -- nothing about it needs its own cache.
*/

async function getJSON(url, opts) {
  try {
    const r = await fetch(url, opts);
    let body = null;
    try { body = await r.json(); } catch { /* empty/non-JSON body */ }
    return { ok: r.ok, status: r.status, body };
  } catch (e) {
    return { ok: false, status: 0, body: null, networkError: String(e) };
  }
}
const postJSON = (url, payload) => getJSON(url, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || {}),
});

const JLENS_TOPK = 5;
const JLENS_TEXT_CAP = 600;        // chars -- bounds one cast assembly's /jlens calls on a long reply
const MAX_JLENS_LAYERS = 6;        // cap on how many per-layer /jlens calls one cast assembly makes (<=12 per spec)
const MAX_ARGUMENT_POSITIONS = 8;  // K: highest-recorded-entropy answer positions fed into the argument
const MAX_ARGUMENT_CELLS = 8;      // cap on distinct candidate cells kept in the merged cloud
const MAX_ARGUMENT_FLIPS = 24;     // cap on distinct lead-change arcs kept

/* ======================================================================== influence map + sky */

export async function getOrComputeInfluenceMap(runId) {
  let res = await getJSON(`/runs/${encodeURIComponent(runId)}/influence-map`);
  if (res.ok && res.body && res.body.available === true) return res.body;
  res = await getJSON(`/runs/${encodeURIComponent(runId)}/influence-map`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
  });
  return (res.ok && res.body && res.body.available === true) ? res.body : null;
}

function sourceIdOf(spanId) { return String(spanId).split(".").slice(0, 2).join("."); }

function displayableSpans(map) {
  const refined = new Set(
    (map.selection && map.selection.refinement && map.selection.refinement.refined_context_span_ids) || []
  );
  return (map.prompt_spans || []).filter(s => s && typeof s.start === "number"
    && (s.level === "fine" || (s.level === "coarse" && !refined.has(s.id))));
}

function spanStrengthMap(map) {
  const m = new Map();
  for (const link of map.links || []) {
    const cur = m.get(link.context_span_id) || 0;
    if (link.abs_delta_nats > cur) m.set(link.context_span_id, link.abs_delta_nats);
  }
  return m;
}

function spanExcerpt(src, span, maxChars) {
  if (!src || typeof src.text !== "string") return "…";
  const raw = src.text.slice(span.start, span.end).replace(/\s+/g, " ").trim();
  if (!raw) return "…";
  return raw.length > maxChars ? raw.slice(0, maxChars - 1) + "…" : raw;
}

/* Maps each trace TOKEN INDEX to the sky indices/weights it was measured to draw on, by char-offset
   overlap between that token's span in the response text and the influence map's answer_spans -- the
   same "is this answer span clearly sourced" signal lens.mjs's buildAnswerMarks reads, just regrouped
   per emitted token instead of per contiguous marked run. */
function mapAnswerTokensToSky(influence, traceTokens, skyIndexById, strengths, maxStrength) {
  const out = new Map();
  const answerSpans = influence.answer_spans || [];
  if (!answerSpans.length) return out;
  const linkByAnswerSpan = new Map(
    ((influence.summary && influence.summary.answer_to_context) || []).map(a => [a.answer_span_id, a])
  );
  let offset = 0;
  const ranges = traceTokens.map(piece => {
    const start = offset, end = offset + piece.length;
    offset = end;
    return [start, end];
  });
  for (const span of answerSpans) {
    const info = linkByAnswerSpan.get(span.id);
    if (!info || !info.clear_source || !(info.top_context_span_ids || []).length) continue;
    const srcs = info.top_context_span_ids.map(cid => {
      const idx = skyIndexById.get(cid);
      if (idx == null) return null;
      const w = maxStrength > 0 ? (strengths.get(cid) || 0) / maxStrength : 0;
      return [idx, Math.max(0.1, w)];
    }).filter(Boolean);
    if (!srcs.length) continue;
    ranges.forEach(([ts, te], i) => {
      if (ts < span.end && te > span.start && !out.has(i)) out.set(i, srcs);
    });
  }
  return out;
}

/* Every recorded alternative is a valid ForceToken candidate: it carries non-empty text (the
   .filter below) and its tokenIndex is, by construction, a valid position in this reply's own
   recorded trace. Runtime readiness is handled by the canonical Time Travel capability path. */
function buildAlmosts(alts) {
  if (!Array.isArray(alts) || !alts.length) return [];
  return alts.slice(0, 3).map(a => {
    const text = String((a && (a.piece != null ? a.piece : a.text)) || "").trim();
    let pull = null;
    if (a && Number.isFinite(a.prob)) pull = a.prob;
    else if (a && Number.isFinite(a.logprob)) pull = Math.exp(a.logprob);
    return { text, pull: Number.isFinite(pull) ? Math.max(0, Math.min(1, pull)) : 0, forkable: true };
  }).filter(a => a.text);
}

/* Cosmetic amplitude only (casting.mjs never displays "turbulence" as a number to the user) -- derived
   from this reply's own recorded per-token entropy spread, not a separately measured signal. */
function turbulenceFromEntropy(ent) {
  const vals = ent.filter(Number.isFinite);
  if (vals.length < 2) return 0.4;
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const variance = vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / vals.length;
  return Math.max(0.15, Math.min(2.2, Math.sqrt(variance)));
}

function topKEntropyBits(row) {
  if (!Array.isArray(row) || !row.length) return null;
  const scores = row.map(r => Number(r && r.score)).filter(Number.isFinite);
  if (!scores.length) return null;
  const m = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) return null;
  const probs = exps.map(e => e / sum);
  const bits = -probs.reduce((acc, p) => (p > 0 ? acc + p * Math.log2(p) : acc), 0);
  return bits;
}
function lastPositionTopKEntropy(readouts) {
  const bits = topKEntropyBits(readouts[readouts.length - 1]);
  return Number.isFinite(bits) ? Math.round(bits * 100) / 100 : 0;
}

function pickLayers(all, max) {
  if (all.length <= max) return all.slice();
  const out = [];
  for (let i = 0; i < max; i++) out.push(all[Math.round(i * (all.length - 1) / (max - 1))]);
  return Array.from(new Set(out));
}

async function buildJlensTrajectoryUncached(text, state) {
  const probe = await postJSON("/jlens", { text, topk: JLENS_TOPK });
  state.jlensProbe = (probe.ok && probe.body) ? probe.body : null;
  if (!probe.ok || !probe.body || !probe.body.available) {
    return { available: false, reason: (probe.body && probe.body.reason) || "J-lens unavailable" };
  }
  const allLayers = Array.isArray(probe.body.available_layers)
    ? probe.body.available_layers.slice().sort((a, b) => a - b) : [];
  if (!allLayers.length) return { available: false, reason: "no J-lens layers reported" };
  const layers = pickLayers(allLayers, MAX_JLENS_LAYERS);
  const perLayer = [];
  for (const L of layers) {
    const res = await postJSON("/jlens", { text, layer: L, topk: JLENS_TOPK });
    if (res.ok && res.body && res.body.available && Array.isArray(res.body.readouts) && res.body.readouts.length) {
      perLayer.push({ layer: L, readouts: res.body.readouts, tokens: res.body.tokens || [] });
    }
  }
  if (!perLayer.length) return { available: false, reason: "layer readouts unavailable" };
  const entropyByDepth = perLayer.map(pl => lastPositionTopKEntropy(pl.readouts));
  return { available: true, layers: perLayer.map(pl => pl.layer), entropyByDepth, perLayer };
}

/* Per-run cache keyed by run id: a run's own recorded response text never changes, so the (several)
   POST /jlens calls buildJlensTrajectory makes need happen only once per run per page-load -- a
   revisit (re-picking the same run, or forking then coming back to a parent already viewed this
   session) reuses it instead of refetching. buildArgument() is cheap/pure and always recomputed. */
const JLENS_CACHE = new Map();   // runId -> Promise<jl>

function buildJlensTrajectoryCached(runId, text, state) {
  if (JLENS_CACHE.has(runId)) {
    const cached = JLENS_CACHE.get(runId);
    cached.then(jl => { state.jlensProbe = jl && jl._probeBody; });
    return cached;
  }
  const p = buildJlensTrajectoryUncached(text, state).then(jl => {
    jl._probeBody = state.jlensProbe;
    return jl;
  });
  JLENS_CACHE.set(runId, p);
  return p;
}

/* ======================================================================== the argument: cells / leadFlips / commitLayer */

/* How many LEADING positions of `traceTokens` the J-lens's own tokenization (`jlensTokens`, from
   whichever fetched layer -- tokenization doesn't depend on layer) reconstructs piece-for-piece. Not
   necessarily the whole reply: JLENS_TEXT_CAP truncates by character count, not token boundary, so
   the tail can legitimately diverge right at the cut -- this stops at the first mismatch rather than
   assuming the truncation landed cleanly. 0 means "don't trust ANY position correspondence here." */
function alignedPrefixLen(traceTokens, jlensTokens) {
  const n = Math.min(traceTokens.length, jlensTokens.length);
  let i = 0;
  while (i < n && traceTokens[i] === jlensTokens[i]) i++;
  return i;
}

/* The K highest-recorded-entropy answer positions within the verified-aligned prefix, in chronological
   (not entropy) order -- so the hero note and any future per-token UI reads left-to-right. */
function pickArgumentPositions(traceEnt, alignedLen, max) {
  const idxs = [];
  for (let i = 0; i < alignedLen; i++) if (Number.isFinite(traceEnt[i])) idxs.push(i);
  idxs.sort((a, b) => traceEnt[b] - traceEnt[a]);
  return idxs.slice(0, max).sort((a, b) => a - b);
}

/* Merge K featured positions' own per-layer top-1 (raw lens logit argmax) trajectories into ONE
   cloud's worth of {cells, leadFlips, commitLayer} -- the data contract has room for exactly one such
   set per cast, so several genuinely-contested moments in the reply are folded into a single storm's
   argument rather than picking just one. Returns {available:false, reason} when there's nothing safe
   to build this from (never guesses a position correspondence it hasn't verified). */
function buildArgument(jl, traceTokens, traceEnt) {
  if (!jl.available || !Array.isArray(jl.perLayer) || !jl.perLayer.length) {
    return { available: false, reason: "no per-layer J-lens readouts to build an argument from" };
  }
  const ref = jl.perLayer.find(pl => Array.isArray(pl.tokens) && pl.tokens.length);
  if (!ref) return { available: false, reason: "the J-lens response carried no token list to align against" };
  const alignedLen = alignedPrefixLen(traceTokens, ref.tokens);
  if (!alignedLen) {
    return { available: false, reason: "the J-lens's own tokenization didn't line up with this run's "
      + "recorded trace tokens -- position correspondence can't be verified" };
  }
  const positions = pickArgumentPositions(traceEnt, alignedLen, MAX_ARGUMENT_POSITIONS);
  if (!positions.length) {
    return { available: false, reason: "no recorded per-token entropy (trace.topk_entropy) to pick "
      + "contested positions from" };
  }

  const nL = jl.perLayer.length;
  const perPositionTop1 = positions.map(pos => jl.perLayer.map(pl => {
    const row = pl.readouts[pos];
    return (Array.isArray(row) && row.length && row[0] && row[0].piece != null) ? String(row[0].piece) : null;
  }));

  const cellCount = new Map();        // piece -> how many (position, layer) draws it ever led
  const cellFinalCount = new Map();   // piece -> how many positions' DEEPEST sampled layer picked it
  perPositionTop1.forEach(seq => {
    seq.forEach(piece => { if (piece != null) cellCount.set(piece, (cellCount.get(piece) || 0) + 1); });
    const final = seq[seq.length - 1];
    if (final != null) cellFinalCount.set(final, (cellFinalCount.get(final) || 0) + 1);
  });
  const allPieces = Array.from(cellCount.keys());
  const keptPieces = allPieces
    .sort((a, b) => (cellFinalCount.get(b) || 0) - (cellFinalCount.get(a) || 0) || cellCount.get(b) - cellCount.get(a))
    .slice(0, MAX_ARGUMENT_CELLS);
  const cellIndex = new Map(keptPieces.map((p, i) => [p, i]));
  const K = positions.length;
  const cells = keptPieces.map((piece, i) => {
    const angle = (i / Math.max(1, keptPieces.length)) * Math.PI * 2;
    const share = (cellFinalCount.get(piece) || 0) / K;
    const trimmed = piece.trim() || piece;
    return {
      x: Math.cos(angle) * 0.5, z: Math.sin(angle) * 0.32, share,
      label: share > 0 ? `${trimmed} \xb7 ${Math.round(share * 100)}%` : undefined,
    };
  });

  const flipKeys = new Set();
  const leadFlips = [];
  let lastFlipLayerIdx = -1;
  let droppedFlips = 0;
  outer:
  for (const seq of perPositionTop1) {
    for (let li = 1; li < seq.length; li++) {
      const from = seq[li - 1], to = seq[li];
      if (from == null || to == null || from === to) continue;
      if (!cellIndex.has(from) || !cellIndex.has(to)) { droppedFlips++; continue; }  // capped-out cell -- omit, don't guess
      const key = `${li}:${cellIndex.get(from)}:${cellIndex.get(to)}`;
      if (flipKeys.has(key)) continue;
      flipKeys.add(key);
      leadFlips.push([li, cellIndex.get(from), cellIndex.get(to)]);
      if (li > lastFlipLayerIdx) lastFlipLayerIdx = li;
      if (leadFlips.length >= MAX_ARGUMENT_FLIPS) break outer;
    }
  }
  const commitLayerIdx = Math.max(0, Math.min(nL - 1, lastFlipLayerIdx + 1));

  return {
    available: true, positions, cells, leadFlips, commitLayer: commitLayerIdx,
    droppedCells: Math.max(0, allPieces.length - keptPieces.length), droppedFlips,
  };
}

/* ======================================================================== real-cast assembly */

/* Never raises: any unexpected shape from a route (malformed influence map, a trace field that isn't
   the array it should be, etc.) falls back to null -> the caller shows a visibly-labeled demo cast
   instead of leaving the page's render chain stuck on an uncaught rejection ("never raise" mirrors the
   discipline clozn/receipts/*.py already hold themselves to). */
export async function assembleRealCast(run, state) {
  try {
    return await assembleRealCastUnsafe(run, state);
  } catch {
    return null;
  }
}

/* assembleRealCastUnsafe sets state.castUnavailableReason right before every early `return null` --
   a side channel (mirroring state.jlensTrajectory/jlensArgument) so the host can show a SPECIFIC
   reason instead of one generic catch-all. Reconstructed Branch Fan children only land here when
   their recorded trace is unavailable; the Branch Fan provenance records that limitation. */
async function assembleRealCastUnsafe(run, state) {
  state.castUnavailableReason = null;
  const text = String(run.response || "").trim();
  if (!text) { state.castUnavailableReason = "this run has no recorded response text"; return null; }

  const influence = await getOrComputeInfluenceMap(state.runId);
  if (!influence) {
    state.castUnavailableReason = "the context-answer influence map is unavailable for this run";
    return null;
  }
  state.influence = influence;

  const ans = influence.answer;
  const alignmentOk = !!(ans && ans.scored_text_matches_recorded && ans.recorded_text === text);
  if (!alignmentOk) {   // can't honestly attribute answer-token provenance -- don't guess
    state.castUnavailableReason = "the influence map's scored text doesn't match this run's recorded reply";
    return null;
  }

  const skySpans = displayableSpans(influence);
  if (!skySpans.length) {
    state.castUnavailableReason = "the influence map has no displayable context spans";
    return null;
  }
  const strengths = spanStrengthMap(influence);
  const maxStrength = Math.max(0, ...Array.from(strengths.values()));
  const sourceById = new Map((influence.prompt_sources || []).map(s => [s.id, s]));
  const skyIndexById = new Map();
  const sky = skySpans.slice(0, 60).map((s, i) => {
    skyIndexById.set(s.id, i);
    const src = sourceById.get(sourceIdOf(s.id));
    const w = maxStrength > 0 ? (strengths.get(s.id) || 0) / maxStrength : 0;
    return { text: spanExcerpt(src, s, 26), weight: Math.max(0.05, w) };
  });

  const traceTokens = (run.trace && Array.isArray(run.trace.tokens)) ? run.trace.tokens : null;
  if (!traceTokens || !traceTokens.length) {
    state.castUnavailableReason = "this run has no recorded per-token trace"
      + (run.parent_run_id
         ? " (a forked child only gets one when its spliced prefix verified token-exact -- this "
           + "one's Branch Fan record says why: changes_applied.branch_fan.trace_provenance)"
         : "");
    return null;
  }
  if (traceTokens.join("") !== text) {   // pieces don't reconstruct the measured text -- don't guess offsets
    state.castUnavailableReason = "this run's recorded trace tokens don't reconstruct its own response text";
    return null;
  }

  const traceEnt = (run.trace && Array.isArray(run.trace.topk_entropy)) ? run.trace.topk_entropy : [];
  const traceAlts = (run.trace && Array.isArray(run.trace.alternatives)) ? run.trace.alternatives : [];
  const sourcesByTokenIdx = mapAnswerTokensToSky(influence, traceTokens, skyIndexById, strengths, maxStrength);

  const tokens = traceTokens.map((piece, i) => ({
    text: piece,
    entropy: Number.isFinite(traceEnt[i]) ? traceEnt[i] : 0,
    sources: sourcesByTokenIdx.get(i) || [],
    almosts: buildAlmosts(traceAlts[i]),
  }));

  const cast = {
    meta: { label: String(run.prompt_summary || state.runId || "this run").slice(0, 40), scripted: false },
    turbulence: turbulenceFromEntropy(traceEnt),
    width: 1,
    crystalline: traceEnt.length > 0 && traceEnt.every(e => !e || e < 0.6),
    cells: [], leadFlips: [],
    sky, tokens,
  };

  const notes = [
    "sky + ground + provenance threads: this run's own context-answer influence map (measured; span "
      + "excerpts stand in for individual words).",
    traceEnt.length
      ? "per-token entropy: this run's own recorded top-k(approx) entropy."
      : "per-token entropy: not recorded on this run -- every planted word reads as calm by default.",
    "ghosts: this run's own recorded alternatives, by probability -- each is a ForceToken candidate. "
      + "Time Travel records generated evidence first; explicit materialization can then create a "
      + "child run.",
  ];

  const jl = await buildJlensTrajectoryCached(state.runId, text.slice(0, JLENS_TEXT_CAP), state);
  state.jlensTrajectory = jl;
  if (jl.available) {
    cast.entropyByDepth = jl.entropyByDepth;
    cast.meta.readoutLayers = jl.layers;
    if (state.health && Number.isFinite(state.health.n_layer)) cast.meta.totalLayers = state.health.n_layer;
    notes.push(`cloud shape: J-lens top-k(k=${JLENS_TOPK}) renormalized entropy at this reply's own `
      + `final token position, at layers [${jl.layers.join(", ")}]`
      + `${text.length > JLENS_TEXT_CAP ? ` (first ${JLENS_TEXT_CAP} chars only)` : ""} -- an `
      + `approximation of full-vocabulary entropy, not the exact quantity.`);

    const arg = buildArgument(jl, traceTokens, traceEnt);
    state.jlensArgument = arg;
    if (arg.available) {
      cast.cells = arg.cells;
      cast.leadFlips = arg.leadFlips;
      cast.meta.commitLayer = arg.commitLayer;
      const posLabel = arg.positions.map(i => `"${traceTokens[i]}"@${i}`).join(", ");
      notes.push(`candidate cells / lead-change arcs / commit stratum: this reply's OWN per-layer `
        + `top-1 (raw lens logit argmax) at its K=${arg.positions.length} highest-recorded-entropy `
        + `answer position(s) (${posLabel}), same ${jl.layers.length} sampled layer(s) as the cloud `
        + `shape above -- merged into ${arg.cells.length} candidate cell(s)`
        + `${arg.droppedCells ? ` (${arg.droppedCells} less-frequent candidate(s) capped out)` : ""} and `
        + `${arg.leadFlips.length} lead-change arc(s)`
        + `${arg.droppedFlips ? ` (${arg.droppedFlips} more omitted -- involved a capped-out cell)` : ""}. `
        + `A linear J-lens always emits SOMETHING even from noise -- shallow layers here often surface `
        + `unrelated tokens (verified live); read every cell/arc as a real, capped measurement, never a `
        + `verified argument.`);
    } else {
      notes.push(`candidate cells / lead-change arcs / commit stratum: NOT computed (${arg.reason}) -- `
        + `the cloud renders as one uncontested blob rather than staging a contest it didn't measure.`);
    }
  } else {
    state.jlensArgument = null;   // no stale value from a previously-assembled run/state reuse
    notes.push(`cloud shape: generic (${jl.reason}) -- no per-layer entropy fed in.`);
    notes.push("candidate cells / lead-change arcs / commit stratum: NOT computed (no J-lens layers to "
      + "read) -- the cloud renders as one uncontested blob rather than staging a contest it didn't measure.");
  }

  return { cast, notes };
}
