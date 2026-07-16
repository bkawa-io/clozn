/* heavnOS · Replay desk — the ζ design, componentized, with the transport verbs LIVE (W1).
   Honesty rules carried over: CAPTURED/DERIVED/SAMPLE tags, provenance verbatim, no mock data
   on live runs, computed-on-demand receipts never implied to pre-exist. */
import { html, useState, useEffect, useRef } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast, normSteps, weightsFor, colsFor, colGeom,
         firstLine, shortTime, REDUCED } from "../state.mjs";
import { api } from "../api.mjs";
import { loadRun } from "../app.mjs";

/* ───────────────────────── module root ───────────────────────── */
export function ReplayModule(){
  const rec = useStore(x => x.rec);
  const live = useStore(x => x.live);
  usePlayEngine();
  if(!rec) return html`<div class="replay-grid"><div class="col">
      <${Monitor} rec=${null}/>
      <div class="mod"><div class="empty">Nothing has spoken yet${live ? " — say something into the monitor above" : ""}.</div></div>
    </div><div class="col"></div></div>`;
  return html`<div class="replay-grid">
    <div class="col">
      <div class="monrow"><${Monitor} rec=${rec}/><${Logs} rec=${rec}/></div>
      <${Cfg} rec=${rec}/>
      <${TapeMod} rec=${rec}/>
      <${ScopeMod} rec=${rec}/>
    </div>
    <div class="col">
      <${Meters} rec=${rec}/>
      <${ReceiptsPanel} rec=${rec}/>
      <${SpanForensics} rec=${rec}/>
      <${LieDetector} rec=${rec}/>
      <${Steer} rec=${rec}/>
      <${Minfl} rec=${rec}/>
      <${SectionsInfl} rec=${rec}/>
    </div>
  </div>`;
}

/* the play engine: one interval, driven by store.playing/speed */
function usePlayEngine(){
  const playing = useStore(x => x.playing);
  const speed = useStore(x => x.speed);
  useEffect(() => {
    if(!playing) return;
    const iv = setInterval(() => {
      const s = store.get(), n = normSteps(s.rec || {}).length;
      if(s.P >= n){ store.set({ playing: false }); return; }
      store.set({ P: s.P + 1 });
    }, 220 / speed);
    return () => clearInterval(iv);
  }, [playing, speed]);
}

/* ───────────────────────── CRT monitor — replay device AND live terminal (W2) ───────────── */
function Monitor({ rec }){
  const P = useStore(x => x.P);
  const playing = useStore(x => x.playing);
  const chatting = useStore(x => x.chatting);
  const chatBuf = useStore(x => x.chatBuf);
  const chatPrompt = useStore(x => x.chatPrompt);
  const liveLens = useStore(x => x.liveLens);
  const trust = useStore(x => rec ? x.trust[rec.id] : null);
  const steps = rec ? normSteps(rec) : [];
  const done = P >= steps.length;
  const trunc = rec && rec.finish_reason === "length";
  const liveView = chatting || chatBuf != null;   /* streaming, or streamed & awaiting the journal */
  const stLabel = chatting ? "LIVE" : chatBuf != null ? "SAVING"
    : !rec ? "IDLE" : playing ? "PLAY" : done ? "END" : P === 0 ? "IDLE" : "PAUSE";

  /* F2 trust shading: fetch the journal-calibrated spans once per live run (pure journal math) */
  useEffect(() => {
    if(!rec || rec._sample || !store.get().live) return;
    if(store.get().trust[rec.id] !== undefined) return;
    let dead = false;
    (async () => {
      const r = await api.trustSpans(rec.id);
      if(dead) return;
      store.set(st => ({ trust: { ...st.trust, [rec.id]: (r && r.available) ? r : null } }));
    })();
    return () => { dead = true; };
  }, [rec && rec.id]);
  /* token index -> its trust span (start/end are TOKEN indices per contracts) */
  const spanOf = i => trust && (trust.spans || []).find(sp => i >= sp.start && i < sp.end);
  const shadedUpto = () => steps.slice(0, P).map((s, i) => {
    const sp = spanOf(i);
    if(!sp || sp.trusted_rate_estimate == null) return html`<span key=${i}>${s.piece}</span>`;
    const tr = sp.trusted_rate_estimate;
    const amber = tr < .55, faint = .55 + .45 * Math.min(1, Math.max(0, tr));
    return html`<span key=${i}
      style=${"opacity:" + faint.toFixed(2)
        + (amber ? ";border-bottom:1px dotted rgba(255,196,120,.75);cursor:pointer" : "")}
      title=${"kept " + Math.round(tr*100) + "% of the time at this confidence in your journal ("
        + sp.bin_n + " runs" + (sp.small_n ? ", small bin — weak evidence" : "") + ") · proxy, not a fact-check"}
      onClick=${amber ? (() => store.set({ verbResult: { verb: "trust", ok: true,
        msg: `“${(sp.text || "").slice(0, 48)}” — conf ${(+sp.mean_conf).toFixed(2)}, kept `
           + `${Math.round(tr*100)}% of the time in your own journal (${sp.bin_n} runs`
           + `${sp.small_n ? ", small bin" : ""}). Acceptance proxy, not a fact-check — press PROVE `
           + `for causal receipts.` } })) : null}>${s.piece}</span>`;
  });

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">output monitor</span>
      <span class="tail">${liveView ? "streaming" : rec ? "tokens 0–" + steps.length : "—"}</span>
      <span class=${"tag " + (liveView ? "der-t" : "cap-t")}>${liveView ? "LIVE" : "CAPTURED"}</span></div>
    <div class="crt-shell"><div class="crt">
      <div class="scan"></div>
      <div class=${"st" + (chatting || playing ? "" : " idle")}><i></i><span>${stLabel}</span></div>
      <div class="sid">${liveView ? "streaming…" : rec ? (rec.id || "—") + " · " + shortTime(rec.created_at) : "no run"}</div>
      <div class="crt-text">
        ${liveView
          ? html`${chatPrompt && html`<span class="dim">⟨you⟩ ${chatPrompt}\n</span>`}${chatBuf || ""}${html`<span class="cursor">▋</span>`}`
          : !rec ? html`<span class="dim">no tape on the deck — say something below…</span>`
          : P === 0 ? html`<span class="dim">awaiting playback…</span>`
          : html`${trust ? shadedUpto() : steps.slice(0, P).map(s => s.piece).join("")}${(done && !playing) ? "" : html`<span class="cursor">▋</span>`}`}
      </div>
    </div></div>
    ${liveView && liveLens && html`<div class="cfg" style="margin:6px 13px 0;border-left-color:var(--lilac)">
      ${liveLens.error
        ? html`<span class="cap">live lens</span><span>${liveLens.error}</span>`
        : html`<span class="cap">disposed L${liveLens.layer}</span>
          ${(liveLens.words || []).map((w,k) => html`<span class=${"jchip d" + Math.min(3, k+1)}
            style="font-size:9.5px">${w.piece || "·"}</span>`)}
          <span style="margin-left:auto;font-size:8px;color:var(--mist)">J-lens, mid-generation · a
            disposition, not a verified thought · content-only</span>`}
    </div>`}
    <${ChatBar} rec=${rec}/>
    <div class="crt-foot">
      ${liveView ? html`<span>the reply is landing in the journal when the stream ends — no run id rides the stream, by design</span>`
        : rec ? html`
          <span>finish <b>${rec.finish_reason || "?"}</b></span>
          <span>tokens <b>${steps.length}</b></span>
          ${rec._sample ? html`<span style="color:var(--mist)">sample reel</span>`
                        : html`<span>recorded on this machine</span>`}
          ${trunc && html`<span class="tag fail-t">TRUNCATED</span>`}
          ${trust && html`<span style="color:var(--mist)">shading = your journal's acceptance rate at
            this confidence — proxy, not a fact-check</span>`}`
        : html`<span>nothing recorded yet</span>`}
    </div>
  </div>`;
}

/* ── the live chat (W2): SSE deltas type onto the CRT; the run is adopted from the journal
      after [DONE] (contracts §18: the stream carries no run id, deliberately) ── */
let chatAbortFn = null;
async function doSend(rec, text, cont){
  const s = store.get();
  if(!s.live){ toast("chat — live server only (this is the sample reel)"); return; }
  if(s.chatting) return;
  const prevTop = (s.runs[0] || {}).id;
  const messages = (cont && rec && Array.isArray(rec.messages) ? rec.messages : [])
    .concat([{ role: "user", content: text }]);
  const lensLayer = s.lensLayer || 0;                 /* F1: 0 = off; else the requested J-lens depth */
  store.set({ chatting: true, chatBuf: "", chatPrompt: text, playing: false, liveLens: null });
  chatAbortFn = api.chatStream(messages, {
    lens: lensLayer ? { layer: lensLayer, topk: 4 } : null,
    onLens: fr => {                                    /* F1: the disposed-to-say readout, mid-stream */
      if(fr && fr.error){ store.set({ liveLens: { error: String(fr.error) } }); return; }
      const row = (fr.readouts && fr.readouts[0]) || [];
      store.set({ liveLens: { layer: fr.layer, t: fr.t,
        words: row.map(x => ({ piece: String(x.piece || "").trim(), score: +x.score })) } });
    },
    onDelta: chunk => store.set(st => ({ chatBuf: (st.chatBuf || "") + chunk })),
    onDone: async () => {
      store.set({ chatting: false });
      for(let i = 0; i < 14; i++){                       /* poll the journal for the new run */
        await new Promise(r => setTimeout(r, 450));
        const l = await api.listRuns();
        if(l && Array.isArray(l.runs) && l.runs.length && l.runs[0].id !== prevTop){
          store.set({ runs: l.runs, chatBuf: null, chatPrompt: null });
          await loadRun(l.runs[0].id);
          toast("run " + l.runs[0].id + " recorded — on the tape");
          return;
        }
      }
      store.set({ chatBuf: null, chatPrompt: null });
      toast("stream ended but the run hasn't appeared in the journal yet — it should land on the next poll");
    },
    onError: err => { store.set({ chatting: false, chatBuf: null, chatPrompt: null });
      toast("chat stream failed: " + err); },
  });
}
const LENS_STOPS = [0, 2, 14, 21, 25];   /* 0 = off, else the fitted J-lens depths */
function ChatBar({ rec }){
  const chatting = useStore(x => x.chatting);
  const live = useStore(x => x.live);
  const lensLayer = useStore(x => x.lensLayer);
  const [text, setText] = useState("");
  const [cont, setCont] = useState(false);
  const send = () => {
    if(store.get().chatting) return;              /* double-send guard (review finding) */
    const t = text.trim(); if(!t) return;
    doSend(rec, t, cont); setText("");
  };
  return html`<div class="chatbar">
    <input type="text" placeholder=${live
        ? "say something — it types onto the monitor and lands in the journal"
        : "live server only — this is the sample reel"}
      value=${text} disabled=${chatting || !live}
      onInput=${e => setText(e.target.value)}
      onKeyDown=${e => { if(e.key === "Enter") send(); }}/>
    <button class="spd" disabled=${!live || chatting}
      style=${lensLayer ? "color:#7A6FB8;border-color:rgba(154,146,200,.7)" : ""}
      title="live lens: stream the J-lens disposed-to-say readout per token while it generates (slows decoding a little)"
      onClick=${() => {
        const next = LENS_STOPS[(LENS_STOPS.indexOf(lensLayer) + 1) % LENS_STOPS.length];
        store.set({ lensLayer: next });
        toast(next ? "live lens ON — layer " + next + " (disposed-to-say, content-only)" : "live lens off");
      }}>${lensLayer ? "◉ LENS L" + lensLayer : "○ LENS"}</button>
    ${chatting
      ? html`<button class="spd" style="color:#C24A31;border-color:rgba(242,109,79,.6)"
          onClick=${() => { chatAbortFn && chatAbortFn();
            store.set({ chatting: false, chatBuf: null, chatPrompt: null });
            toast("stream aborted"); }}>ABORT</button>`
      : html`<button class="spd" disabled=${!live} onClick=${send}>SEND ⏎</button>`}
    ${rec && Array.isArray(rec.messages) && rec.messages.length
      ? html`<label><input type="checkbox" checked=${cont}
          onChange=${e => setCont(e.target.checked)}/> continue this run</label>` : null}
  </div>`;
}

/* ───────────────────────── logs (derived from the record — never invented) ── */
function Logs({ rec }){
  const t = shortTime(rec.created_at) || "—";
  const mem = rec.memory || {};
  const dials = Object.entries((rec.behavior || {}).active_dials || {}).filter(([,v]) => v);
  const rows = [];
  const row = (col, tag, msg, ok = true) => rows.push({ col, tag, msg, ok });
  row("var(--blue)", "RUN", "recorded — " + (rec.id || ""));
  if((mem.cards_applied || []).length)
    row("var(--teal)", "MEMORY", `block injected · gate ${mem.gate != null ? (+mem.gate).toFixed(2) : "—"} · ${mem.cards_applied.length} card(s)`);
  if((mem.anchored || []).length)
    row("var(--lilac)", "ANCHOR", `J-space steer · L${mem.anchored_layer || 21} · ${mem.anchored.length} bag(s)`);
  if(mem.anchored_skipped)
    row("var(--coral)", "ANCHOR", mem.anchored_skipped, false);
  if(dials.length) row("var(--lilac2)", "STEER", "dials · " + dials.map(([k,v]) => k + " " + (+v).toFixed(1)).join(" · "));
  (rec.tiny_tests || []).forEach(tt =>
    row(tt.pass ? "var(--teal)" : "var(--coral)", "TEST", `expectation “${tt.name || "?"}” · ${tt.kind || "static"}`, !!tt.pass));
  if(rec.error) row("var(--coral)", "ERROR", String(rec.error).slice(0, 120), false);
  const trunc = rec.finish_reason === "length";
  row(trunc ? "var(--coral)" : "var(--teal)", "FINISH", rec.finish_reason || "?", !trunc);
  return html`<div class="mod">
    <span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led blue beats"></span><span class="cap">local logs</span>
      <span class="tail">${rows.length} entries</span></div>
    <div class="logbox">
      ${rows.map(r => html`<div class="logline">
        <span class="t">${t}</span>
        <span class="lt" style=${"color:" + r.col}>${r.tag}</span>
        <span class="m">${r.msg}
          <span class=${r.ok ? "ok" : "fail"} style="float:right">${r.ok ? "✓" : "fail"}</span></span>
      </div>`)}
    </div>
  </div>`;
}

function Cfg({ rec }){
  const d = (rec.meta && rec.meta.decode) || {};
  const bit = (k, v) => v != null && html`<span class="dot">·</span><span>${k} <b>${String(v)}</b></span>`;
  return html`<div class="cfg">
    <span class="cap">run</span><b>${rec.model || "—"}</b>
    ${bit("quant", d.quant)}${bit("decode", d.mode)}${bit("temp", d.temperature)}${bit("seed", d.seed)}
    ${bit("finish", rec.finish_reason || "?")}
    <span style="margin-left:auto" class="tag cap-t">CAPTURED</span>
  </div>`;
}

/* ───────────────────────── the tape module ───────────────────────── */
function TapeMod({ rec }){
  return html`<div class="mod" style="margin-top:0">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">replay tape</span>
      <span class="tail"><${RunTail}/></span></div>
    <${RunStrip}/>
    <${Transport} rec=${rec}/>
    <${VerbResult}/>
    <${Leader} rec=${rec}/>
    <${Arrangement} rec=${rec}/>
    <${JlensProvenance}/>
    <${LineageTree} rec=${rec}/>
  </div>`;
}
function RunTail(){
  const s = useStore(x => ({ n: x.runs.length, live: x.live }));
  return s.n ? (s.live ? s.n + " recorded" : "sample reel") : "—";
}

function RunStrip(){
  const s = useStore(x => ({ runs: x.runs, currentId: x.currentId, live: x.live, full: x.full }));
  return html`<div class="runstrip">
    ${s.runs.map(r => html`<button key=${r.id} class=${"runchip" + (s.currentId === r.id ? " on" : "")}
        onClick=${() => loadRun(r.id)}>
      <span class="t">${r.prompt_summary || r.id}</span>
      <${Thumb} id=${r.id}/>
      <span class="m">${shortTime(r.created_at)} · ${r.finish_reason || ""}
        ${r.parent_run_id && html` <span class="child">⑂ child</span>`}
        ${(r.flags || []).includes("error") && html` <span style="color:var(--coral)">· error</span>`}</span>
    </button>`)}
  </div>`;
}
function Thumb({ id }){
  const ref = useRef(null);
  const full = useStore(x => x.full[id]);
  useEffect(() => {
    let dead = false;
    (async () => {
      let rec = full;
      if(!rec && store.get().live){
        const r = await api.getRun(id); rec = r && (r.run || r);
        if(rec) store.set(st => ({ full: { ...st.full, [id]: rec } }));
      }
      if(dead || !rec || !ref.current) return;
      const conf = (rec.trace || {}).confidence || normSteps(rec).map(s => s.conf ?? 0);
      const cv = ref.current;
      if(!conf.length){ cv.style.display = "none"; return; }
      const x = cv.getContext("2d"), W = cv.width, H = cv.height, n = conf.length;
      x.clearRect(0,0,W,H);
      const gr = x.createLinearGradient(0,0,W,0);
      gr.addColorStop(0,"rgba(44,191,232,.85)"); gr.addColorStop(1,"rgba(95,200,188,.85)");
      x.fillStyle = gr;
      conf.forEach((c,i) => { const a = Math.max(1,(c||0)*(H-2)/2);
        x.fillRect(i*(W/n), H/2 - a, Math.max(1, W/n - 1), a*2); });
    })();
    return () => { dead = true; };
  }, [id, full]);
  return html`<canvas ref=${ref} width="136" height="15"></canvas>`;
}

/* ── transport: the verbs are LIVE ── */
function Transport({ rec }){
  const P = useStore(x => x.P);
  const playing = useStore(x => x.playing);
  const speed = useStore(x => x.speed);
  const busy = useStore(x => x.busy);
  const steps = normSteps(rec);
  const n = steps.length;
  const stops = (() => {
    const s = [0];
    const shaky = steps.findIndex(x => (x.conf ?? 1) < .5);
    if(shaky > 0) s.push(shaky);
    s.push(n);
    return [...new Set(s)].sort((a,b) => a-b);
  })();
  const cur = steps[Math.max(0, P-1)];
  const setP = v => store.set({ P: Math.max(0, Math.min(n, v)), playing: false });

  return html`<div class="transport">
    <div class="tbtns">
      <button class="tbtn" onClick=${() => setP(0)}>⇤</button>
      <button class="tbtn" onClick=${() => setP(stops.filter(x => x < P).pop() ?? 0)}>◂◂</button>
      <button class="tbtn play" onClick=${() => {
        if(playing){ store.set({ playing: false }); return; }
        store.set({ P: P >= n ? 0 : P, playing: true });
      }}>${playing ? "❚❚" : "▶"}</button>
      <button class="tbtn" onClick=${() => setP(stops.find(x => x > P) ?? n)}>▸▸</button>
      <button class="tbtn" onClick=${() => setP(n)}>⇥</button>
    </div>
    <div class="pos"><span class="big">${P}</span>
      <span class="of">/ ${n} · ${P ? `“${(cur.piece || "").trim() || "·"}” conf ${cur.conf != null ? cur.conf.toFixed(2) : "—"}` : "start"}</span></div>
    <div class="right">
      <button class="spd" onClick=${() => store.set({ speed: speed === 1 ? 2 : speed === 2 ? 4 : speed === 4 ? .5 : 1 })}>${speed.toFixed(1)}×</button>
      <button class=${"spd" + (busy.rederive ? " busy" : "")} onClick=${() => doRederive(rec)}>⟲ RE-DERIVE</button>
      <button class=${"spd" + (busy.replay ? " busy" : "")} onClick=${() => doReplay(rec)}>⏵ REPLAY</button>
      <button class=${"spd" + (busy.branch ? " busy" : "")} onClick=${() => doBranch(rec)}>⑂ BRANCH</button>
      <button class=${"spd" + (busy.prove ? " busy" : "")}
        style="color:#1B7F74;border-color:rgba(95,200,188,.7)" onClick=${() => doProve(rec)}>
        ${busy.prove ? "PROVING…" : "PROVE"}</button>
      <button class="spd" onClick=${() => {
        if(rec._sample) return toast("EXPORT — live runs only (this is the sample reel)");
        window.open(api.exportUrl(rec.id), "_blank");
      }}>⤓ EXPORT</button>
      <button class="spd" title="one self-contained HTML receipt card — save it, post it"
        onClick=${() => {
          if(rec._sample) return toast("CARD — live runs only (this is the sample reel)");
          window.open(api.cardUrl(rec.id), "_blank");
        }}>⧉ CARD</button>
    </div>
  </div>`;
}

/* verb result strip — outcomes reported plainly, failures included */
function VerbResult(){
  const vr = useStore(x => x.verbResult);
  if(!vr) return null;
  return html`<div class="cfg" style="margin:8px 13px 0;border-left-color:${vr.ok ? "var(--teal)" : "var(--coral)"}">
    <span class="cap" style=${vr.ok ? "" : "color:#C24A31"}>${vr.verb}</span>
    <span>${vr.msg}</span>
    <button class="spd" style="margin-left:auto;height:21px;font-size:8.5px"
      onClick=${() => store.set({ verbResult: null })}>dismiss</button>
  </div>`;
}
const setBusy = (verb, v) => store.set(st => ({ busy: { ...st.busy, [verb]: v } }));
const guardSample = rec => {
  if(rec._sample){ toast("Live runs only — this is the sample reel"); return true; }
  return false;
};

async function doRederive(rec){
  if(guardSample(rec)) return;
  setBusy("rederive", true); toast("re-deriving — teacher-forced re-scoring under the run's own conditions…");
  const res = await api.rederive(rec.id, {});
  setBusy("rederive", false);
  if(!res){ store.set({ verbResult: { verb: "re-derive", ok: false,
    msg: "no answer — rederive needs the engine substrate (score_tokens); is the engine up?" } }); return; }
  /* contracts §3: {text, steps:[{piece,token_id,logprob,conf}], meta:{retokenized, block_source, n_tokens}} */
  const m = res.meta || {};
  const confs = (res.steps || []).map(s => s.conf).filter(c => c != null);
  const mean = confs.length ? (confs.reduce((a,b) => a+b, 0) / confs.length) : null;
  store.set({ verbResult: { verb: "re-derive", ok: true,
    msg: `re-scored ${m.n_tokens ?? (res.steps || []).length} tokens deterministically · mean conf ${mean != null ? mean.toFixed(3) : "—"}`
       + ` · ${m.retokenized ? "retokenized (boundary-approximate)" : "exact token ids"}`
       + (m.block_source ? ` · conditions from ${m.block_source}` : "") } });
}

/* replay/branch both return THE FULL CHILD RUN RECORD (contracts §4/§5) */
async function adoptChild(res, verb, extra){
  const l = await api.listRuns();
  store.set(st => ({
    runs: (l && Array.isArray(l.runs)) ? l.runs : st.runs,
    full: { ...st.full, [res.id]: res },
    currentId: res.id, rec: res, P: 0, receipts: null, leaning: null,
    verbResult: { verb, ok: true, msg: `child ${res.id} recorded — now on the tape (parent: ${res.parent_run_id || "?"})${extra || ""}` },
  }));
}
async function doReplay(rec){
  if(guardSample(rec)) return;
  setBusy("replay", true); toast("replaying — greedy, state-safe…");
  /* contracts §4: the knobs live INSIDE changes_applied; greedy:true forces attributable decode */
  const res = await api.replay(rec.id, { changes_applied: { greedy: true } });
  setBusy("replay", false);
  if(!res || !res.id){ store.set({ verbResult: { verb: "replay", ok: false,
    msg: "replay didn't answer — needs a chat-capable substrate; is the engine up?" } }); return; }
  await adoptChild(res, "replay", " · greedy");
}
async function doBranch(rec){
  if(guardSample(rec)) return;
  const edited = window.prompt("Branch — edit the user turn (regenerates from there, greedy):", firstLine(rec));
  if(edited == null) return;
  /* contracts §5: turn = the USER-TURN ordinal; single-turn studio runs branch at their last user turn */
  const msgs = rec.messages || rec.assembled_messages || [];
  const userTurns = msgs.filter(x => x.role === "user").length;
  const turn = Math.max(0, userTurns - 1);
  setBusy("branch", true); toast("branching at turn " + turn + "…");
  const res = await api.branch(rec.id, { turn, alt_user: edited, sample: false });
  setBusy("branch", false);
  if(!res || !res.id){ store.set({ verbResult: { verb: "branch", ok: false,
    msg: "branch didn't answer (turn out of range, or generation error)" } }); return; }
  await adoptChild(res, "branch");
}

async function doProve(rec){
  if(guardSample(rec)) return;
  setBusy("prove", true);
  toast("asking what mattered — leave-one-out both-arms-greedy + forced scoring (mode: both)…");
  const res = await api.receipts(rec.id, { mode: "both" });   // contracts §7
  setBusy("prove", false);
  if(!res){ store.set({ verbResult: { verb: "prove", ok: false,
    msg: "receipts didn't answer — they regenerate arms and can take minutes; is the engine up?" } }); return; }
  const list = res.receipts || [];
  const forced = res.forced_receipts || [];
  const key = inf => JSON.stringify(inf || {});
  const forcedBy = new Map(forced.map(f => [key(f.influence), f]));
  /* leaning heat: the strongest causally-verified forced receipt's per-token |deltas| (contracts §6) */
  let leaning = null, leaningLabel = null;
  for(const f of forced){
    if(!f.causal_verified || !Array.isArray(f.deltas)) continue;
    if(!leaning || Math.abs(f.sum_nats || 0) > Math.abs(leaningLabel.sum_nats || 0)){
      leaning = f.deltas; leaningLabel = f;
    }
  }
  store.set({
    receipts: { raw: res, list, forcedBy, skipped: res.skipped || [],
                redundant: res.redundant_pairs || [], keyFn: key },
    leaning, leaningInfluence: leaningLabel ? leaningLabel.influence : null,
    verbResult: { verb: "prove", ok: true,
      msg: `${list.length} influence(s) measured, ${forced.length} force-scored`
         + (res.skipped && res.skipped.length ? ` · ${res.skipped.length} skipped (honestly)` : "")
         + (leaning ? " · forced Δ painted on the clips" : "") },
  });
}

/* ── leader ── */
function Leader({ rec }){
  const fp = rec.final_prompt || null, am = rec.assembled_messages;
  const body = fp || (Array.isArray(am) ? am.map(x => `⟨${x.role}⟩ ${x.content}`).join("\n")
    : Array.isArray(rec.messages) ? rec.messages.map(x => `⟨${x.role}⟩ ${x.content}`).join("\n") : "");
  if(!body) return null;
  return html`<details class="leader">
    <summary>Leader — the exact prompt the model saw
      <span style="margin-left:auto">${fp ? "final_prompt" : "messages"}</span></summary>
    <div class="leader-body">${body}</div>
  </details>`;
}

/* ───────────────────────── arrangement ───────────────────────── */
function Arrangement({ rec }){
  const P = useStore(x => x.P);
  const leaning = useStore(x => x.leaning);
  const steps = normSteps(rec);
  const w = weightsFor(steps);
  const cols = colsFor(w);
  const innerW = Math.max(680, w.reduce((a,b) => a+b, 0) * 8.5 + 130);
  const hasEnt = steps.some(s => s.ent != null);
  const mem = rec.memory || {};
  const arrRef = useRef(null);
  const [sel, setSel] = useState(null);
  const [jl, setJl] = useState(null);
  const [jlBusy, setJlBusy] = useState(false);
  useEffect(() => { setJl(null); setSel(null); }, [rec.id]);

  /* playhead + shade positioning */
  const phRef = useRef(null), shadeRef = useRef(null);
  const plateW = 118, pad = 13;
  useEffect(() => {
    const arr = arrRef.current, ph = phRef.current;
    if(!arr || !ph) return;
    const W = arr.clientWidth - plateW - pad*2;
    const g = colGeom(w, W);
    const x = P >= steps.length ? (g.length ? g[g.length-1].x1 : 0) : (P <= 0 ? 0 : g[P-1].x1);
    ph.hidden = false;
    ph.style.left = (plateW + pad + x) + "px";
  }, [P, rec.id]);
  useEffect(() => {
    const arr = arrRef.current, shade = shadeRef.current;
    if(!arr || !shade) return;
    const move = e => {
      const r = arr.getBoundingClientRect(), x = e.clientX - r.left - plateW - pad;
      if(x < 0){ shade.hidden = true; return; }
      const g = colGeom(w, arr.clientWidth - plateW - pad*2);
      const i = g.findIndex(c => x >= c.x0 && x < c.x1);
      if(i < 0){ shade.hidden = true; return; }
      shade.hidden = false;
      shade.style.left = (plateW + pad + g[i].x0) + "px";
      shade.style.width = (g[i].x1 - g[i].x0) + "px";
    };
    const leave = () => { shade.hidden = true; };
    const dbl = e => {
      const r = arr.getBoundingClientRect(), x = e.clientX - r.left - plateW - pad;
      if(x < 0) return;
      const g = colGeom(w, arr.clientWidth - plateW - pad*2);
      const i = g.findIndex(c => x >= c.x0 && x < c.x1);
      if(i >= 0) store.set({ P: i + 1, playing: false });
    };
    arr.addEventListener("mousemove", move);
    arr.addEventListener("mouseleave", leave);
    arr.addEventListener("dblclick", dbl);
    return () => { arr.removeEventListener("mousemove", move);
      arr.removeEventListener("mouseleave", leave); arr.removeEventListener("dblclick", dbl); };
  }, [rec.id]);

  const leanMax = leaning ? Math.max(1e-6, ...leaning.map(Math.abs)) : 1;

  return html`<div class="arr-wrap"><div class="arr" ref=${arrRef} style=${"width:" + innerW + "px"}>
    <div class="playhead" ref=${phRef} hidden></div>
    <div class="colshade" ref=${shadeRef} hidden></div>

    <div class="lane ruler">
      <div class="plate"><span><span class="cap">ruler</span><span class="sub">index · locators</span></span></div>
      <div class="lane-body">
        <div class="cells" style=${"grid-template-columns:" + cols + ";height:100%"}>
          ${steps.map((s,i) => html`<div class=${"tick" + (i % 5 === 0 ? " five" : "")}>
            ${i % 5 === 0 && html`<span class="idx">${i}</span>`}</div>`)}
        </div>
        <${Locators} steps=${steps} mem=${mem} rec=${rec} w=${w} innerW=${innerW}/>
      </div>
    </div>

    <div class="lane clips">
      <div class="plate"><span><span class="cap">run</span><span class="sub">token clips${leaning ? " · forced Δ" : ""}</span></span></div>
      <div class="lane-body"><div class="cells" style=${"grid-template-columns:" + cols}>
        ${steps.map((s,i) => html`<div key=${i}
            class=${"clip-t" + ((s.conf ?? 1) < .55 ? " shaky" : "") + (sel === i ? " sel" : "")}
            style=${"--c:" + (s.conf ?? .6).toFixed(2)}
            onClick=${e => { e.stopPropagation(); setSel(sel === i ? null : i); }}>
          ${(s.piece || "").trim() || "·"}
          ${leaning && leaning[i] != null && html`<span class="lean"
            style=${"--l:" + Math.round(Math.abs(leaning[i]) / leanMax * 14)}></span>`}
        </div>`)}
      </div>
      ${sel != null && html`<${Pop} step=${steps[sel]} i=${sel} w=${w} plateW=${plateW} pad=${pad} arrRef=${arrRef} rec=${rec}
          jl=${jl} jlBusy=${jlBusy} onReadJl=${() => loadJl(rec, setJl, setJlBusy, 21)}/>`}
      </div>
    </div>

    <div class="lane wave">
      <div class="plate"><span><span class="cap">confidence</span><span class="sub">CAPTURED</span></span></div>
      <div class="lane-body"><${WaveCanvas} steps=${steps} w=${w}/></div>
    </div>

    ${hasEnt && html`<div class="lane auto">
      <div class="plate"><span><span class="cap">entropy</span><span class="sub">DERIVED</span></span></div>
      <div class="lane-body"><${EntropySvg} steps=${steps} w=${w}/></div>
    </div>`}

    <div class="lane jl">
      <div class="plate"><span><span class="cap">disposed</span><span class="sub">J-lens · DERIVED</span></span>
        ${jl && (jl.layers || []).length ? html`<span style="margin-left:auto">
          ${(jl.layers).map(L => html`<button class="mono"
            style=${"font-size:7.5px;padding:0 3px;color:" + (L == jl.layer ? "var(--navy)" : "var(--mist)")}
            onClick=${() => loadJl(rec, setJl, setJlBusy, L)}>${L}</button>`)}</span>` : null}
      </div>
      <div class="lane-body">
        ${jl ? html`<div class="cells" style=${"grid-template-columns:" + cols + ";align-items:center"}>
            ${steps.map((s,i) => {
              const c = jl.byPos[i];
              return c && c.length
                ? html`<div class="jcell">${c.slice(0,3).map((t,k) =>
                    html`<span class=${"jchip d" + (k+1)}>${String(t).trim()}</span>`)}</div>`
                : html`<div></div>`;
            })}
          </div>`
        : jlBusy ? html`<div class="lane-action"><span class="none">listening back…</span></div>`
        : jl === false ? html`<div class="lane-action"><span class="none">no lens fitted for this model — blank ≠ nothing, but there is nothing fitted to read with</span></div>`
        : html`<div class="lane-action"><button onClick=${() => loadJl(rec, setJl, setJlBusy)}>Read dispositions</button></div>`}
      </div>
    </div>
  </div></div>`;
}

function Locators({ steps, mem, rec, w, innerW }){
  const W = innerW - 118 - 26, g = colGeom(w, W);
  const shaky = steps.findIndex(s => (s.conf ?? 1) < .5);
  return html`
    ${((mem.cards_applied || []).length || (mem.anchored || []).length) ? html`<span class="loc" style=${"left:" + (g[0].x0 + 2) + "px"}>
      ▸ ${(mem.anchored || []).length ? "anchored" : "memory"}${mem.gate != null ? " " + (+mem.gate).toFixed(2) : ""}</span>` : null}
    ${shaky >= 0 && html`<span class="loc" style=${"left:" + g[shaky].x0 + "px"}>▸ low conf</span>`}
    <span class=${"loc" + (rec.finish_reason === "length" ? " bad" : "")} style="right:3px">
      ▸ ${rec.finish_reason || "end"}</span>`;
}

/* ── click-a-span SIGNALS popover — the entry point: click a token of the reply, see its signals,
      one-click into an Experiment. Every signal renders ONLY when its data exists (honest absence,
      never a fabricated value); every honesty label matches replay.mjs / experiment.mjs. `jl` is the
      Arrangement's loaded J-lens (byPos/layer), or `false` (no lens fitted), or null (not read yet). ── */
function Pop({ step, i, w, plateW, pad, arrRef, rec, jl, jlBusy, onReadJl }){
  const ref = useRef(null);
  const forkBusy = useStore(x => x.busy.fork);
  useEffect(() => {
    const arr = arrRef.current, el = ref.current;
    if(!arr || !el) return;
    const g = colGeom(w, arr.clientWidth - plateW - pad*2);
    el.style.left = Math.max(4, Math.min(arr.clientWidth - 244, plateW + pad + g[i].xc - 116)) + "px";
    el.style.top = "34px";
  }, [i]);
  const alts = (step.alts || []).slice(0, 3);
  const spanText = (step.piece || "").trim();
  const disp = (jl && jl.byPos) ? jl.byPos[i] : null;          // disposed pieces at this position (or null)
  const lensAbsent = jl === false;                             // a lens was asked for and honestly isn't fitted
  const mem = rec.memory || {};
  const cards = mem.cards_applied || [], appliedIds = mem.applied_ids || [];
  const dials = Object.entries((rec.behavior || {}).active_dials || {}).filter(([, v]) => v);

  /* F3: click an almost-said token to FORK reality from this position (greedy continuation) */
  const fork = async piece => {
    if(rec && guardSample(rec)) return;
    if(store.get().busy.fork) return;
    setBusy("fork", true); toast(`forking at token ${i} — “${piece.trim()}” instead, greedy from there…`);
    const res = await api.fork(rec.id, i, piece);
    setBusy("fork", false);
    if(!res || !res.id){ store.set({ verbResult: { verb: "fork", ok: false,
      msg: "fork didn't answer — is the engine up?" } }); return; }
    await adoptChild(res, "fork", ` · “${piece.trim()}” at ${i}`);
  };
  /* the ACTION: deep-link into the Experiment drawer, pre-filled, via the store handoff */
  const openExp = (ctype, fields) => {
    if(rec._sample){ toast("live runs only — this is the sample reel"); return; }
    store.set({ pendingExperiment: { ctype, fields: fields || {}, method: "" }, route: "experiment" });
    toast("opening Experiment — " + ctype + " pre-filled");
  };
  const userTurns = (rec.messages || rec.assembled_messages || []).filter(x => x.role === "user").length;
  const forkTurn = Math.max(0, userTurns - 1);

  return html`<div class="pop signals" ref=${ref} onClick=${e => e.stopPropagation()}>
    <h4>signals · token ${i}</h4>

    <div class="sig-lbl">confidence — raw, uncalibrated</div>
    <div class="alt win"><span>${spanText || "·"}</span>
      <b title="raw model probability for the committed token — UNCALIBRATED; self-confidence is not correctness">
        ${step.conf != null ? step.conf.toFixed(2) : "—"}</b></div>

    <div class="sig-lbl">almost said${alts.length ? "" : " — none recorded"}</div>
    ${alts.map(a => {
        const piece = ((a.piece ?? a.text ?? "") + "");
        return html`<div class="alt">
          <span>${piece.trim() || "·"}</span>
          <span>${a.prob != null ? (+a.prob).toFixed(2) : ""}</span>
          ${piece && !rec._sample && html`<button class="spd"
            style="height:17px;font-size:7.5px;padding:0 5px" disabled=${forkBusy}
            onClick=${e => { e.stopPropagation(); fork(piece); }}>⑂ fork</button>`}
        </div>`; })}

    <div class="sig-lbl">disposed to say${jl && jl.layer != null ? " · J-lens L" + jl.layer : " · J-lens"}</div>
    ${lensAbsent
      ? html`<span class="none" style="font-size:9px">no lens fitted for this model — nothing to read with (blank ≠ nothing)</span>`
      : disp && disp.length
        ? html`<div class="sig-run">${disp.slice(0, 3).map(p =>
            html`<span class="jchip d1">${String(p).trim() || "·"}</span>`)}</div>`
        : (jl && jl.byPos)
          ? html`<span class="none" style="font-size:9px">no disposition read at this position</span>`
          : html`<button class="spd" style="font-size:8.5px" disabled=${jlBusy}
              onClick=${e => { e.stopPropagation(); onReadJl && onReadJl(); }}>
              ${jlBusy ? "reading…" : "read disposition"}</button>`}
    ${!lensAbsent && html`<div class="sig-note">a disposition read, NOT the literal thought · lens-blind to abstractions</div>`}

    <div class="sig-lbl">active on this run</div>
    <div class="sig-run">
      ${cards.length ? html`<span class="jchip">memory · ${cards.length} card${cards.length > 1 ? "s" : ""}</span>`
        : html`<span class="none" style="font-size:9px">memory: none rode</span>`}
      ${dials.length ? dials.map(([k, v]) => html`<span class="jchip">${k} ${(+v).toFixed(1)}</span>`)
        : html`<span class="none" style="font-size:9px">dials: none</span>`}
    </div>

    ${!rec._sample && html`<div class="sig-lbl">experiment on this span</div>
    <div class="sig-actions">
      <button class="spd" onClick=${() => openExp("edit_turn", { turn: forkTurn })}>fork from here</button>
      <button class="spd" onClick=${() => openExp("ablate_memory", {})}>turn memory off</button>
      ${spanText && html`<button class="spd"
        onClick=${() => openExp("swap_concept", { to_concept: spanText, from_hint: spanText })}>swap “${spanText}”</button>`}
      ${appliedIds.length && cards.length ? html`<button class="spd"
        onClick=${() => openExp("ablate_card", { card_id: appliedIds[0] })}>prove a card</button>` : null}
    </div>`}
  </div>`;
}

function WaveCanvas({ steps, w }){
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current; if(!cv) return;
    const cell = cv.parentElement;
    const W = cell.clientWidth, H = cell.clientHeight, dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W*dpr; cv.height = H*dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const geo = colGeom(w, W), mid = H/2;
    g.strokeStyle = "rgba(90,130,155,.25)"; g.beginPath(); g.moveTo(0,mid); g.lineTo(W,mid); g.stroke();
    const gr = g.createLinearGradient(0,0,W,0);
    gr.addColorStop(0,"rgba(44,191,232,.6)"); gr.addColorStop(1,"rgba(95,200,188,.6)");
    g.fillStyle = gr; g.strokeStyle = "rgba(27,127,116,.85)";
    const top = [], bot = [];
    geo.forEach((c,i) => {
      const amp = Math.max(1.5, (steps[i].conf ?? 0) * (H/2 - 4));
      const seg = Math.max(2, Math.floor((c.x1 - c.x0)/3));
      for(let k = 0; k <= seg; k++){
        const x = c.x0 + (c.x1 - c.x0)*k/seg;
        const jag = 1 - .22*Math.abs(Math.sin(k*2.7 + i*1.3));
        top.push([x, mid - amp*jag]); bot.push([x, mid + amp*jag]);
      }});
    g.beginPath(); top.forEach((p,i) => i ? g.lineTo(p[0],p[1]) : g.moveTo(p[0],p[1]));
    bot.reverse().forEach(p => g.lineTo(p[0],p[1])); g.closePath(); g.fill();
    g.beginPath(); top.forEach((p,i) => i ? g.lineTo(p[0],p[1]) : g.moveTo(p[0],p[1])); g.stroke();
  }, [steps, w]);
  return html`<canvas ref=${ref}></canvas>`;
}

function EntropySvg({ steps, w }){
  const ref = useRef(null);
  const [dims, setDims] = useState(null);
  useEffect(() => {
    if(ref.current) setDims({ W: ref.current.clientWidth, H: ref.current.clientHeight });
  }, [steps]);
  if(!dims) return html`<div ref=${ref} style="height:100%"></div>`;
  const { W, H } = dims;
  const geo = colGeom(w, W);
  const mx = Math.max(1e-6, ...steps.map(s => s.ent || 0));
  const pts = steps.map((s,i) => [geo[i].xc, H - 4 - ((s.ent ?? 0)/mx)*(H - 9)]);
  return html`<div ref=${ref} style="height:100%">
    <svg viewBox=${"0 0 " + W + " " + H} preserveAspectRatio="none" style="display:block;width:100%;height:100%">
      <polyline points=${pts.map(p => p.join(",")).join(" ")} fill="none" stroke="#9A92C8" stroke-width="1.3"/>
      ${pts.map(p => html`<circle cx=${p[0]} cy=${p[1]} r="2" fill="#EAF4F9" stroke="#9A92C8" stroke-width="1.1"/>`)}
    </svg></div>`;
}

/* jlens loading (module-level cache per run+layer) — contracts §9:
   POST-only; 200 + {available:false, reason} is HONEST ABSENCE, never an error;
   readouts index over the LENS's own tokenization of run.response (usually == trace, checked). */
const jlCache = new Map();
async function loadJl(rec, setJl, setBusy, layer){
  const key = rec.id + ":" + (layer ?? "default");
  if(jlCache.has(key)){ setJl(jlCache.get(key)); return; }
  setBusy(true); setJl(null);
  let data = rec._jlens || null;
  if(!data) data = await api.jlens(rec.id, layer);
  setBusy(false);
  if(!data){ setJl(false); store.set({ jlProvenance: null, jlReason: "the server didn't answer" }); return; }
  if(data.available === false){
    setJl(false); store.set({ jlProvenance: null, jlReason: data.reason || "no lens available" }); return;
  }
  const n = normSteps(rec).length;
  const byPos = Array.from({ length: n }, () => null);
  let lensN = n;
  if(data.chips) Object.entries(data.chips).forEach(([p,c]) => { if(byPos[+p] !== undefined) byPos[+p] = c; });
  else if(Array.isArray(data.readouts)){
    lensN = data.readouts.length;
    data.readouts.forEach((r,i) => {
      if(i < n && Array.isArray(r) && r.length) byPos[i] = r.map(x => x.piece ?? x.label ?? x); });
  }
  const pv = data.provenance;
  const p = (pv && typeof pv === "object") ? pv : {};
  const base = typeof pv === "string" ? pv
    : [p.note, p.fit_model ? "fitted: " + p.fit_model : null].filter(Boolean).join(" · ");
  const provText = [base || null,
                    lensN !== n ? `(lens read ${lensN} tokens vs ${n} in the trace — aligned by index)` : null]
                   .filter(Boolean).join(" · ") || null;
  const out = { byPos, layer: data.layer,
                layers: data.available_layers || data.layers || p.layers || [],
                provenance: provText };
  jlCache.set(key, out); setJl(out);
  store.set({ jlProvenance: provText, jlReason: null });
}
function JlensProvenance(){
  const pv = useStore(x => x.jlProvenance);
  const reason = useStore(x => x.jlReason);
  if(pv) return html`<div class="provenance"><b>J-lens</b> — ${pv}</div>`;
  if(reason) return html`<div class="provenance"><b>J-lens</b> — unavailable: ${reason} · blank ≠ nothing, but there is nothing fitted to read with</div>`;
  return null;
}

/* ───────────────────────── state scope ───────────────────────── */
function ScopeMod({ rec }){
  const canvasRef = useRef(null);
  const [rot, setRot] = useState(42);
  const [zoom, setZoom] = useState(100);
  const [focus, setFocus] = useState(3);
  const [meta, setMeta] = useState({ real: false, layerIds: [0,6,12,18,24] });
  const dataRef = useRef(null);

  useEffect(() => {
    let dead = false;
    /* per-pane glowing words: dedupe top-1 dispositions across positions, keep the 4 strongest */
    const topWords = read => {
      const best = {};
      ((read && read.readouts) || []).forEach(cell => {
        const t = cell && cell[0]; if(!t) return;
        const p = String(t.piece || "").trim(); if(!p) return;
        const s = +t.score || 0;
        if(!(p in best) || s > best[p]) best[p] = s;
      });
      const list = Object.entries(best).sort((a,b) => b[1] - a[1]).slice(0, 4);
      const mx = Math.max(1e-6, ...list.map(x => x[1]));
      return list.map(([piece, s]) => ({ piece, s01: s / mx }));
    };
    (async () => {
      const steps = normSteps(rec);
      const text = firstLine(rec);
      let norms = null, layerIds = null, real = false, words = null;
      let rawNorms = null;
      if(store.get().live && !rec._sample){
        const r = await api.engineLayers(text);
        if(!dead && r && Array.isArray(r.norms) && r.norms.length) rawNorms = r.norms;
        /* THE FUSION: read the same text through the lens at every fitted depth */
        const first = await api.jlensText(text);
        if(!dead && first && first.available !== false){
          const lensLayers = first.available_layers ||
            (first.provenance && first.provenance.layers) || [first.layer];
          const reads = { [first.layer]: first };
          for(const L of lensLayers){
            if(reads[L] || dead) continue;
            const rr = await api.jlensText(text, L);
            if(rr && rr.available !== false) reads[L] = rr;
          }
          layerIds = lensLayers.slice();
          words = layerIds.map(L => reads[L] ? topWords(reads[L]) : null);
        }
      }
      if(rawNorms){
        const nl = rawNorms.length;
        if(!layerIds){
          layerIds = [0, Math.floor(nl*.25), Math.floor(nl*.5), Math.floor(nl*.75), nl-1];
        }
        norms = layerIds.map(L => rawNorms[Math.min(nl - 1, Math.max(0, L))]);
        real = true;
      }
      if(!norms){
        if(!layerIds) layerIds = [0, 6, 12, 18, 24];
        norms = layerIds.map((L,li) => steps.map((s,i) =>
          (s.conf ?? .6) * (0.45 + 0.55*Math.abs(Math.sin(li*1.3 + i*.7)))));
      }
      if(dead) return;
      dataRef.current = { norms, layerIds, words };
      setMeta({ real, lens: !!(words && words.some(Boolean)), layerIds });
    })();
    return () => { dead = true; };
  }, [rec.id]);

  useEffect(() => {
    const d = dataRef.current, cv = canvasRef.current;
    if(!d || !cv) return;
    let raf = 0;
    const wrap = cv.parentElement;
    const W = wrap.clientWidth - 24, H = 230, dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W*dpr; cv.height = H*dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const { norms, layerIds, words } = d;
    const NP = norms.length;
    const paneW = 82*(zoom/100), paneH = 156*(zoom/100), skew = 10 + (rot/100)*26;
    const span = W - paneW - 56, x0 = 30;
    const PAL = [[76,141,240],[95,200,188],[143,168,232],[182,176,218],[127,180,240]];
    const mx = Math.max(1e-6, ...norms.flat());
    const colsN = 6, rowsN = 9;
    const grids = norms.map(row => {
      const out = [];
      for(let r = 0; r < rowsN; r++) for(let c = 0; c < colsN; c++){
        const t = (r*colsN + c)/(rowsN*colsN);
        out.push((row[Math.min(row.length-1, Math.floor(t*row.length))] || 0)/mx);
      } return out; });
    const quad = i => { const px = x0 + span*(i/(NP-1)), py = H/2;
      return { tl:[px, py-paneH/2 - skew*.4], tr:[px+paneW, py-paneH/2 + skew*.4],
               br:[px+paneW, py+paneH/2 + skew*.4], bl:[px, py+paneH/2 - skew*.4], y:py }; };
    function frame(ms){
      const T = ms*.001;
      g.clearRect(0,0,W,H);
      for(let i = 0; i < NP-1; i++){
        const a = quad(i), b = quad(i+1);
        for(let k = 0; k < 9; k++){
          const ya = a.y - paneH/2 + paneH*(k+.5)/9, yb = b.y - paneH/2 + paneH*((k*3.7)%9 + .5)/9;
          const c1 = PAL[i%PAL.length], c2 = PAL[(i+1)%PAL.length];
          const gr = g.createLinearGradient(a.tr[0], ya, b.tl[0], yb);
          gr.addColorStop(0, `rgba(${c1[0]},${c1[1]},${c1[2]},.30)`);
          gr.addColorStop(1, `rgba(${c2[0]},${c2[1]},${c2[2]},.30)`);
          g.strokeStyle = gr; g.lineWidth = .8;
          g.beginPath(); g.moveTo(a.tr[0]-4, ya);
          const mid = (a.tr[0] + b.tl[0])/2, sway = REDUCED ? 0 : Math.sin(T*.7 + i*2 + k)*5;
          g.bezierCurveTo(mid, ya + sway, mid, yb - sway, b.tl[0]+4, yb); g.stroke();
        }
      }
      for(let i = 0; i < NP; i++){
        const q = quad(i), c = PAL[i%PAL.length], isF = i === focus;
        const wl = (words && words[i]) || null;          /* the pane's glowing dispositions */
        g.beginPath(); g.moveTo(...q.tl); g.lineTo(...q.tr); g.lineTo(...q.br); g.lineTo(...q.bl); g.closePath();
        g.fillStyle = `rgba(255,255,255,${isF ? .55 : .35})`; g.fill();
        g.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${isF ? .95 : .55})`; g.lineWidth = isF ? 1.6 : 1; g.stroke();
        const cells = grids[i];
        const dotDim = wl && wl.length ? .5 : 1;          /* dots recede when the words are present */
        for(let r = 0; r < rowsN; r++) for(let cc = 0; cc < colsN; cc++){
          const u = (cc+.5)/colsN, v = (r+.5)/rowsN;
          const x = q.tl[0] + (q.tr[0]-q.tl[0])*u;
          const yTop = q.tl[1] + (q.tr[1]-q.tl[1])*u, yBot = q.bl[1] + (q.br[1]-q.bl[1])*u;
          const y = yTop + (yBot-yTop)*v;
          let a = cells[r*colsN+cc] * dotDim;
          if(!REDUCED) a *= .8 + .2*Math.sin(T*1.57 + i + r*.6 + cc*.4);
          g.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${.15 + a*.8})`;
          g.beginPath(); g.arc(x, y, 1.3 + a*2.3, 0, 7); g.fill();
        }
        /* THE FUSION: the disposed-to-say words glowing inside the glass, per depth */
        if(wl) wl.forEach((wd, k) => {
          const u = .5 + .17*Math.sin(k*2.1 + i*1.3);
          const v = .16 + k*.22;
          const x = q.tl[0] + (q.tr[0]-q.tl[0])*u;
          const yTop = q.tl[1] + (q.tr[1]-q.tl[1])*u, yBot = q.bl[1] + (q.br[1]-q.bl[1])*u;
          const y = yTop + (yBot-yTop)*v + (REDUCED ? 0 : Math.sin(T*.9 + i + k*1.7)*2.5);
          const fs = (8.5 + wd.s01*5) * (isF ? 1.12 : 1);
          g.font = "600 " + fs.toFixed(1) + "px " + '"IBM Plex Mono",monospace';
          g.textAlign = "center";
          const pulse = REDUCED ? 1 : (.88 + .12*Math.sin(T*1.57 + k*.9 + i));   /* the heartbeat */
          g.shadowColor = `rgba(${c[0]},${c[1]},${c[2]},.95)`;
          g.shadowBlur = 6 + wd.s01*8;
          g.fillStyle = `rgba(255,255,255,${(.5 + .45*wd.s01) * pulse})`;
          g.fillText(wd.piece, x, y);
          g.shadowBlur = 0;
          g.fillStyle = `rgba(${Math.min(255,c[0]+30)},${Math.min(255,c[1]+30)},${Math.min(255,c[2]+30)},${(.3 + .4*wd.s01) * pulse})`;
          g.fillText(wd.piece, x, y);
        });
        g.font = "600 9px " + '"IBM Plex Mono",monospace'; g.textAlign = "left";
        g.fillStyle = `rgba(22,50,74,${isF ? .95 : .6})`;
        g.fillText("L" + layerIds[i] + (i === 0 && layerIds[i] === 0 ? " IN" : i === NP-1 ? " OUT" : ""), q.tl[0], q.tl[1] - 8);
      }
      if(!REDUCED) raf = requestAnimationFrame(frame);
    }
    frame(REDUCED ? 0 : performance.now());
    return () => cancelAnimationFrame(raf);
  }, [rot, zoom, focus, meta, rec.id]);

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led lilac"></span><span class="cap">state scope</span>
      <span class="tail">${meta.lens && meta.real ? "norms + dispositions · the lens's fitted depths"
        : meta.lens ? "dispositions live · norms sample"
        : meta.real ? "live · depth × position" : "sample pattern"}</span>
      <span class=${"tag " + (meta.real ? "cap-t" : meta.lens ? "der-t" : "smp-t")}>${
        meta.real && meta.lens ? "LIVE" : meta.real ? "CAPTURED" : meta.lens ? "DERIVED" : "SAMPLE"}</span></div>
    <div class="scope-body">
      <span class="grace">you're an angel!</span>
      <canvas ref=${canvasRef} height="230"></canvas>
    </div>
    <div class="scope-ctl">
      <span>Rotate</span><input type="range" min="0" max="100" value=${rot}
        onInput=${e => setRot(+e.target.value)}/>
      <span>Zoom</span><input type="range" min="60" max="140" value=${zoom}
        onInput=${e => setZoom(+e.target.value)}/>
      <span>Focus</span><select value=${focus} onChange=${e => setFocus(+e.target.value)}>
        ${meta.layerIds.map((L,i) => html`<option value=${i}>L${L}${i === 0 ? " in" : i === meta.layerIds.length-1 ? " out" : ""}</option>`)}
      </select>
      <button class="spd" style="margin-left:auto"
        onClick=${() => toast("LAYER INSPECTOR — the deep pane view opens in a later phase")}>OPEN LAYER INSPECTOR</button>
    </div>
    <div class="scope-note">${meta.lens && meta.real
      ? "dots = residual norms (one causal forward) · glowing words = J-lens dispositions, read at the lens's own fitted depths — a disposition, not a verified thought"
      : meta.lens
      ? "glowing words = live J-lens dispositions per depth; dot pattern is sample (norms unavailable) — a disposition, not a verified thought"
      : meta.real
      ? "residual norms — one causal forward over this run's prompt (no lens fitted for this model — dots only)"
      : "sample pattern — the live scope reads real /engine/layers norms and J-lens dispositions when the engine is up"}</div>
  </div>`;
}

/* ───────────────────────── right column ───────────────────────── */
function AreaChart({ vals, rgb }){
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current; if(!cv) return;
    const W = cv.parentElement.clientWidth, H = 70, dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W*dpr; cv.style.width = W + "px"; cv.height = H*dpr; cv.style.height = H + "px";
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.clearRect(0,0,W,H);
    for(let k = 1; k < 4; k++){ g.strokeStyle = "rgba(90,130,155,.14)";
      g.beginPath(); g.moveTo(0, H*k/4); g.lineTo(W, H*k/4); g.stroke(); }
    if(!vals.length) return;
    const mx = Math.max(1e-6, ...vals);
    const pts = vals.map((v,i) => [i/(vals.length-1 || 1)*W, H - 4 - (v/mx)*(H - 10)]);
    const gr = g.createLinearGradient(0,0,0,H);
    gr.addColorStop(0, `rgba(${rgb},.4)`); gr.addColorStop(1, `rgba(${rgb},.03)`);
    g.beginPath(); g.moveTo(0,H); pts.forEach(p => g.lineTo(p[0],p[1])); g.lineTo(W,H); g.closePath();
    g.fillStyle = gr; g.fill();
    g.beginPath(); pts.forEach((p,i) => i ? g.lineTo(p[0],p[1]) : g.moveTo(p[0],p[1]));
    g.strokeStyle = `rgba(${rgb},.9)`; g.lineWidth = 1.4; g.stroke();
    const e = pts[pts.length-1];
    g.fillStyle = `rgba(${rgb},1)`; g.beginPath(); g.arc(e[0], e[1], 2.6, 0, 7); g.fill();
  }, [vals, rgb]);
  return html`<canvas ref=${ref} height="70"></canvas>`;
}
function Meters({ rec }){
  const steps = normSteps(rec);
  return html`<div class="mod">
    <span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led blue"></span><span class="cap">multi-meter</span>
      <span class="tag cap-t">CAPTURED</span></div>
    <div class="meterbody">
      <div class="mcap"><span><i style="background:var(--blue)"></i>confidence</span></div>
      <${AreaChart} vals=${steps.map(s => s.conf ?? 0)} rgb="44,191,232"/>
      <div class="mcap" style="margin-top:8px"><span><i style="background:var(--lilac2)"></i>entropy</span>
        <span class="tag der-t" style="margin-left:auto">DERIVED</span></div>
      <${AreaChart} vals=${steps.map(s => s.ent ?? 0)} rgb="154,146,200"/>
    </div>
  </div>`;
}

/* influence descriptor -> a human name, resolving card ids through the run's own memory manifest */
function influenceName(inf, rec){
  if(!inf) return "influence";
  if(inf.card_id){
    const mem = rec.memory || {};
    const ids = mem.applied_ids || [], texts = mem.cards_applied || [];
    const i = ids.indexOf(inf.card_id);
    return i >= 0 && texts[i] ? `card · ${String(texts[i]).slice(0, 40)}` : `card ${inf.card_id}`;
  }
  if(inf.dial) return `dial · ${inf.dial}`;
  // sections now fire through the same prove-all as cards/dials (receipts.deltas._section_influences) --
  // without this branch a section falls through to the raw-JSON fallback below.
  if(inf.section) return `section · ${inf.section}`;
  if(inf.memory_off) return "memory (whole block)";
  if(inf.behavior_off) return "behavior (all dials)";
  return JSON.stringify(inf).slice(0, 40);
}

function ReceiptsPanel({ rec }){
  const receipts = useStore(x => x.receipts);
  const busy = useStore(x => x.busy.prove);
  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">receipts</span>
      <span class="tail">${busy ? "proving…" : receipts ? "measured · mode both" : "on demand"}</span></div>
    <div class="receipts-box">
      ${!receipts && !busy && html`<div class="none" style="padding:6px 0 10px">
        Receipts are computed on demand, never pre-attached — press PROVE to run leave-one-out
        (both-arms-greedy) plus forced scoring over this run's memory and dials.</div>`}
      ${busy && html`<div class="none" style="padding:6px 0 10px">regenerating arms greedy — this takes a while, honestly…</div>`}
      ${receipts && (receipts.list.length ? receipts.list.map(r => {
        const f = receipts.forcedBy.get(receipts.keyFn(r.influence));
        const changed = !!r.has_effect;
        /* silent influence — the server's own formula (contracts §6): text unchanged but the
           forced dependence clearly beats its matched null floor */
        const silent = !changed && f && f.null_floor && f.null_floor.exceeds_floor_by_order_of_magnitude;
        return html`<div class="receipt-row">
          <span>${influenceName(r.influence, rec)}
            ${silent && html` <span class="tag" style="color:#B0509E;background:rgba(228,140,216,.14);border:1px solid rgba(228,140,216,.55)">SILENT INFLUENCE</span>`}</span>
          <span class="nats">${f && f.sum_nats != null ? (+f.sum_nats).toFixed(2) + " nats" : "—"}</span>
          <span class="sub">
            <span class=${changed ? "yes" : "no"}>changed the answer · ${changed ? "yes" : "no"}</span>
            <span class=${r.causal_verified ? "yes" : "no"}>causal_verified · ${String(r.causal_verified)}</span>
            ${r.delta && r.delta.changed != null && html`<span>word-shift ${r.delta.changed}%</span>`}
            ${f && f.mean_nats_per_token != null && html`<span>forced Δ ${(+f.mean_nats_per_token).toFixed(3)} nats/tok</span>`}
            ${(r.ablation_note || (f && !f.causal_verified && f.note)) &&
              html`<span>${String(r.ablation_note || f.note).slice(0, 90)}</span>`}
          </span>
        </div>`; })
      : html`<div class="none" style="padding:6px 0 10px">no fired influences to measure on this run
          (no memory or dials rode it).</div>`)}
      ${receipts && receipts.skipped.length ? html`<div class="none" style="padding:4px 0 0">
        skipped: ${receipts.skipped.map(s => (s.reason || "")).join(" · ").slice(0, 140)}</div>` : null}
      ${receipts && receipts.redundant.length ? receipts.redundant.map(p => html`
        <div class="none" style="padding:4px 0 0">redundant pair: ${(p.redundant || []).join(" + ")} — ${p.note || ""}</div>`) : null}
      ${receipts && html`<div class="none" style="padding:6px 0 2px;font-size:8px">
        forced Δ measures dependence — a nonzero delta does NOT mean the answer would have differed.
        Pairwise redundancy guard only, not the full power set.</div>`}
    </div>
  </div>`;
}

function Steer({ rec }){
  const dials = Object.entries((rec.behavior || {}).active_dials || {}).filter(([,v]) => v);
  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">quick steer</span>
      <span class="tail">this run · read-only</span></div>
    ${dials.length ? dials.map(([k,v]) => html`<div class="steer-row">
        <span>${k}</span><span class="v">${(+v).toFixed(1)}</span>
        <span class="bar"><i style=${"width:" + Math.round(Math.min(1, Math.abs(v))*100) + "%"}></i></span>
      </div>`)
      : html`<div class="steer-row"><span class="none">no dials rode this run</span></div>`}
    <div class="sooncard">
      <span><b>Patch interventions</b><span class="ds">any-concept dials · swap receipts</span></span>
      <span class="stag">soon</span>
    </div>
  </div>`;
}

/* ── F3: lineage subway map — the whole branch family as a clickable tree ── */
function LineageTree({ rec }){
  const [fam, setFam] = useState(null);
  const [open, setOpen] = useState(false);
  useEffect(() => { setFam(null); setOpen(false); }, [rec.id]);
  const load = async () => {
    if(fam || rec._sample || !store.get().live) return;
    const r = await api.family(rec.id);
    setFam((r && Array.isArray(r.runs)) ? r.runs : []);
  };
  if(rec._sample) return null;
  const kids = {};
  (fam || []).forEach(r => { if(r.parent_run_id) (kids[r.parent_run_id] = kids[r.parent_run_id] || []).push(r); });
  const byId = Object.fromEntries((fam || []).map(r => [r.id, r]));
  const roots = (fam || []).filter(r => !r.parent_run_id || !byId[r.parent_run_id]);
  const verb = r => (r.changes_applied && Object.keys(r.changes_applied)[0]) || r.source || "";
  const Node = ({ r, depth }) => html`
    <div style=${"padding-left:" + (depth * 18) + "px;display:flex;align-items:center;gap:6px"}>
      <span style="color:var(--mist);font-size:9px">${depth ? "└⑂" : "●"}</span>
      <button class="mono" style=${"font-size:9px;padding:1px 5px;border-radius:4px;cursor:pointer;"
          + (r.id === rec.id ? "color:var(--navy);background:rgba(182,176,218,.25);font-weight:600" : "color:#4A5878")}
        onClick=${() => loadRun(r.id)}>${r.id}</button>
      <span style="font-size:8.5px;color:var(--mist)">${verb(r)} · ${shortTime(r.created_at)}</span>
    </div>
    ${(kids[r.id] || []).map(c => html`<${Node} r=${c} depth=${depth + 1}/>`)}`;
  return html`<details class="leader" onToggle=${e => { setOpen(e.target.open); if(e.target.open) load(); }}>
    <summary>Lineage — every branch/replay/fork of this run, as a tree
      <span style="margin-left:auto">${fam ? fam.length + " run(s)" : "on demand"}</span></summary>
    <div style="padding:8px 12px 10px;display:flex;flex-direction:column;gap:3px">
      ${!fam && open ? html`<span class="none">reading the family…</span>`
        : fam && !fam.length ? html`<span class="none">this run has no recorded relatives</span>`
        : fam ? roots.map(r => html`<${Node} r=${r} depth=${0}/>`) : null}
    </div>
  </details>`;
}

/* ── F4: span forensics — ablate a phrase from the prompt, attribute the change causally ── */
function SpanForensics({ rec }){
  const [phrase, setPhrase] = useState("");
  const [out, setOut] = useState(null);
  const busy = useStore(x => x.busy.spanReceipt);
  useEffect(() => { setOut(null); setPhrase(""); }, [rec.id]);
  const run = async () => {
    if(guardSample(rec)) return;
    const p = phrase.trim(); if(!p) return;
    if(store.get().busy.spanReceipt) return;
    setBusy("spanReceipt", true); setOut(null);
    toast("span forensics — regenerating without that span + forced-scoring the original…");
    const r = await api.spanReceipt(rec.id, p);
    setBusy("spanReceipt", false);
    if(!r){ setOut({ error: "no answer — is the engine up?" }); return; }
    if(r.__status && r.__status >= 400){ setOut({ error: r.error || ("failed (" + r.__status + ")") }); return; }
    setOut(r);
  };
  const forced = (out && (out.forced || {})) || {};
  const changed = out && (out.answer_changed ?? out.has_effect
    ?? (out.regen && out.regen.has_effect)
    ?? (out.baseline_reply != null && out.ablated_reply != null
        && out.baseline_reply !== out.ablated_reply));
  return html`<div class="mod">
    <div class="mod-h"><span class="led lilac"></span><span class="cap">span forensics</span>
      <span class="tail">${busy ? "ablating…" : "which words in the context did this?"}</span></div>
    <div style="padding:2px 13px 10px;display:flex;flex-direction:column;gap:6px">
      <div style="display:flex;gap:6px">
        <input type="text" placeholder="a phrase from the prompt/context to ablate"
          value=${phrase} disabled=${busy}
          style="flex:1;font-size:10px;padding:4px 7px;border:1px solid rgba(90,130,155,.4);border-radius:5px;background:rgba(255,255,255,.6)"
          onInput=${e => setPhrase(e.target.value)}
          onKeyDown=${e => { if(e.key === "Enter") run(); }}/>
        <button class="spd" disabled=${busy} onClick=${run}>${busy ? "…" : "ABLATE"}</button>
      </div>
      ${out && out.error && html`<span class="none">${out.error}</span>`}
      ${out && !out.error && html`
        <div class="receipt-row">
          <span>“${((out.influence || {}).text || phrase).slice(0, 44)}”
            <span class=${changed ? "tag warn-t" : "tag cap-t"} style=${changed
              ? "color:#C24A31;background:rgba(242,109,79,.10);border:1px solid rgba(242,109,79,.45)" : ""}>
              ${changed ? "CHANGED THE ANSWER" : "no change"}</span></span>
          <span class="nats">${forced.sum_nats != null ? (+forced.sum_nats).toFixed(2) + " nats" : ""}</span>
          <span class="sub">
            ${out.baseline_reply != null && html`<span>with: “${String(out.baseline_reply).slice(0, 60)}”</span>`}
            ${out.ablated_reply != null && html`<span>without: “${String(out.ablated_reply).slice(0, 60)}”</span>`}
            ${forced.mean_nats_per_token != null && html`<span>forced Δ ${(+forced.mean_nats_per_token).toFixed(3)} nats/tok</span>`}
          </span>
        </div>
        <span class="none" style="font-size:8px">ablation-causal: the span was removed and the run
          re-derived — measured, not guessed. Agent transcripts: paste the suspect retrieved sentence.</span>`}
    </div>
  </div>`;
}

/* ── F5: the introspection check — the model's own "why" vs the causal receipts ── */
function LieDetector({ rec }){
  const [out, setOut] = useState(null);
  const busy = useStore(x => x.busy.narrate);
  useEffect(() => setOut(null), [rec.id]);
  const run = async () => {
    if(guardSample(rec)) return;
    if(store.get().busy.narrate) return;
    setBusy("narrate", true); setOut(null);
    toast("asking the model why — then checking its story against the receipts…");
    const r = await api.narrate(rec.id);
    setBusy("narrate", false);
    if(!r){ setOut({ error: "no answer from the server" }); return; }
    if(r.__status && r.__status >= 400){
      setOut({ error: r.__status === 503
        ? "narration needs the qwen (HF) substrate — the pure-engine substrate can't run the constrained self-report. Switch substrates to use the lie detector."
        : (r.error || ("failed (" + r.__status + ")")) });
      return;
    }
    setOut(r);
  };
  const claims = (out && (out.claims || out.narration_claims || [])) || [];
  const nar = out && (out.narration || out.text || out.narrative || null);
  return html`<div class="mod">
    <div class="mod-h"><span class="led" style="background:var(--pink);box-shadow:0 0 8px var(--pink)"></span>
      <span class="cap">self-report check</span>
      <span class="tail">${busy ? "narrating…" : "measure, don't ask — then compare"}</span></div>
    <div style="padding:2px 13px 10px;display:flex;flex-direction:column;gap:6px">
      ${!out && !busy && html`
        <span class="none">Ask the model to explain its own answer, then score that story against the
          causal receipts. X1 measured: content is legible, process is not — expect confabulation, and
          see it caught.</span>
        <button class="spd" style="align-self:start" onClick=${run}>ASK WHY — THEN CHECK IT</button>`}
      ${busy && html`<span class="none">generating the constrained narration…</span>`}
      ${out && out.error && html`<span class="none">${out.error}</span>`}
      ${out && !out.error && html`
        ${nar && html`<div style="font-size:9.5px;color:#2A3252;border-left:2px solid var(--lilac);padding-left:8px">
          ${String(nar).slice(0, 400)}</div>`}
        ${claims.length ? claims.map(c => html`<div class="receipt-row">
          <span>${String(c.claim || c.text || "").slice(0, 52)}</span>
          <span class="sub">
            <span class=${c.supported ?? c.receipt_supported ? "yes" : "no"}>
              receipt-supported · ${String(c.supported ?? c.receipt_supported ?? "?")}</span>
            ${(c.verdict || c.flag) && html`<span>${String(c.verdict || c.flag)}</span>`}
          </span>
        </div>`) : null}
        ${out.note && html`<span class="none" style="font-size:8px">${String(out.note).slice(0, 220)}</span>`}`}
    </div>
  </div>`;
}

function Minfl({ rec }){
  const mem = rec.memory || {};
  const cards = mem.cards_applied || [];
  const anchored = mem.anchored || [];
  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">memory influence</span>
      <span class="tail">top contributors</span></div>
    <div style="padding-bottom:4px">
      ${cards.length ? cards.map((c,i) => {
        const rel = (mem.relevance || [])[i];
        return html`<div class="steer-row">
          <span style="font-size:9.5px">${String(c).slice(0, 42)}</span>
          <span class="v">${rel != null ? (+rel).toFixed(2) : "—"}</span>
          <span class="bar"><i style=${"width:" + (rel != null ? Math.round(rel*100) : 0) + "%"}></i></span>
        </div>`; })
      : !anchored.length ? html`<div class="steer-row"><span class="none">no memory rode this run</span></div>` : null}
      ${anchored.length ? html`<div class="none" style="padding:8px 14px 2px;text-transform:uppercase;letter-spacing:.1em">
        anchored bags (${anchored.length})</div>` : null}
      ${anchored.map(b => html`<div class="steer-row">
        <span style="font-size:9.5px">${String(b.card_id || "anchored").slice(0, 32)}</span>
        <span class="v">${b.gate != null ? (+b.gate).toFixed(2) : "—"}</span>
        <span class="sub" style="grid-column:1/-1;gap:5px;flex-wrap:wrap">
          ${(b.alpha_top3 || []).map(t => html`<span class="jchip d1" style="font-size:9px">
            ${String(t.token || "").slice(0, 18)} ${t.alpha != null ? ((+t.alpha) >= 0 ? "+" : "") + (+t.alpha).toFixed(2) : ""}</span>`)}
        </span>
      </div>`)}
      ${mem.gate != null && html`<div class="steer-row">
        <span style="font-size:9.5px">topic gate</span>
        <span class="v">${(+mem.gate).toFixed(2)}</span>
        <span class="bar"><i style=${"width:" + Math.round(mem.gate*100) + "%"}></i></span>
      </div>`}
    </div>
  </div>`;
}

/* ── prompt-section influence (foundation): the run's own prompt-section manifest -- system prompt /
   RAG context / memory-card blocks -- from clozn.runs.sections (rec.sections, a fixed contract with the
   server). Each row gets a FAST, approximate influence reading from the new /section-influence route
   (teacher-forced log-prob delta -- NOT causal; the fixed contract this module codes against, built in
   parallel on the server, so its implementation is deliberately not read here) plus a real causal receipt
   via the SAME single-influence /receipt call SpanForensics above already uses (a section is just one
   more `_ablation_changes` arm server-side -- "section receipts are just more arms in the same
   prove-all", this app's per-item receipt call included). Renders nothing when a run carries no
   `sections` manifest (old runs) or an empty one -- no error, no empty card. */

/* api.mjs (untouched by this feature) has no wrapper for the new route yet -- this is a deliberate,
   minimal local mirror of that file's own postE (same contract: __status rides even on a non-2xx
   response; null only on a genuine network failure), scoped to the one path this module needs. */
async function postSectionInfluence(id){
  const c = new AbortController();
  const k = setTimeout(() => c.abort(), 30000);
  try{
    const r = await fetch("/runs/" + encodeURIComponent(id) + "/section-influence", {
      method: "POST", signal: c.signal,
      headers: { "Content-Type": "application/json" }, body: "{}" });
    clearTimeout(k);
    let o = null; try{ o = await r.json(); }catch(e){ o = {}; }
    return Object.assign({ __status: r.status }, o);
  }catch(e){ clearTimeout(k); return null; }
}

/* source -> one of the existing .tag color variants, borrowed for their COLOR only (own text below, not
   their usual CAPTURED/DERIVED/sample labels) -- so a section's provenance reads consistently with the
   rest of the instrument without a new theme.css rule. */
function sectionTagClass(source){
  return source === "auto" ? "der-t" : source === "memory_card" ? "smp-t" : "cap-t";
}

/* one section: collapsible (details.leader, the same pattern LineageTree above uses) -- name/source/share%
   always visible in the summary, delta/preview/receipt only on expand. `fast` is undefined (still
   reading), null (no fast reading for this section), or the route's own per-section entry. */
function SectionRow({ rec, sec, fast }){
  const [out, setOut] = useState(null);      // this row's own causal receipt, once proven
  const [busy, setBusy] = useState(false);
  const share = fast && fast.influence_share != null ? +fast.influence_share : null;
  const pct = share != null ? Math.round(Math.max(0, Math.min(1, share)) * 100) : null;
  const prove = async () => {
    if(guardSample(rec) || busy) return;
    setBusy(true);
    toast(`proving “${sec.name || sec.id}” — regenerating greedy, with vs without this section…`);
    /* contracts-adjacent: the single-influence /receipt route (api.receipt) already accepts
       {influence:{section, ...}} server-side (receipts.deltas._ablation_changes) -- a real per-section
       causal receipt, not just this app's blunt global PROVE. */
    const r = await api.receipt(rec.id, { influence: { section: sec.name }, mode: "regen" });
    setBusy(false);
    if(!r || r.error){
      setOut({ error: (r && r.error) || "receipt failed — needs a chat-capable substrate; is the engine up?" });
      return;
    }
    setOut(r);
  };
  return html`<details class="leader">
    <summary>
      <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${String(sec.name || sec.id || "section")}</span>
      <span class="tag ${sectionTagClass(sec.source)}">${String(sec.source || "?")}</span>
      <span style="font-size:9px;color:#1B87A8;font-weight:600;white-space:nowrap">${fast === undefined ? "…" : pct != null ? pct + "%" : "—"}</span>
    </summary>
    <div class="leader-body">
      ${sec.char_count != null && html`<div>${sec.char_count} chars</div>`}
      ${sec.preview && html`<div style="font-style:italic">“${String(sec.preview).slice(0, 120)}”</div>`}
      ${fast === undefined && html`<div class="none">reading fast influence…</div>`}
      ${fast === null && html`<div class="none">no fast influence reading for this run.</div>`}
      ${fast && fast.log_prob_delta != null && html`<div class="none">Δ logprob ${(+fast.log_prob_delta).toFixed(2)}${fast.per_token_delta != null ? ` (${(+fast.per_token_delta).toFixed(3)}/token)` : ""}</div>`}
      ${fast && fast.summary && html`<div class="none">${String(fast.summary).slice(0, 160)}</div>`}
      ${!out && html`<button class="spd" style="margin-top:6px" disabled=${busy}
        onClick=${prove}>${busy ? "PROVING…" : "PROVE THIS SECTION"}</button>`}
      ${out && out.error && html`<div class="none" style="margin-top:6px">${String(out.error)}</div>`}
      ${out && !out.error && html`<div class="receipt-row" style="border-top:none;padding-top:6px">
        <span>${out.has_effect ? "changed the answer" : "no change"}
          ${out.ablated_reply_truncated && html` <span class="tag der-t">early-stopped prefix</span>`}</span>
        <span class="sub">
          <span class=${out.causal_verified ? "yes" : "no"}>causal_verified · ${String(out.causal_verified)}</span>
          ${out.delta && out.delta.changed != null && html`<span>word-shift ${out.delta.changed}%</span>`}
          ${(out.ablation_note || out.early_stop_note) && html`<span>${String(out.ablation_note || out.early_stop_note).slice(0, 120)}</span>`}
        </span>
      </div>`}
    </div>
  </details>`;
}

function SectionsInfl({ rec }){
  const [infl, setInfl] = useState();   // undefined = not asked yet, null = asked but failed/empty
  useEffect(() => {
    setInfl();
    if(rec._sample || !Array.isArray(rec.sections) || !rec.sections.length) return;
    let dead = false;
    (async () => {
      const r = await postSectionInfluence(rec.id);
      if(dead) return;
      setInfl((r && r.__status < 300 && Array.isArray(r.sections) && r.sections.length) ? r : null);
    })();
    return () => { dead = true; };
  }, [rec.id]);

  const sections = Array.isArray(rec.sections) ? rec.sections : [];
  if(!sections.length) return null;   // no manifest on this run (old run, or a source that never built one)
  const byId = {};
  ((infl && infl.sections) || []).forEach(s => { if(s && s.id != null) byId[s.id] = s; });

  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">prompt sections</span>
      <span class="tail">${infl === undefined ? "reading…" : sections.length + " section(s)"}</span></div>
    ${infl && infl.any_meaningful === false && html`<div class="none" style="padding:3px 14px 4px;font-size:9px;opacity:.9">No section measurably influenced this reply — the shares below are within noise, not a ranking (it likely came from the model's own knowledge).</div>`}
    ${infl && infl.note && html`<div class="none" style="padding:0 14px 6px;font-size:8px">${String(infl.note).slice(0, 200)}</div>`}
    <div style="padding-bottom:4px">
      ${sections.map((sec, i) => html`<${SectionRow} key=${sec.id || i} rec=${rec} sec=${sec}
        fast=${infl === undefined ? undefined : (byId[sec.id] || null)}/>`)}
    </div>
    <div class="none" style="padding:2px 14px 10px;font-size:8px">Share/Δ above are approximate
      (teacher-forced log-probability deltas) — NOT causal proof; press a row's own PROVE for the real,
      measured ablation receipt.</div>
  </div>`;
}
