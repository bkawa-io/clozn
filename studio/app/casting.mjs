/* clozn studio -- THE CASTING: the Observatory hero visualization, as a reusable component.
   Spec: notes/UX_INFORMATION_ARCHITECTURE.md SS7.6 "THE CASTING -- storm-grammar spec" (LOCKED) --
   referenced by path only, never quoted into this file. Faithful port of the BK-approved prototype
   (scratchpad casting-prototype.html), reorganized into a mountable component + documented data
   contract; the tuned geometry/motion constants (camera, particle field, jag/turbulence formulas,
   phase timings) are carried over UNCHANGED -- this file does not redesign the approved look.

   Core idea (BK's tesseract/flatland insight): we don't visualize the model -- we visualize the
   shadow it casts around one question. SKY = the prompt. CLOUD = the interior (entropy silhouette,
   turbulence, candidate cells, lead-change arcs, a commit stratum). GROUND = the answer, struck one
   token at a time. Every visual element below either reads a field from the data contract or is
   explicitly a declared decoration (see "HONESTY" below) -- no unearned glow.

   Colors are NEVER hardcoded: every fill/stroke reads studio/app/tokens.css custom properties via
   getComputedStyle(document.documentElement), so the scene inherits dawn/night automatically from
   whatever the host page has set (prefers-color-scheme or an explicit [data-theme]). setNight() is
   kept for API parity with light.mjs's mountLight and to force an immediate cache refresh right after
   a host-driven theme flip; a MutationObserver + matchMedia listener also keep colors live without it.

   ============================================================================ DATA CONTRACT
   mountCasting(container, opts).update(data) accepts one "cast" -- everything THE CASTING needs to
   render one question's storm. Shape (all fields optional unless noted; see DEMO_CASTING for a
   worked example of every field):

   {
     meta: {
       label:          string   -- short human label, e.g. "tallest mountain" (cast-picker UI)
       note:           string   -- plain-text one-line description shown in-scene (NOT html; escaped
                                    as text). Optional.
       scripted:       boolean  -- true iff this cast's numbers are demo/scripted, not measured.
                                    Purely a documentation flag for the HOST to disclose; the
                                    component itself never fabricates a number regardless of this flag.
       commitLayer:    number   -- REQUIRED for the commit ring: index into entropyByDepth (and
                                    readoutLayers, if given) where rank-1 stabilizes.
       readoutLayers:  number[] -- optional; the true transformer-layer index each entropyByDepth
                                    entry corresponds to (readouts are often a stratified subsample,
                                    not literally one per layer -- e.g. [0,6,12,18,24,28,31]). Without
                                    this, layer labels degrade honestly to "stratum i/N" instead of
                                    inventing a layer number.
       totalLayers:    number   -- optional; the model's total layer count, for "L18 of 32" labels.
     },
     entropyByDepth:  number[]  -- REQUIRED. Per-readout-layer entropy in bits, shallow -> deep. One
                                    array entry = one J-lens readout stratum. Drives: cloud silhouette
                                    width per stratum, condensation morph, bolt vertex COUNT (one kink
                                    per array entry, per the grammar's "bolt vertices -- ONE PER
                                    LAYER"), and the HUD entropy readout.
     effectiveRankByDepth: number[] -- optional, same length/ordering as entropyByDepth. Real
                                    effective-rank trajectory (a1_5-style). When absent the HUD rank
                                    readout honestly shows "--" rather than inventing a curve.
     turbulence:      number    -- how near-tied the distribution is (peakedness); >=0, prototype
                                    range ~0.3-2.1. Drives cloud roil amplitude.
     width:           number    -- overall storm-cloud size scalar, ~0.6-1.4.
     crystalline:     boolean   -- true for a tight, closed storm (e.g. an equation with no
                                    contested candidates) -- tighter, brighter particle rendering.
     cells: [{ x, z, share, label? }]  -- top-k candidate cells with probability mass ("share", 0..1,
                                    need not sum to 1). x/z are layout coordinates in the cloud's local
                                    space (roughly -1..1); label e.g. "Everest \xb7 66%".
     leadFlips: [[layerIdx, fromCellIdx, toCellIdx], ...] -- per-layer argmax flips: the intra-cloud
                                    arcs. layerIdx indexes entropyByDepth/readoutLayers. Empty = a
                                    silent storm, never contested.
     sky: [{ text, weight }]    -- the prompt/context words, in order. weight (0..1) is per-word draw
                                    strength; used for the "GATHERING inhales ALL once, then only the
                                    used" intake-thread behavior.
     tokens: [{
       text:      string        -- the emitted token/word, planted into the ground on its strike.
       entropy:   number        -- bits at emission. Drives bolt kink size ("that layer's uncertainty"
                                    read at commit), strike violence/spark count, and the "shaky"
                                    flicker (entropy above a visual threshold reads as unsteady).
       sources:   [[skyIdx, weight], ...] -- provenance: which sky words fed this token and how hard
                                    (weight 0..1). Empty = parametric ("from the weights -- not from
                                    your words").
       almosts: [{
         text:          string   -- the alternate token that almost won.
         pull:          number   -- its probability mass (0..1). Drives ghost brightness + bolt lean.
         continuation:  [{text, entropy}, ...]  -- OPTIONAL. If the host already knows how the
                                    alternate branch would have continued (e.g. scripted demo data, or
                                    a prefetched branch), the component self-animates the fork replay
                                    from this. If omitted, the ghost is still forkable (see below) but
                                    the component waits for the host to call update() with the real
                                    branched data after bit-exact checkpoint/branch machinery runs.
         forkable:      boolean  -- OPTIONAL, defaults to true iff continuation is given. Set this
                                    explicitly to mark a ghost forkable even when continuation is not
                                    yet known (branch machinery exists but hasn't been fetched).
       }, ...]
     }, ...]
   }

   update(data) diffs against whatever is currently mounted: if `data` is a byte-for-byte TEXT
   continuation of the currently-showing run (same sky, tokens[0..k) unchanged, more tokens appended)
   it is treated as the SAME storm still deciding -- new strikes land without replaying GATHERING
   (best-effort live-streaming affordance for /runs/watch-style incremental calls; falls back to a
   full replay whenever that can't be established, which is always visually correct even if not
   maximally smooth). Otherwise it is a NEW cast: the current storm RELEASES, then the new one
   GATHERS. update(null) explicitly releases back to AT REST.

   opts.onFork(tokenIndex, altText) fires whenever a forkable ghost is clicked. THE COMPONENT DOES NOT
   DECIDE WHAT HAPPENS NEXT -- the host owns the real bit-exact checkpoint/branch machinery and is
   expected to eventually call update() again with the branched run's data.

   ============================================================================ HONESTY
   Every draw call below is commented back to the data field (or explicitly marked decoration):
   droplets are NOT implemented in this build (the spec marks them "MEDIUM today (declared)" pending
   the live SAE/BrainReadout feed -- there is no honest scripted stand-in for a per-feature signal, so
   this build omits them rather than fake one; see the report for how to wire them once SAE-with-null
   lands). The demo/scripted flag on DEMO_CASTING is carried through to the demo page's footer, never
   silently dropped. `prefers-reduced-motion` renders one static, fully-resolved (INSPECT) frame --
   motion is earned by an actual cast or interaction, never mood.

   LAYERING (spec SS9.2, shader commitment): a WebGL2 canvas sits BENEATH this one and carries
   atmosphere ONLY -- the cloud as a volume (fbm density field shaped by entropyByDepth/turbulence/
   width/crystalline), bolt glow along a real strike, and ground embers that breathe with a planted
   word's own entropy. Text, precise bolt-vertex geometry, hit-testing, and every existing particle/
   cell/HUD element ABOVE stay exactly this Canvas2D code, unchanged. See createAtmosphere() below for
   the full uniform-to-data mapping and the graceful-absence contract (no WebGL2 -> this file's
   Canvas2D path renders byte-for-byte as it always has). */

/* ============================================================================ tokens.css bridge */

const TOKEN_NAMES = [
  "ink", "ink-soft", "ink-faint", "ground-0", "ground-1", "line-strong",
  "evidence", "almost", "fork", "shaky", "confidence",
  "iri-pink", "iri-gold", "iri-peri", "iri-mint", "iri-lilac",
];
// Fallback palette (tokens.css's night block) so the component never renders black-on-black if used
// before tokens.css has loaded, or in isolation without it.
const FALLBACK = {
  ink: "#E8E9F8", "ink-soft": "#A7ACD6", "ink-faint": "#7A80AC",
  "ground-0": "#0B0D1E", "ground-1": "#12142E", "line-strong": "#3B4070",
  evidence: "#5FC8B4", almost: "#E896D6", fork: "#B6B0FA", shaky: "#E8926C", confidence: "#8FB0F0",
  "iri-pink": "#E896D6", "iri-gold": "#F0D096", "iri-peri": "#96AFF2", "iri-mint": "#6ECDBE", "iri-lilac": "#BAB2E2",
};
const FALLBACK_FONT = 'ui-monospace,"Cascadia Mono","SFMono-Regular",Consolas,monospace';

function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || "").trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function readTokens() {
  const cs = getComputedStyle(document.documentElement);
  const out = {};
  for (const name of TOKEN_NAMES) {
    const raw = cs.getPropertyValue("--" + name).trim();
    out[name] = hexToRgb(raw) || hexToRgb(FALLBACK[name]);
  }
  const font = cs.getPropertyValue("--voice-machine").trim();
  out.font = font || FALLBACK_FONT;
  out.IRI = [out["iri-pink"], out["iri-gold"], out["iri-peri"], out["iri-mint"], out["iri-lilac"]];
  return out;
}

function rgbaC(rgb, a) { return `rgba(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0},${a})`; }

function iriAt(IRI, t) {
  t = clamp(t, 0, 1) * (IRI.length - 1);
  const i = t | 0, f = t - i, a = IRI[i], b = IRI[Math.min(i + 1, IRI.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/* ============================================================================ small math */

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function ease(u) { return u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; }
const SHAKY_ENTROPY_BITS = 1.8; // visual threshold, not a data field: above this a planted word flickers

/* ============================================================================ scene geometry (unchanged from the approved prototype) */

const Y_SKY = -1.04, Y_TOP = -.58, Y_BASE = .40, Y_GROUND = .90;
const N_PARTICLES = 7000;

function rowSize(n) { return Math.max(5, Math.min(9, Math.round(Math.sqrt(Math.max(1, n)) * 2.2))); }

function layoutSky(sky) {
  const perRow = rowSize(sky.length);
  return sky.map((w, i) => {
    const row = (i / perRow) | 0, j = i % perRow;
    const n = Math.min(sky.length - row * perRow, perRow);
    const u = n > 1 ? j / (n - 1) : .5;
    return {
      text: String(w.text != null ? w.text : ""),
      weight: Number.isFinite(w.weight) ? w.weight : 0,
      x: -1.2 + u * 2.4,
      z: -.15 + .25 * Math.sin(u * Math.PI),
      y: Y_SKY + row * .14 + .04 * Math.sin(u * 7),
    };
  });
}

function posFor(i, total, perRow) {
  const row = (i / perRow) | 0, j = i % perRow;
  const n = Math.min(total - row * perRow, perRow);
  const u = n > 1 ? j / (n - 1) : .5;
  return { gx: -1.05 + u * 2.1, gz: .30 - row * .32 };
}

function normalizeTokens(tokens) {
  return tokens.map(tk => ({
    text: String(tk.text != null ? tk.text : ""),
    entropy: Number.isFinite(tk.entropy) ? tk.entropy : 0,
    sources: Array.isArray(tk.sources) ? tk.sources : [],
    almosts: (Array.isArray(tk.almosts) ? tk.almosts : []).map(a => ({
      text: String(a.text != null ? a.text : ""),
      pull: Number.isFinite(a.pull) ? a.pull : 0,
      continuation: Array.isArray(a.continuation) ? a.continuation : null,
      forkable: a.forkable != null ? !!a.forkable : !!(Array.isArray(a.continuation) && a.continuation.length),
    })),
  }));
}

function derivePreset(data) {
  const entropyByDepth = (Array.isArray(data.entropyByDepth) && data.entropyByDepth.length) ? data.entropyByDepth : [1];
  const nL = entropyByDepth.length;
  const meta = data.meta || {};
  const commitLayerIdx = clamp(Number.isFinite(meta.commitLayer) ? meta.commitLayer : nL - 1, 0, nL - 1);
  const commitY = Y_TOP + (commitLayerIdx / Math.max(1, nL - 1)) * (Y_BASE - Y_TOP);
  const cells = (Array.isArray(data.cells) && data.cells.length) ? data.cells : [{ x: 0, z: 0, share: 1 }];
  let acc = 0;
  const cellCum = cells.map(c => acc += (Number(c.share) || 0));
  const rankByDepth = (Array.isArray(data.effectiveRankByDepth) && data.effectiveRankByDepth.length === nL)
    ? data.effectiveRankByDepth : null;
  return {
    meta,
    entropyByDepth, nL, commitLayerIdx, commitY,
    maxEnt: Math.max(...entropyByDepth, 0.001),
    turbulence: Number(data.turbulence) || 0,
    width: Number.isFinite(data.width) ? data.width : 1,
    crystalline: !!data.crystalline,
    cells, cellCum,
    leadFlips: Array.isArray(data.leadFlips) ? data.leadFlips : [],
    sky: layoutSky(Array.isArray(data.sky) ? data.sky : []),
    tokens: normalizeTokens(Array.isArray(data.tokens) ? data.tokens : []),
    rankByDepth,
    readoutLayers: Array.isArray(meta.readoutLayers) ? meta.readoutLayers : null,
    totalLayers: Number.isFinite(meta.totalLayers) ? meta.totalLayers : null,
  };
}

function entAt(y, preset) {
  const e = preset.entropyByDepth, nL = e.length;
  const u = clamp((y - Y_TOP) / (Y_BASE - Y_TOP), 0, 1) * (nL - 1);
  const i = u | 0, f = u - i;
  return e[i] + (e[Math.min(i + 1, nL - 1)] - e[i]) * f;
}
function cloudW(y, preset) {
  const env = Math.sin(clamp((y - Y_TOP + .10) / (Y_BASE - Y_TOP + .24), 0, 1) * Math.PI);
  return (0.34 + 0.85 * (entAt(y, preset) / preset.maxEnt)) * preset.width * Math.max(env, .12);
}
function cellSep(y, preset) { return clamp((preset.commitY - y) / .5, 0, 1); }
function layerY(i, preset) { return Y_TOP + (i + .5) / preset.nL * (Y_BASE - Y_TOP); }

function cellOf(p, preset) {
  for (let i = 0; i < preset.cellCum.length; i++) if (p.lr <= preset.cellCum[i]) return i;
  return preset.cells.length - 1;
}

function groundBolt(violence, preset, cellIdx, tx, tz, almosts) {
  const cell = preset.cells[cellIdx] || preset.cells[0];
  const nL = preset.nL, segs = [], ghosts = [];
  const lean = (almosts && almosts.length) ? (almosts[0].pull * (Math.random() < .5 ? 1 : -1)) : 0;
  let x = cell.x * cellSep(Y_TOP, preset) * 1.1 + (Math.random() - .5) * .15;
  let z = cell.z * cellSep(Y_TOP, preset) + (Math.random() - .5) * .15;
  let snapIdx = -1, lastFree = 0, dir = Math.random() < .5 ? 1 : -1;
  segs.push([x, Y_TOP - .02, z]);
  for (let i = 0; i < nL; i++) {
    const y = Y_TOP + (i + .7) / nL * (Y_BASE - Y_TOP);
    const committed = y > preset.commitY;
    if (!committed) {
      // bolt vertices -- ONE PER LAYER: kink size is that layer's uncertainty (entropyByDepth[i]).
      const jag = (preset.entropyByDepth[i] / preset.maxEnt) * .34 * (.5 + violence * .5);
      dir = -dir; x += dir * jag + lean * jag * .8; z += (Math.random() - .5) * jag * .5;
      lastFree = segs.length;
    } else {
      if (snapIdx < 0) snapIdx = segs.length;
      x *= .45; z *= .45; // dead straight below commit
    }
    segs.push([x, y, z]);
  }
  segs.push([tx * .35, Y_BASE + .10, tz * .2]);
  segs.push([tx, Y_GROUND, tz]);
  (almosts || []).forEach((a, ai) => {
    const v = segs[lastFree], side = (ai % 2 ? -1 : 1) * (0.25 + a.pull * 0.9);
    const g = [[v[0], v[1], v[2]]];
    let gx = v[0], gy = v[1], gz = v[2];
    for (let j = 0; j < 3; j++) {
      gx += side * .14 + (Math.random() - .5) * .05; gy += .10; gz += (Math.random() - .5) * .06;
      g.push([gx, gy, gz]);
    }
    ghosts.push({
      segs: g, alt: a.text, continuation: a.continuation, forkable: a.forkable,
      label: a.text + " \xb7" + Math.round(a.pull * 100) + "%" + (a.forkable ? "  ⑂" : ""),
      pull: a.pull,
    });
  });
  return { segs, ghosts, life: 1, violence, snapIdx };
}

function flipArc(preset, flip) {
  const a = preset.cells[flip[1]], b = preset.cells[flip[2]];
  if (!a || !b) return null;
  const y = layerY(flip[0], preset), s = cellSep(y, preset) || .15, segs = [], n = 12;
  for (let i = 0; i <= n; i++) {
    const u = i / n;
    segs.push([
      (a.x * s) * (1 - u) + (b.x * s) * u + (Math.random() - .5) * .07,
      y + (Math.random() - .5) * .04,
      (a.z * s) * (1 - u) + (b.z * s) * u + (Math.random() - .5) * .07,
    ]);
  }
  return { segs, life: 1, layer: flip[0] };
}

function isContinuation(oldD, newD) {
  if (!oldD || !newD) return false;
  if (!Array.isArray(oldD.tokens) || !Array.isArray(newD.tokens)) return false;
  if (newD.tokens.length < oldD.tokens.length) return false;
  if (!sameTexts(oldD.sky, newD.sky)) return false;
  for (let i = 0; i < oldD.tokens.length; i++) {
    const a = oldD.tokens[i], b = newD.tokens[i];
    if ((a && a.text) !== (b && b.text)) return false;
  }
  return true;
}
function sameTexts(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if ((a[i] && a[i].text) !== (b[i] && b[i].text)) return false;
  return true;
}

/* ============================================================================ atmosphere (WebGL2 enhancement layer, spec SS9.2)
   A second canvas, painted BENEATH the Canvas2D layer above it (text, precise bolt geometry, and hit-
   testing STAY Canvas2D -- legibility constitution: no texture over text). Raw WebGL2, hand-written
   GLSL, zero deps, matching studio/app/light.mjs's idiom (fullscreen triangle via gl_VertexID, dual
   parity via a uNight uniform, activity-gated motion via uAct -- "time flows with activity", decaying
   so motion must be re-earned). This module does NOT import or modify light.mjs; the fullscreen-
   triangle trick and the dual-parity/activity pattern are the house convention, reproduced
   independently here for the casting's own uniform surface.

   GRACEFUL ABSENCE: if WebGL2 context creation (or shader compile/link) fails, createAtmosphere()
   returns null, the extra <canvas class="cst-gl"> is removed from the DOM, and mountCasting's
   Canvas2D path renders EXACTLY as it did before this layer existed -- same opaque gradient fill,
   same particles/cells/bolts/text. This layer only ever ADDS a backdrop; it never gates legibility.

   UNIFORM SURFACE -- every one maps to a real quantity already in the data contract or the phase
   machine; nothing here is unearned decoration:
     uT                     entropy-fall / effective-rank collapse (0 at REST -> 1 once committed) --
                            the cloud's presence/opacity (data: entropyByDepth via the same t the
                            Canvas2D layer already animates)
     uTurb / uWidth / uCryst  data.turbulence / data.width / data.crystalline, unchanged
     uSheet                 the "sheet-light" event flash (decision/flip lighting the cloud from
                            within) -- the same `sheet` value the Canvas2D layer already tracks
     uCommitU               meta.commitLayer's position along the envelope (0..1)
     uEnv[12]               cloudW(y)/entropyByDepth sampled as screen-projected half-widths,
                            shallow -> deep -- i.e. the SAME silhouette function the Canvas2D particle
                            field already uses, projected through the SAME camera (project())
     uCloudC / uAxis / uCloudH  the real camera projection of the cloud's vertical axis; yaw/pitch
                            orbit (drag-to-rotate) moves this exactly like the Canvas2D particle field
     uBoltPts / uBoltLife   active ground-bolt polylines (per-layer kink vertices), projected each
                            frame from the SAME bolt segments the Canvas2D layer strokes -- a real
                            strike, not a decorative flare
     uEmberPts / uEmberAmt / uEmberHue  planted words' ground positions; uEmberAmt is the identical
                            entropy-driven breathing formula the Canvas2D word-glow already uses
     uAct                   phase/event-driven activity (MOTION = MEANING): idle is a near-still
                            breath, a cast in progress is alive; gates drift/intensity, never the
                            base cloud shape (that's uT, a measured quantity, not an activity level)
     uNight                 dual parity: DAWN PAINTS TOWARD the hue (mix), NIGHT ADDS light -- the
                            hard-won lesson that additive glow on a light ground clips to white and
                            colors vanish
   Colors are the SAME rgb values casting.mjs already reads from tokens.css via readTokens(); there is
   no separate palette for this layer. */

const ENVN = 12, BMAX = 3, PMAX = 16, EMAX = 24;

function luma(rgb) { return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]; }
function deriveNight(colors) { return luma(colors["ground-0"]) < 128 ? 1 : 0; }

const ATMO_VS = `#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0,1);}`;

const ATMO_FS = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uTime,uNight,uAct,uT,uTurb,uWidth,uCryst,uSheet,uCommitU;
uniform vec2 uCloudC,uAxis;
uniform float uCloudH;
uniform float uEnv[${ENVN}];
uniform vec2 uBoltPts[${BMAX * PMAX}];
uniform float uBoltLife[${BMAX}];
uniform vec2 uEmberPts[${EMAX}];
uniform float uEmberAmt[${EMAX}],uEmberHue[${EMAX}];
uniform vec3 uG0,uG1,uFork,uShaky,uConfidence,uInkSoft;
uniform vec3 uIri[5];
out vec4 O;
float h21(vec2 p){p=fract(p*vec2(234.34,435.345));p+=dot(p,p+34.23);return fract(p.x*p.y);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){float v=0.,a=.5;
  for(int i=0;i<4;i++){v+=a*vnoise(p);p=p*2.03+vec2(1.7,9.2);a*=.55;}return v;}
vec3 iriAt(float t){
  float f=clamp(t,0.,1.)*4.0; int i=int(floor(f)); float fr=f-float(i);
  return mix(uIri[i],uIri[min(i+1,4)],fr);
}
float envAt(float u){
  float f=clamp(u,0.,1.)*float(${ENVN - 1}); int i=int(floor(f)); float fr=f-float(i);
  return mix(uEnv[i],uEnv[min(i+1,${ENVN - 1})],fr);
}
float segDist(vec2 p,vec2 a,vec2 b){
  vec2 pa=p-a,ba=b-a; float h=clamp(dot(pa,ba)/max(dot(ba,ba),1e-6),0.,1.);
  return length(pa-ba*h);
}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*uRes)/uRes.y;

  /* background -- the SAME top(ground-0)/base(ground-1) gradient the Canvas2D fill it replaces used */
  float vy=1.0-gl_FragCoord.y/uRes.y; // gl_FragCoord.y is bottom-up; flip so vy=0 is the sky (top)
  vec3 bg=mix(uG0,uG1,vy);

  /* CLOUD volume -- data: entropyByDepth (uEnv), turbulence, width, crystalline, condensation (uT) */
  vec2 rel=uv-uCloudC;
  float warpSpeed=uTime*.15*max(uAct,.06);
  vec2 warp=vec2(fbm(rel*2.2+warpSpeed),fbm(rel*2.2+vec2(4.1,1.7)-warpSpeed))-0.5;
  rel+=warp*uTurb*(0.16-0.08*uCryst);
  float along=dot(rel,uAxis);
  float u=clamp(along/max(uCloudH,1e-4)*0.5+0.5,0.,1.);
  vec2 perp=rel-uAxis*along;
  float across=length(perp);
  float w=envAt(u)*uWidth;
  float band=mix(.16,.05,uCryst);
  float edge=w-across;
  float cloudMask=smoothstep(-band,band,edge)*smoothstep(0.0,0.12,u)*smoothstep(1.0,0.82,u);
  cloudMask*=uT;
  float sheetFlash=uSheet*(0.5+0.5*fbm(rel*3.0+uTime*.4));
  float cloudGlow=cloudMask*(0.55+0.65*fbm(rel*1.6-warpSpeed*.6))*(1.0+sheetFlash*1.1);
  vec3 cloudTint=iriAt(u*.85+.05);
  // commit stratum -- data: meta.commitLayer (uCommitU); a NEUTRAL brightening (tints toward
  // uInkSoft, the exact color the Canvas2D commit ring above already uses), distinct from the
  // candidate-color cloud tint so it reads as a structural event, not another cell.
  float commitGlow=smoothstep(0.10,0.0,abs(u-uCommitU))*uT*0.6;

  /* BOLT glow -- data: active strike polylines, projected each frame from the real bolt segments */
  float boltG=0.0; vec3 boltCol=vec3(0.);
  for(int b=0;b<${BMAX};b++){
    float life=uBoltLife[b]; if(life<=0.0) continue;
    float d=1e6;
    for(int i=0;i<${PMAX - 1};i++){
      d=min(d,segDist(uv,uBoltPts[b*${PMAX}+i],uBoltPts[b*${PMAX}+i+1]));
    }
    float g=life*(0.0026/(d*d+0.0013));
    boltG+=g; boltCol+=uIri[2]*g; // iri-peri -- matches the Canvas2D strike's own glow hue
  }

  /* GROUND EMBERS -- data: planted words' entropy-driven breathing (same amount as the Canvas2D glow) */
  float emberG=0.0; vec3 emberCol=vec3(0.);
  for(int e=0;e<${EMAX};e++){
    float amt=uEmberAmt[e]; if(amt<=0.0) continue;
    float d=length(uv-uEmberPts[e]);
    float g=amt*(0.0017/(d*d+0.0022));
    vec3 hueCol=mix(mix(uConfidence,uShaky,step(0.5,uEmberHue[e])),uFork,step(1.5,uEmberHue[e]));
    emberG+=g; emberCol+=hueCol*g;
  }

  /* DUAL PARITY -- dawn paints TOWARD the hue (mix), never adds toward white; night adds light. */
  vec3 dawn=bg;
  dawn=mix(dawn,cloudTint,clamp(cloudGlow,0.,1.)*.62);
  dawn=mix(dawn,uInkSoft,clamp(commitGlow,0.,1.));
  dawn=mix(dawn,uIri[2],clamp(boltG,0.,1.));
  dawn=mix(dawn,emberCol/max(emberG,1e-4),clamp(emberG,0.,1.));

  vec3 night=bg;
  night+=cloudTint*cloudGlow*.85;
  night+=uInkSoft*commitGlow*.7;
  night+=boltCol;
  night+=emberCol;

  vec3 col=mix(dawn,night,uNight);
  O=vec4(col,1.0);
}`;

function createAtmosphere(glCanvas) {
  const gl = glCanvas.getContext("webgl2", { antialias: false, alpha: false });
  if (!gl) return null; // caller's Canvas2D-only path remains, untouched
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(s); gl.deleteShader(s); throw new Error(log || "shader compile failed");
    }
    return s;
  };
  let prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, ATMO_VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, ATMO_FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog) || "link failed");
  } catch {
    return null; // a compile/link failure must degrade to "no atmosphere", never break the storm
  }
  gl.useProgram(prog);
  const U = n => gl.getUniformLocation(prog, n);
  const loc = {
    uRes: U("uRes"), uTime: U("uTime"), uNight: U("uNight"), uAct: U("uAct"), uT: U("uT"),
    uTurb: U("uTurb"), uWidth: U("uWidth"), uCryst: U("uCryst"), uSheet: U("uSheet"),
    uCommitU: U("uCommitU"), uCloudC: U("uCloudC"), uAxis: U("uAxis"), uCloudH: U("uCloudH"),
    uEnv: U("uEnv"), uBoltPts: U("uBoltPts"), uBoltLife: U("uBoltLife"),
    uEmberPts: U("uEmberPts"), uEmberAmt: U("uEmberAmt"), uEmberHue: U("uEmberHue"),
    uG0: U("uG0"), uG1: U("uG1"),
    uFork: U("uFork"), uShaky: U("uShaky"), uConfidence: U("uConfidence"), uInkSoft: U("uInkSoft"),
    uIri: U("uIri"),
  };
  let dead = false;

  function setColors(colors) {
    if (dead) return;
    gl.useProgram(prog);
    const set3 = (l, rgb) => gl.uniform3f(l, rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
    set3(loc.uG0, colors["ground-0"]); set3(loc.uG1, colors["ground-1"]);
    set3(loc.uFork, colors.fork); set3(loc.uShaky, colors.shaky);
    set3(loc.uConfidence, colors.confidence); set3(loc.uInkSoft, colors["ink-soft"]);
    const flat = new Float32Array(15);
    colors.IRI.forEach((rgb, i) => { flat[i * 3] = rgb[0] / 255; flat[i * 3 + 1] = rgb[1] / 255; flat[i * 3 + 2] = rgb[2] / 255; });
    gl.uniform3fv(loc.uIri, flat);
  }
  function resize(wDev, hDev) { if (!dead) gl.viewport(0, 0, wDev, hDev); }
  function draw(u) {
    if (dead) return;
    gl.useProgram(prog);
    gl.uniform2f(loc.uRes, u.resW, u.resH);
    gl.uniform1f(loc.uTime, u.time);
    gl.uniform1f(loc.uNight, u.night);
    gl.uniform1f(loc.uAct, u.act);
    gl.uniform1f(loc.uT, u.t);
    gl.uniform1f(loc.uTurb, u.turb);
    gl.uniform1f(loc.uWidth, u.width);
    gl.uniform1f(loc.uCryst, u.cryst);
    gl.uniform1f(loc.uSheet, u.sheet);
    gl.uniform1f(loc.uCommitU, u.commitU);
    gl.uniform2f(loc.uCloudC, u.cloudC[0], u.cloudC[1]);
    gl.uniform2f(loc.uAxis, u.axis[0], u.axis[1]);
    gl.uniform1f(loc.uCloudH, u.cloudH);
    gl.uniform1fv(loc.uEnv, u.env);
    gl.uniform2fv(loc.uBoltPts, u.boltPts);
    gl.uniform1fv(loc.uBoltLife, u.boltLife);
    gl.uniform2fv(loc.uEmberPts, u.emberPts);
    gl.uniform1fv(loc.uEmberAmt, u.emberAmt);
    gl.uniform1fv(loc.uEmberHue, u.emberHue);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  function destroy() {
    if (dead) return;
    dead = true;
    const ext = gl.getExtension("WEBGL_lose_context");
    if (ext) ext.loseContext();
  }
  return { setColors, resize, draw, destroy };
}

/* ============================================================================ chrome (DOM + injected, scoped styles) */

const PHASE = { REST: 0, CONDENSE: 1, STORM: 2, RELEASE: 3, INSPECT: 4 };
const PHASE_LABEL = {
  [PHASE.REST]: "AT REST", [PHASE.CONDENSE]: "GATHERING", [PHASE.STORM]: "DECIDING",
  [PHASE.RELEASE]: "RELEASING", [PHASE.INSPECT]: "INSPECTING — click a word, fork a ghost",
};
const DUR = { [PHASE.REST]: 2.4, [PHASE.CONDENSE]: 3.4, [PHASE.RELEASE]: 4.0 };

function ensureStyles() {
  if (document.getElementById("cst-styles")) return;
  const style = document.createElement("style");
  style.id = "cst-styles";
  style.textContent = `
    .cst-root{position:relative;width:100%;height:100%;overflow:hidden;isolation:isolate;
      border-radius:var(--r-pane,13px);border:1px solid var(--pearl-edge,var(--line,#333));
      background:var(--ground-0,#0B0D1E)}
    .cst-gl{position:absolute;inset:0;width:100%;height:100%;display:block;z-index:0;pointer-events:none}
    .cst-canvas{position:absolute;inset:0;z-index:1;display:block;width:100%;height:100%;cursor:grab;touch-action:none}
    .cst-canvas:active{cursor:grabbing}
    .cst-note{position:absolute;left:14px;top:12px;max-width:min(320px,44%);
      font-family:var(--voice-machine);font-size:var(--text-label,11px);line-height:1.6;
      color:var(--ink-faint);pointer-events:none;text-shadow:0 1px 6px rgba(0,0,0,.45)}
    .cst-phase{position:absolute;right:14px;top:12px;font-family:var(--voice-machine);
      font-size:var(--text-label,11px);letter-spacing:var(--track-label,.1em);text-transform:uppercase;
      color:var(--ink-soft);pointer-events:none;text-shadow:0 1px 6px rgba(0,0,0,.45)}
    .cst-phase b{color:var(--fork);font-weight:600}
    .cst-hud{position:absolute;left:14px;bottom:12px;font-family:var(--voice-machine);
      font-size:var(--text-label,11px);line-height:1.8;color:var(--ink-soft);pointer-events:none;
      text-shadow:0 1px 6px rgba(0,0,0,.45)}
    .cst-hud b{color:var(--ink);font-variant-numeric:tabular-nums;font-weight:600}
    .cst-center{position:absolute;right:12px;bottom:12px;font:inherit;font-family:var(--voice-machine);
      font-size:var(--text-label,11px);letter-spacing:.08em;color:var(--ink-soft);
      background:color-mix(in srgb,var(--surface,#151833) 82%,transparent);
      border:1px solid var(--line);border-radius:var(--r-pill,18px);padding:5px 12px;cursor:pointer}
    .cst-center:hover{color:var(--ink);border-color:var(--line-strong)}
    .cst-center:focus-visible{outline:none;box-shadow:var(--focus,0 0 0 2px currentColor)}
  `;
  document.head.appendChild(style);
}

/* ============================================================================ mountCasting */

export function mountCasting(container, opts = {}) {
  if (!container || typeof container.appendChild !== "function") {
    throw new TypeError("mountCasting(container, opts): container must be a DOM element");
  }
  ensureStyles();

  const reduce = opts.reducedMotion != null ? !!opts.reducedMotion
    : matchMedia("(prefers-reduced-motion: reduce)").matches;
  const onSelect = typeof opts.onSelect === "function" ? opts.onSelect : () => {};

  container.innerHTML = `
    <div class="cst-root">
      <canvas class="cst-gl" aria-hidden="true"></canvas>
      <canvas class="cst-canvas" aria-label="${escapeAttr(opts.ariaLabel
        || "A storm cloud that inhales sky words along influence threads, argues via lead-change "
        + "flashes, strikes answer words into the ground, and can be forked from any almost-chosen token")}"></canvas>
      <div class="cst-note"></div>
      <div class="cst-phase">phase <b>AT REST</b></div>
      <div class="cst-hud">
        entropy <b class="cst-ent">—</b> bits<br>
        effective rank <b class="cst-rank">—</b><br>
        commit depth <b class="cst-depth">—</b>
      </div>
      <button type="button" class="cst-center">⌖ center on the words</button>
    </div>`;
  const rootEl = container.querySelector(".cst-root");
  const glCanvas = container.querySelector(".cst-gl");
  const canvas = container.querySelector(".cst-canvas");
  const noteEl = container.querySelector(".cst-note");
  const phaseEl = container.querySelector(".cst-phase b");
  const entVal = container.querySelector(".cst-ent");
  const rankVal = container.querySelector(".cst-rank");
  const depthVal = container.querySelector(".cst-depth");
  const centerBtn = container.querySelector(".cst-center");
  const ctx = canvas.getContext("2d");
  if (!ctx) { container.innerHTML = ""; return null; }

  // atmosphere is an ENHANCEMENT ONLY -- graceful absence: no WebGL2 (or a compile/link failure)
  // means createAtmosphere() returns null, the extra canvas is dropped, and every `if (atmosphere)`
  // branch below is simply skipped, leaving the Canvas2D path identical to before this layer existed.
  const atmosphere = createAtmosphere(glCanvas);
  if (!atmosphere) glCanvas.remove();

  let colors = readTokens();
  let atmNightTarget = deriveNight(colors), atmNightT = atmNightTarget;
  let atmAct = 0, atmActT = 0, atmT = 20; // atmT offset matches light.mjs's own silkT convention
  if (atmosphere) atmosphere.setColors(colors);
  function refreshTokens() {
    colors = readTokens();
    if (particles.length) for (const p of particles) p.c = iriAt(colors.IRI, p.colorSeed);
    atmNightTarget = deriveNight(colors);
    if (atmosphere) atmosphere.setColors(colors);
  }

  let W = 0, H = 0, DPR = 1;
  function resize() {
    DPR = Math.min(devicePixelRatio || 1, 2);
    const b = canvas.getBoundingClientRect();
    W = b.width; H = b.height;
    canvas.width = Math.max(1, W * DPR); canvas.height = Math.max(1, H * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    if (atmosphere) {
      glCanvas.width = canvas.width; glCanvas.height = canvas.height;
      atmosphere.resize(canvas.width, canvas.height);
    }
    if (reduce) renderOnce();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  const themeObserver = new MutationObserver(refreshTokens);
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const prefersDark = matchMedia("(prefers-color-scheme: dark)");
  const onPrefersChange = () => refreshTokens();
  prefersDark.addEventListener("change", onPrefersChange);

  /* ---------------- particle field (built once; reused across every cast, colored by tokens) */
  const particles = [];
  {
    let s = 7;
    const rng = () => (s = Math.imul(48271, s) % 2147483647) / 2147483647;
    for (let i = 0; i < N_PARTICLES; i++) {
      const g = () => (rng() + rng() + rng() - 1.5) * 1.05;
      const colorSeed = rng();
      particles.push({
        rest: [g() * 1.35, g() * .9, g() * 1.35],
        y: Y_TOP + Math.pow(rng(), .85) * (Y_BASE - Y_TOP),
        th: rng() * 6.28318, ru: Math.sqrt(rng()), lr: rng(),
        ph: rng() * 6.28, ph2: rng() * 6.28, sp: .3 + rng() * 1.0,
        colorSeed, c: iriAt(colors.IRI, colorSeed),
        b: rng() < .10 ? 1.9 : (.5 + rng() * .8),
      });
    }
  }

  /* ---------------- run state */
  let preset = derivePreset({ entropyByDepth: [1], sky: [], tokens: [] });
  let currentData = null, pendingData = null;
  let phase = PHASE.REST, ph_t = 0, t = 0;
  let tokIdx = -1, seqOffset = 0, seqTotal = 0, seqCell = 0, stormDur = 4, perRow = 7;
  let SEQ = [];
  let bolts = [], crawlers = [], sparks = [], planted = [], sel = null, forked = false;
  let sheet = 0, allFlash = 0, firedFlips = new Set();
  let ghostRects = [];
  let dead = false;

  function applyData(data) {
    currentData = data;
    preset = derivePreset(data);
    bolts = []; crawlers = []; sparks = []; sel = null;
  }

  function setPhase(p) {
    phase = p; ph_t = 0;
    phaseEl.textContent = PHASE_LABEL[p];
    rootEl.dataset.phase = PHASE_LABEL[p];
    if (p === PHASE.CONDENSE) {
      planted = []; sel = null; forked = false;
      SEQ = preset.tokens; seqOffset = 0; seqTotal = SEQ.length; seqCell = 0;
      stormDur = Math.max(4, SEQ.length * .7);
      perRow = rowSize(seqTotal);
      firedFlips = new Set();
      noteEl.textContent = preset.meta.note || "";
    }
    if (p === PHASE.STORM) tokIdx = -1;
    if (p === PHASE.INSPECT) allFlash = 1.4;
  }

  function fastForward(data) {
    applyData(data);
    t = 1; phase = PHASE.INSPECT; ph_t = 0;
    phaseEl.textContent = PHASE_LABEL[PHASE.INSPECT];
    rootEl.dataset.phase = PHASE_LABEL[PHASE.INSPECT];
    noteEl.textContent = preset.meta.note || "";
    planted = [];
    const tot = preset.tokens.length, pr2 = rowSize(tot);
    preset.tokens.forEach((tk, i) => {
      const pos = posFor(i, tot, pr2);
      planted.push({
        word: tk.text, ent: tk.entropy, src: tk.sources, almosts: tk.almosts,
        x: pos.gx, z: pos.gz, bolt: null, ghosts: [], br: false, tokenIndex: i,
      });
    });
    tokIdx = tot - 1;
    SEQ = preset.tokens; seqOffset = 0; seqTotal = tot; perRow = pr2;
    renderOnce();
  }

  function extendCurrent(newData) {
    currentData = newData;
    const oldSeqTotal = Math.max(1, seqTotal);
    preset = derivePreset(newData);
    SEQ = preset.tokens; seqTotal = SEQ.length; perRow = rowSize(seqTotal - seqOffset);
    const perTokenDur = stormDur / Math.max(1, oldSeqTotal - seqOffset);
    stormDur = perTokenDur * Math.max(1, seqTotal - seqOffset);
    if (phase === PHASE.INSPECT) setPhase(PHASE.STORM); // more tokens streamed in after we'd settled
  }

  function update(data) {
    if (data === null) {
      if (reduce) return;
      if (currentData) { pendingData = null; setPhase(PHASE.RELEASE); }
      return;
    }
    if (!data || typeof data !== "object" || !Array.isArray(data.tokens)) return;
    if (reduce) { fastForward(data); return; }
    if (currentData && isContinuation(currentData, data)
        && (phase === PHASE.STORM || phase === PHASE.CONDENSE || phase === PHASE.INSPECT)) {
      extendCurrent(data); return;
    }
    if (!currentData) { pendingData = data; return; } // let the opening idle breath play once
    if (t > .1) { pendingData = data; setPhase(PHASE.RELEASE); }
    else { applyData(data); setPhase(PHASE.CONDENSE); }
  }

  function setNight(nightArg) {
    refreshTokens(); // re-reads tokens.css; also derives atmNightTarget as a CSS-luminance fallback
    if (typeof nightArg === "boolean") atmNightTarget = nightArg ? 1 : 0; // the host's explicit, immediate flip wins
    if (reduce) renderOnce();
  }

  /* ---------------- camera + interaction */
  let yaw = .45, pitch = .22, dragging = false, moved = false, lx = 0, ly = 0, autoRot = true;
  let mx = -1, my = -1;
  const pr = [0, 0, 0];
  function project(p, out) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    let x = p[0] * cy + p[2] * sy, z = -p[0] * sy + p[2] * cy, y = p[1] * cp - z * sp;
    z = p[1] * sp + z * cp;
    const d = 3.1 / (3.1 + z), s = Math.min(W, H) * .36;
    out[0] = W * .5 + x * s * d; out[1] = H * .47 + y * s * d; out[2] = d;
    return out;
  }

  function onPointerDown(e) { dragging = true; moved = false; autoRot = false; lx = e.clientX; ly = e.clientY; }
  function onPointerUp() { dragging = false; }
  function onCanvasPointerMove(e) {
    const b = canvas.getBoundingClientRect(); mx = e.clientX - b.left; my = e.clientY - b.top;
  }
  function onWindowPointerMove(e) {
    if (!dragging) return;
    if (Math.abs(e.clientX - lx) + Math.abs(e.clientY - ly) > 3) moved = true;
    yaw += (e.clientX - lx) * .005;
    pitch = clamp(pitch + (e.clientY - ly) * .004, -.9, .9);
    lx = e.clientX; ly = e.clientY;
    if (reduce) renderOnce();
  }
  function onCenter() { yaw = 0; pitch = .14; autoRot = false; if (reduce) renderOnce(); }
  function onClick() {
    if (moved) return;
    for (const r of ghostRects) {
      if (mx >= r.x && mx <= r.x + r.w && my >= r.y - 10 && my <= r.y + 6 && r.forkable) { doFork(r); return; }
    }
    for (const e of planted) {
      project([e.x, Y_GROUND, e.z], pr);
      if (Math.abs(pr[0] - mx) < 26 && Math.abs(pr[1] - my) < 14) {
        sel = (sel === e ? null : e);
        if (sel) {
          try { onSelect(sel.tokenIndex); } catch { /* host callbacks must not break canvas interaction */ }
        }
        if (reduce) renderOnce();
        return;
      }
    }
    sel = null; if (reduce) renderOnce();
  }
  function selectToken(index) {
    const value = Number(index);
    if (!Number.isFinite(value) || !planted.length) return;
    const clamped = Math.max(0, Math.min(preset.tokens.length - 1, Math.trunc(value)));
    sel = planted.find(entry => entry.tokenIndex === clamped) || null;
    if (reduce) renderOnce();
  }
  canvas.addEventListener("pointerdown", onPointerDown);
  addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointermove", onCanvasPointerMove);
  addEventListener("pointermove", onWindowPointerMove);
  canvas.addEventListener("click", onClick);
  centerBtn.addEventListener("click", onCenter);

  function doFork(rect) {
    const idx = planted.indexOf(rect.plantedEntry);
    if (idx < 0) return;
    const tokenIndex = rect.plantedEntry.tokenIndex, altText = rect.alt;
    if (typeof opts.onFork === "function") {
      try { opts.onFork(tokenIndex, altText); } catch { /* a host callback error must not break the storm */ }
    }
    if (!rect.continuation || !rect.continuation.length) { return; } // await the host's update() with the real branch
    planted = planted.slice(0, idx); sel = null; forked = true;
    SEQ = rect.continuation.map(c => ({
      text: String(c.text != null ? c.text : ""),
      entropy: Number.isFinite(c.entropy) ? c.entropy : 0,
      sources: [], almosts: [],
    }));
    seqOffset = idx; seqTotal = idx + SEQ.length; perRow = rowSize(seqTotal - seqOffset);
    seqCell = preset.cells.length > 1 ? 1 : 0; // the OTHER cell, per the grammar
    stormDur = Math.max(3, SEQ.length * .7);
    noteEl.textContent = (preset.meta.note || "") + `  ⑂ forked: "${rect.plantedWord}" → "${altText}"`;
    setPhase(PHASE.STORM);
  }

  /* ---------------- simulation step (no drawing) */
  let lastTime = performance.now();
  function step(dt) {
    if (!reduce) { ph_t += dt; if (autoRot && !dragging) yaw += dt * .05; }

    if (phase === PHASE.REST && pendingData && ph_t > DUR[PHASE.REST]) {
      const nd = pendingData; pendingData = null; applyData(nd); setPhase(PHASE.CONDENSE);
    }
    if (phase === PHASE.CONDENSE) {
      t = ease(clamp(ph_t / DUR[PHASE.CONDENSE], 0, 1));
      preset.leadFlips.forEach(f => {
        const due = (f[0] + 1) / (preset.nL + 1);
        if (!firedFlips.has(f) && t > due) {
          firedFlips.add(f);
          const arc = flipArc(preset, f);
          if (arc) { crawlers.push(arc); sheet = Math.min(1, sheet + .6); }
        }
      });
      if (ph_t > DUR[PHASE.CONDENSE]) setPhase(PHASE.STORM);
    }
    if (phase === PHASE.STORM) {
      t = 1;
      const k = Math.min(SEQ.length - 1, (ph_t / stormDur * SEQ.length) | 0);
      if (k > tokIdx) {
        tokIdx = k;
        const tk = SEQ[k];
        const pos = posFor(seqOffset + k, seqTotal, perRow);
        const b = groundBolt(tk.entropy, preset, seqCell, pos.gx, pos.gz, tk.almosts);
        bolts.push(b); sheet = Math.min(1, sheet + .8);
        planted.push({
          word: tk.text, ent: tk.entropy, src: tk.sources || [], almosts: tk.almosts || [],
          x: pos.gx, z: pos.gz, bolt: b.segs, ghosts: b.ghosts, br: forked, tokenIndex: seqOffset + k,
        });
        for (let i = 0; i < 10 * tk.entropy; i++) {
          sparks.push({ p: [pos.gx + (Math.random() - .5) * .15, Y_GROUND, pos.gz + (Math.random() - .5) * .15], life: 1 });
        }
      }
      if (ph_t > stormDur) setPhase(PHASE.INSPECT);
    }
    if (phase === PHASE.INSPECT) {
      t = 1;
      if (preset.leadFlips.length && Math.random() < dt * .25) { // faint echoes of the argument
        const arc = flipArc(preset, preset.leadFlips[(Math.random() * preset.leadFlips.length) | 0]);
        if (arc) crawlers.push(arc);
      }
    }
    if (phase === PHASE.RELEASE) {
      t = 1 - ease(clamp(ph_t / DUR[PHASE.RELEASE], 0, 1));
      if (ph_t > DUR[PHASE.RELEASE]) {
        if (pendingData) { const nd = pendingData; pendingData = null; applyData(nd); setPhase(PHASE.CONDENSE); }
        else setPhase(PHASE.REST);
      }
    }
    sheet = Math.max(0, sheet - dt * 2.2);
    allFlash = Math.max(0, allFlash - dt * .9);
    crawlers.forEach(c => c.life -= dt * 2.6); crawlers = crawlers.filter(c => c.life > 0);
    bolts.forEach(b => b.life -= dt * 2.2); bolts = bolts.filter(b => b.life > 0);
    sparks.forEach(s => { s.life -= dt * 1.6; s.p[0] *= 1.02; s.p[2] *= 1.02; s.p[1] += dt * .10; });
    sparks = sparks.filter(s => s.life > 0);
    updateHud();

    /* atmosphere clock -- MOTION = MEANING: idle is a near-still breath; real events (strikes via
       `sheet`, the all-bolt flare via `allFlash`, the phase itself) are what earn faster time/glow.
       Under reduced motion (dt===0, no RAF to let smoothing converge) both targets are snapped
       immediately so the single static frame is still correct, never a stale lag from mount. */
    if (atmosphere) {
      atmAct = clamp(0.08 + sheet * .7 + allFlash * .35 + (
        phase === PHASE.STORM ? .5 : phase === PHASE.CONDENSE ? .4 : phase === PHASE.RELEASE ? .3
        : phase === PHASE.INSPECT ? .15 : 0
      ), 0, 1);
      const actRate = reduce ? 1 : (atmAct > atmActT ? .12 : .03);
      atmActT += (atmAct - atmActT) * actRate;
      atmT += dt * (0.05 + 0.95 * atmActT);
      atmNightT += (atmNightTarget - atmNightT) * (reduce ? 1 : .06);
    }
  }

  function updateHud() {
    if (phase === PHASE.REST) { entVal.textContent = "—"; rankVal.textContent = "—"; depthVal.textContent = "—"; return; }
    let entropy;
    if ((phase === PHASE.STORM || phase === PHASE.INSPECT) && SEQ.length) {
      entropy = SEQ[clamp(tokIdx, 0, SEQ.length - 1)].entropy;
    } else {
      const first = preset.entropyByDepth[0], last = preset.entropyByDepth[preset.entropyByDepth.length - 1];
      entropy = first + (last - first) * t;
    }
    entVal.textContent = entropy.toFixed(1);
    if (preset.rankByDepth) {
      const first = preset.rankByDepth[0], last = preset.rankByDepth[preset.rankByDepth.length - 1];
      rankVal.textContent = Math.round(first + (last - first) * t).toLocaleString();
    } else rankVal.textContent = "—";
    if (t > .9 && preset.readoutLayers) {
      const layer = preset.readoutLayers[preset.commitLayerIdx];
      depthVal.textContent = preset.totalLayers ? `L${layer} of ${preset.totalLayers}` : `L${layer}`;
    } else if (t > .9 && !preset.readoutLayers) {
      depthVal.textContent = `stratum ${preset.commitLayerIdx + 1}/${preset.nL}`;
    } else depthVal.textContent = "—";
  }

  /* ---------------- draw (reads state written by step(); never mutates simulation state) */
  function draw(time) {
    ctx.globalCompositeOperation = "source-over";
    if (atmosphere) {
      // the WebGL2 layer beneath paints the sky/ground gradient + cloud volume; leave it transparent here
      ctx.clearRect(0, 0, W, H);
    } else {
      const grad = ctx.createLinearGradient(0, 0, 0, H); // SKY at top, GROUND at bottom -- literal fit
      grad.addColorStop(0, rgbaC(colors["ground-0"], 1));
      grad.addColorStop(1, rgbaC(colors["ground-1"], 1));
      ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);
    }
    ctx.globalCompositeOperation = "lighter";
    ghostRects = [];

    let hov = sel;
    if (!hov) for (const e of planted) {
      project([e.x, Y_GROUND, e.z], pr);
      if (Math.abs(pr[0] - mx) < 26 && Math.abs(pr[1] - my) < 14) hov = e;
    }
    const srcW = si => { if (!hov) return 0; const s = (hov.src || []).find(s => s[0] === si); return s ? s[1] : 0; };

    /* SKY words -- data: sky[].text/weight; source-lit words use tokens.evidence (provenance) */
    preset.sky.forEach((w, si) => {
      const active = phase === PHASE.CONDENSE ? .85
        : (phase === PHASE.STORM || phase === PHASE.RELEASE || phase === PHASE.INSPECT) ? w.weight : .18;
      const hw = srcW(si);
      project([w.x, w.y, w.z], pr);
      ctx.font = (w.weight > .5 ? "600 " : "") + "12px " + colors.font;
      ctx.textAlign = "center";
      ctx.fillStyle = hw > 0 ? rgbaC(colors.evidence, .55 + .45 * hw) : rgbaC(colors.ink, .25 + .55 * active * Math.max(t, .3));
      ctx.fillText(w.text, pr[0], pr[1]);
      if (hw > 0) {
        const rg = ctx.createRadialGradient(pr[0], pr[1] - 3, 0, pr[0], pr[1] - 3, 10 + 16 * hw);
        rg.addColorStop(0, rgbaC(colors.evidence, .25 + .4 * hw)); rg.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(pr[0], pr[1] - 3, 10 + 16 * hw, 0, 6.283); ctx.fill();
      }
    });
    /* intake THREADS -- data: sky[].weight (idle draw strength) + hov.src (per-token provenance) */
    if (t > .05) preset.sky.forEach((w, si) => {
      const active = phase === PHASE.CONDENSE ? .9
        : (phase === PHASE.STORM || phase === PHASE.INSPECT) ? w.weight
        : phase === PHASE.RELEASE ? w.weight * .4 : 0;
      if (active < .03) return;
      const hw = srcW(si), lit = hw > 0;
      ctx.beginPath();
      for (let i = 0; i <= 16; i++) {
        const u = i / 16;
        const yy = w.y + .04 + (Y_TOP + .06 - w.y) * u * u * (2 - u);
        const sway = Math.sin(u * 5 + time * 1.1 + si) * .03 * u;
        project([w.x * (1 - .75 * u) + sway, yy, w.z * (1 - .8 * u)], pr);
        i ? ctx.lineTo(pr[0], pr[1]) : ctx.moveTo(pr[0], pr[1]);
      }
      ctx.setLineDash([5, 9]);
      ctx.lineDashOffset = -(time * 46 * (0.4 + w.weight));
      ctx.strokeStyle = lit ? rgbaC(colors.evidence, .35 + .55 * hw) : rgbaC(colors["iri-peri"], .10 + .38 * active * t);
      ctx.lineWidth = lit ? 1.4 + 2 * hw : .6 + 1.6 * w.weight * t;
      ctx.stroke(); ctx.setLineDash([]);
    });

    /* CLOUD particles -- data: entropyByDepth (silhouette), turbulence (roil), width, crystalline */
    const glow = 1 + sheet * .9;
    for (const p of particles) {
      const ci = cellOf(p, preset), cell = preset.cells[ci], sep = cellSep(p.y, preset);
      const wHere = cloudW(p.y, preset);
      const r = wHere * p.ru * (1 - (1 - Math.sqrt(cell.share || 0)) * sep);
      let ox = Math.cos(p.th) * r + cell.x * sep * preset.width;
      let oz = Math.sin(p.th) * r * .72 + cell.z * sep * preset.width, oy = p.y;
      const tb = preset.turbulence * (.045 + .03 * Math.sin(time * .13));
      ox += Math.sin(time * .5 * p.sp + p.ph) * tb + Math.sin(time * .21 + p.ph2 + oy * 3.1) * tb * .7;
      oz += Math.cos(time * .43 * p.sp + p.ph2) * tb + Math.sin(time * .17 + p.ph + ox * 2.7) * tb * .7;
      oy += Math.sin(time * .31 * p.sp + p.ph) * tb * .45;
      const x = p.rest[0] + (ox - p.rest[0]) * t, y = p.rest[1] + (oy - p.rest[1]) * t, z = p.rest[2] + (oz - p.rest[2]) * t;
      project([x, y, z], pr);
      const a = (.10 + .20 * t) * p.b * pr[2] * (preset.crystalline ? 1.25 : 1) * glow;
      ctx.fillStyle = rgbaC(p.c, Math.min(a, .9));
      const s = (p.b > 1.5 ? 2.2 : 1.4) * pr[2] * (preset.crystalline ? .85 : 1);
      ctx.fillRect(pr[0], pr[1], s, s);
    }
    /* candidate cells -- data: cells[].x/z/share/label */
    if (t > .5) preset.cells.forEach((cell, ci) => {
      const yC = (Y_TOP + preset.commitY) / 2, sep = cellSep(yC, preset);
      if (preset.cells.length > 1 && sep < .05) return;
      project([cell.x * sep * preset.width, yC, cell.z * sep * preset.width], pr);
      const c = iriAt(colors.IRI, ci / Math.max(1, preset.cells.length - 1) * .9);
      const rad = (30 * (cell.share || 0) + 10) * t * (1 + sheet * .35) * pr[2];
      const rg = ctx.createRadialGradient(pr[0], pr[1], 0, pr[0], pr[1], rad);
      rg.addColorStop(0, rgbaC(colors.ink, .30 * t * (1 + sheet * .6)));
      rg.addColorStop(.4, rgbaC(c, .18 * t)); rg.addColorStop(1, rgbaC(c, 0));
      ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(pr[0], pr[1], rad, 0, 6.283); ctx.fill();
      if (cell.label && sep > .12) {
        ctx.fillStyle = rgbaC(colors.ink, .72 * sep * t); ctx.font = "600 10px " + colors.font;
        ctx.textAlign = "center"; ctx.fillText(cell.label, pr[0], pr[1] - rad * .5 - 4);
      }
    });
    /* commit stratum -- data: meta.commitLayer (+ readoutLayers/totalLayers for the label) */
    if (t > .72) {
      const y = preset.commitY, rr = cloudW(y, preset) * 1.15, pulse = .5 + .5 * Math.sin(time * 2.6);
      ctx.beginPath();
      for (let a2 = 0; a2 <= 48; a2++) {
        const th = a2 / 48 * 6.28318;
        project([Math.cos(th) * rr, y, Math.sin(th) * rr * .72], pr);
        a2 ? ctx.lineTo(pr[0], pr[1]) : ctx.moveTo(pr[0], pr[1]);
      }
      ctx.strokeStyle = rgbaC(colors["ink-soft"], (.16 + .14 * pulse) * t); ctx.lineWidth = 1.4; ctx.stroke();
      project([rr * 1.08, y, 0], pr);
      const layerLabel = preset.readoutLayers ? "L" + preset.readoutLayers[preset.commitLayerIdx]
        : "stratum " + (preset.commitLayerIdx + 1);
      ctx.fillStyle = rgbaC(colors["ink-faint"], .7 * t); ctx.font = "10px " + colors.font;
      ctx.textAlign = "left"; ctx.fillText("⊙ commit — " + layerLabel, pr[0] + 4, pr[1] + 3);
    }
    /* lead-change arcs -- data: leadFlips[[layer,fromCell,toCell]] */
    for (const c of crawlers) {
      drawPath(c.segs, rgbaC(colors.ink, .6 * c.life), 1.3);
      drawPath(c.segs, rgbaC(colors.fork, .25 * c.life), 3.4);
    }
    /* live strikes -- data: tokens[].entropy (violence) via groundBolt() */
    for (const b of bolts) {
      const flick = (Math.random() < .25 ? 1.6 : 1) * b.life;
      drawPath(b.segs, rgbaC(colors.ink, .85 * flick), 2.2);
      drawPath(b.segs, rgbaC(colors["iri-peri"], .35 * flick), 5);
      if (b.snapIdx > 0) {
        project(b.segs[b.snapIdx], pr);
        ctx.fillStyle = rgbaC(colors.ink, .9 * flick);
        ctx.beginPath(); ctx.arc(pr[0], pr[1], 3.4 * flick + 1, 0, 6.283); ctx.fill();
      }
      drawGhosts(b.ghosts, b.life, null);
    }
    /* the whole answer's argument, one frame -- data: every planted[].bolt (stored per-token strikes) */
    if (phase === PHASE.INSPECT || allFlash > 0) {
      const base = (phase === PHASE.INSPECT ? .10 : 0) + .45 * allFlash;
      for (const e of planted) if (e.bolt && e !== hov) {
        drawPath(e.bolt, rgbaC(e.br ? colors.fork : colors["iri-peri"], base), 1);
      }
    }
    for (const s of sparks) {
      project(s.p, pr);
      ctx.fillStyle = rgbaC(colors.ink, .7 * s.life);
      ctx.fillRect(pr[0], pr[1], 1.6, 1.6);
    }

    /* GROUND -- decoration: concentric locator rings (no data claim; purely spatial orientation) */
    for (let ri = 0; ri < 2; ri++) {
      const rr = .95 + ri * .5;
      ctx.beginPath();
      for (let a2 = 0; a2 <= 48; a2++) {
        const th = a2 / 48 * 6.28318;
        project([Math.cos(th) * rr, Y_GROUND + .01, Math.sin(th) * rr * .5], pr);
        a2 ? ctx.lineTo(pr[0], pr[1]) : ctx.moveTo(pr[0], pr[1]);
      }
      ctx.strokeStyle = rgbaC(colors["line-strong"], .06 * Math.max(t, .2)); ctx.lineWidth = 1; ctx.stroke();
    }
    /* planted words -- data: tokens[].text/entropy; shaky=coral, confident=cool-blue, forked=lilac */
    planted.forEach(e => {
      const shaky = e.ent > SHAKY_ENTROPY_BITS, isHov = hov === e;
      const fl = shaky ? (.55 + .35 * Math.sin(time * 11 + e.x * 9) + (Math.random() < .05 ? -.25 : 0)) : 1;
      const c = e.br ? colors.fork : (shaky ? colors.shaky : colors.confidence);
      project([e.x, Y_GROUND, e.z], pr);
      ctx.font = (isHov ? "600 " : "") + "12px " + colors.font; ctx.textAlign = "center";
      ctx.fillStyle = rgbaC(c, (.55 + .4 * fl) * (isHov ? 1 : 0.92));
      ctx.fillText(e.word, pr[0], pr[1]);
      const rg = ctx.createRadialGradient(pr[0], pr[1] + 2, 0, pr[0], pr[1] + 2, isHov ? 22 : 12);
      rg.addColorStop(0, rgbaC(c, (shaky ? .28 : .18) * fl * (isHov ? 2 : 1))); rg.addColorStop(1, rgbaC(c, 0));
      ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(pr[0], pr[1] + 2, isHov ? 22 : 12, 0, 6.283); ctx.fill();
    });

    /* HOVER / SELECTED lineage -- data: hov.bolt/ghosts/src (weighted provenance) */
    if (hov) {
      if (hov.bolt) {
        drawPath(hov.bolt, rgbaC(colors.evidence, .9), 2.2);
        drawPath(hov.bolt, rgbaC(colors.evidence, .30), 6);
        drawGhosts(hov.ghosts, 1, hov);
      }
      const entry = hov.bolt ? hov.bolt[0] : [0, Y_TOP, 0];
      (hov.src || []).forEach(s => {
        const w = preset.sky[s[0]]; if (!w) return;
        const wt = s[1];
        const mid = [w.x * .45, (Y_SKY + Y_TOP) / 2 + .12, w.z * .45];
        const path = [[w.x, w.y + .03, w.z], mid, [entry[0] * .7, Y_TOP - .02, entry[2] * .7], entry];
        drawPath(path, rgbaC(colors.evidence, .35 + .6 * wt), 1 + 2.2 * wt);
        drawPath(path, rgbaC(colors.evidence, .12 + .25 * wt), 4 + 4 * wt);
      });
      project([hov.x, Y_GROUND + .10, hov.z], pr);
      ctx.font = "10px " + colors.font; ctx.textAlign = "center";
      ctx.fillStyle = rgbaC(colors.evidence, .95);
      const lbl = (hov.src && hov.src.length)
        ? ("fed by: " + hov.src.map(s => (preset.sky[s[0]] ? preset.sky[s[0]].text : "?") + " \xb7" + Math.round(s[1] * 100) + "%").join("  "))
        : "from the weights — not from your words";
      ctx.fillText(lbl, pr[0], pr[1] + 14);
      if (sel === hov && hov.almosts && hov.almosts.some(a => a.forkable)) {
        ctx.fillStyle = rgbaC(colors.fork, .9);
        ctx.fillText("click a ⑂ ghost to fork the timeline", pr[0], pr[1] + 27);
      }
    }
  }

  function drawGhosts(ghosts, life, plantedEntry) {
    if (!ghosts) return;
    for (const g of ghosts) {
      drawPath(g.segs, rgbaC(colors.almost, (.30 + .45 * g.pull) * life), 1.2);
      const tip = g.segs[g.segs.length - 1]; project(tip, pr);
      for (let d = 1; d <= 3; d++) {
        ctx.fillStyle = rgbaC(colors.almost, (.5 - .14 * d) * g.pull * life);
        ctx.fillRect(pr[0] + d * 3, pr[1] + d * 4, 1.6, 1.6);
      }
      const forkable = g.forkable && plantedEntry;
      ctx.fillStyle = forkable ? rgbaC(colors.fork, .85 * life) : rgbaC(colors.almost, (.45 + .4 * g.pull) * life);
      ctx.font = (forkable ? "600 " : "") + "9px " + colors.font; ctx.textAlign = "left";
      ctx.fillText(g.label, pr[0] + 6, pr[1] + 3);
      if (plantedEntry) {
        ghostRects.push({
          x: pr[0] + 6, y: pr[1] - 3, w: ctx.measureText(g.label).width, h: 12,
          forkable: g.forkable, continuation: g.continuation, alt: g.alt,
          plantedEntry, plantedWord: plantedEntry.word,
        });
      }
    }
  }
  function drawPath(segs, style, w) {
    ctx.beginPath();
    for (let i = 0; i < segs.length; i++) {
      project(segs[i], pr);
      i ? ctx.lineTo(pr[0], pr[1]) : ctx.moveTo(pr[0], pr[1]);
    }
    ctx.strokeStyle = style; ctx.lineWidth = w; ctx.stroke();
  }

  /* ---------------- atmosphere uniform prep (reads the SAME state draw() reads; never mutates it).
     Reuses casting.mjs's own project() camera each frame, so the shader's cloud axis/envelope/bolt
     glow/embers track the real orbit (yaw/pitch drag) exactly like the Canvas2D particle field. */
  const _atmScratchA = [0, 0, 0], _atmScratchB = [0, 0, 0];
  const _atmEnv = new Float32Array(ENVN);
  const _atmBoltPts = new Float32Array(BMAX * PMAX * 2);
  const _atmBoltLife = new Float32Array(BMAX);
  const _atmEmberPts = new Float32Array(EMAX * 2);
  const _atmEmberAmt = new Float32Array(EMAX);
  const _atmEmberHue = new Float32Array(EMAX);

  function toAtmUV(p3) {
    project(p3, _atmScratchA);
    const rx = canvas.width, ry = canvas.height || 1;
    return [
      (_atmScratchA[0] * DPR - 0.5 * rx) / ry,
      ((ry - _atmScratchA[1] * DPR) - 0.5 * ry) / ry, // flip to match the shader's bottom-up gl_FragCoord
    ];
  }

  function computeAtmoUniforms(nowSec) {
    const [tux, tuy] = toAtmUV([0, Y_TOP, 0]);
    const [bux, buy] = toAtmUV([0, Y_BASE, 0]);
    const cloudC = [(tux + bux) / 2, (tuy + buy) / 2];
    let axX = bux - tux, axY = buy - tuy;
    const axLen = Math.hypot(axX, axY) || 1e-4;
    axX /= axLen; axY /= axLen;

    for (let i = 0; i < ENVN; i++) {
      const u = i / (ENVN - 1);
      const y = Y_TOP + u * (Y_BASE - Y_TOP);
      const [cx, cy] = toAtmUV([0, y, 0]);
      const wLocal = cloudW(y, preset);
      const [ex, ey] = toAtmUV([wLocal, y, 0]);
      _atmEnv[i] = Math.hypot(ex - cx, ey - cy);
    }

    const nb = Math.min(BMAX, bolts.length);
    _atmBoltLife.fill(0);
    for (let bI = 0; bI < BMAX; bI++) {
      if (bI >= nb) continue;
      const b = bolts[bolts.length - nb + bI];
      _atmBoltLife[bI] = clamp(b.life, 0, 1);
      const segs = b.segs, n = segs.length;
      for (let i = 0; i < PMAX; i++) {
        const si = n > 1 ? Math.min(n - 1, Math.round((i * (n - 1)) / (PMAX - 1))) : 0;
        const [px, py] = toAtmUV(segs[si] || segs[0] || [0, 0, 0]);
        _atmBoltPts[(bI * PMAX + i) * 2] = px;
        _atmBoltPts[(bI * PMAX + i) * 2 + 1] = py;
      }
    }

    const ne = Math.min(EMAX, planted.length);
    _atmEmberAmt.fill(0);
    for (let eI = 0; eI < EMAX; eI++) {
      if (eI >= ne) continue;
      const e = planted[planted.length - ne + eI];
      const shaky = e.ent > SHAKY_ENTROPY_BITS;
      const fl = shaky ? (.55 + .35 * Math.sin(nowSec * 11 + e.x * 9)) : 1;
      _atmEmberAmt[eI] = (shaky ? .30 : .20) * fl;
      _atmEmberHue[eI] = e.br ? 2 : (shaky ? 1 : 0);
      const [ex, ey] = toAtmUV([e.x, Y_GROUND, e.z]);
      _atmEmberPts[eI * 2] = ex; _atmEmberPts[eI * 2 + 1] = ey;
    }

    return {
      resW: canvas.width, resH: canvas.height,
      time: atmT, night: atmNightT, act: atmActT, t,
      turb: preset.turbulence, width: preset.width, cryst: preset.crystalline ? 1 : 0,
      sheet, commitU: preset.nL > 1 ? preset.commitLayerIdx / (preset.nL - 1) : 0.5,
      cloudC, axis: [axX, axY], cloudH: axLen / 2,
      env: _atmEnv, boltPts: _atmBoltPts, boltLife: _atmBoltLife,
      emberPts: _atmEmberPts, emberAmt: _atmEmberAmt, emberHue: _atmEmberHue,
    };
  }

  function frame(now) {
    if (dead) return;
    const dt = Math.min(.05, (now - lastTime) / 1000); lastTime = now;
    step(dt);
    const nowSec = now / 1000;
    draw(nowSec);
    if (atmosphere) atmosphere.draw(computeAtmoUniforms(nowSec));
    if (!reduce) requestAnimationFrame(frame);
  }
  function renderOnce() {
    const nowSec = performance.now() / 1000;
    step(0);
    draw(nowSec);
    if (atmosphere) atmosphere.draw(computeAtmoUniforms(nowSec));
  }

  resize();
  if (opts.data) update(opts.data);
  if (reduce) renderOnce(); else requestAnimationFrame(frame);

  return {
    update,
    setNight,
    selectToken,
    destroy() {
      dead = true;
      if (atmosphere) atmosphere.destroy();
      ro.disconnect();
      themeObserver.disconnect();
      prefersDark.removeEventListener("change", onPrefersChange);
      canvas.removeEventListener("pointerdown", onPointerDown);
      removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointermove", onCanvasPointerMove);
      removeEventListener("pointermove", onWindowPointerMove);
      canvas.removeEventListener("click", onClick);
      centerBtn.removeEventListener("click", onCenter);
      container.innerHTML = "";
    },
  };
}

function escapeAttr(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

/* ============================================================================ DEMO_CASTING
   Scripted demo data -- shapes ported from the approved prototype (its five demo casts), NOT
   measured. meta.scripted=true on every entry; hosts must disclose this (see casting-demo.html's
   footer) rather than presenting these numbers as real receipts. entropyByDepth/leadFlips/cells
   carry the prototype's tuned shapes; effectiveRankByDepth is a NEW field this port adds (the
   prototype instead computed a fake rank from a hardcoded formula -- see casting.mjs's header
   comment / the build report for why that was replaced) -- its curve here is likewise scripted,
   shaped to roughly track each cast's entropy story, and clearly gated behind meta.scripted. */

const READOUT_LAYERS = [0, 6, 12, 18, 24, 28, 31];
const TOTAL_LAYERS = 32;

export const DEMO_CASTING = {
  fact: {
    meta: {
      label: "tallest mountain", scripted: true,
      note: "grounded fact — one lead-change: Everest overtakes K2 at L12. Click “Everest”, "
        + "then the ⑂ K2 ghost to fork the timeline.",
      commitLayer: 4, readoutLayers: READOUT_LAYERS, totalLayers: TOTAL_LAYERS,
    },
    entropyByDepth: [5.9, 5.2, 4.1, 2.6, 1.2, 0.9, 0.7],
    effectiveRankByDepth: [3800, 3100, 2200, 1200, 500, 300, 220],
    turbulence: 1.0, width: 1.0, crystalline: false,
    cells: [{ x: -.42, z: .10, share: .66, label: "Everest \xb7 66%" }, { x: .48, z: -.14, share: .34, label: "K2 \xb7 34%" }],
    leadFlips: [[2, 1, 0]],
    sky: [{ text: "the", weight: .08 }, { text: "tallest", weight: .95 }, { text: "mountain", weight: .85 },
      { text: "on", weight: .06 }, { text: "Earth", weight: .55 }, { text: "is", weight: .10 }],
    tokens: [
      { text: "Mount", entropy: 1.2, sources: [[1, .5], [2, .5]], almosts: [] },
      { text: "Everest", entropy: 0.9, sources: [[1, .45], [2, .35], [4, .20]], almosts: [
        { text: "K2", pull: .31, continuation: [
          { text: "K2", entropy: 1.1 }, { text: ",", entropy: 0.4 }, { text: "in", entropy: 0.9 },
          { text: "the", entropy: 0.5 }, { text: "Karakoram", entropy: 1.3 }, { text: ",", entropy: 0.4 },
          { text: "stands", entropy: 1.0 }, { text: "second", entropy: 0.8 }, { text: ".", entropy: 0.3 },
        ] },
      ] },
      { text: ",", entropy: 0.4, sources: [], almosts: [] },
      { text: "which", entropy: 2.8, sources: [], almosts: [{ text: "and", pull: .38 }] },
      { text: "rises", entropy: 2.2, sources: [[1, 1]], almosts: [{ text: "stands", pull: .36 }] },
      { text: "to", entropy: 1.1, sources: [], almosts: [] },
      { text: "8,849", entropy: 1.6, sources: [[1, .6], [2, .4]], almosts: [
        { text: "8,848", pull: .44, continuation: [
          { text: "8,848", entropy: 1.2 }, { text: "metres", entropy: 0.7 }, { text: "by", entropy: 0.9 },
          { text: "the", entropy: 0.4 }, { text: "older", entropy: 1.1 }, { text: "survey", entropy: 0.9 },
          { text: ".", entropy: 0.3 },
        ] },
      ] },
      { text: "metres", entropy: 0.8, sources: [], almosts: [] },
      { text: "above", entropy: 0.6, sources: [], almosts: [] },
      { text: "sea level.", entropy: 0.5, sources: [[4, 1]], almosts: [] },
    ],
  },
  math: {
    meta: {
      label: "47 \xd7 6", scripted: true,
      note: "equation — no lead-changes at all; 282 leads from L12 and never argues. A silent, closed storm.",
      commitLayer: 2, readoutLayers: READOUT_LAYERS, totalLayers: TOTAL_LAYERS,
    },
    entropyByDepth: [4.8, 3.1, 1.6, 0.8, 0.5, 0.4, 0.3],
    effectiveRankByDepth: [3600, 2400, 900, 300, 150, 100, 80],
    turbulence: 0.35, width: 0.66, crystalline: true,
    cells: [{ x: 0, z: 0, share: 1 }],
    leadFlips: [],
    sky: [{ text: "compute", weight: .12 }, { text: ":", weight: .04 }, { text: "47", weight: .9 },
      { text: "\xd7", weight: .8 }, { text: "6", weight: .9 }, { text: "=", weight: .4 }],
    tokens: [
      { text: "47", entropy: 0.6, sources: [[2, 1]], almosts: [] },
      { text: "\xd7", entropy: 0.3, sources: [[3, 1]], almosts: [] },
      { text: "6", entropy: 0.3, sources: [[4, 1]], almosts: [] },
      { text: "=", entropy: 0.2, sources: [[5, 1]], almosts: [] },
      { text: "282", entropy: 0.5, sources: [[2, .4], [3, .2], [4, .4]], almosts: [{ text: "294", pull: .08 }] },
      { text: ".", entropy: 0.2, sources: [], almosts: [] },
    ],
  },
  poem: {
    meta: {
      label: "a line about the sea", scripted: true,
      note: "poem — the lead changes hands three times among five wisps: a storm arguing with itself "
        + "over a sky it barely reads.",
      commitLayer: 6, readoutLayers: READOUT_LAYERS, totalLayers: TOTAL_LAYERS,
    },
    entropyByDepth: [5.9, 5.7, 5.4, 5.0, 4.4, 3.2, 1.8],
    effectiveRankByDepth: [3900, 3700, 3400, 3000, 2400, 1400, 600],
    turbulence: 2.1, width: 1.35, crystalline: false,
    cells: [{ x: -.65, z: .2, share: .24 }, { x: -.2, z: -.35, share: .22 }, { x: .25, z: .3, share: .20 },
      { x: .6, z: -.1, share: .18 }, { x: 0, z: 0, share: .16 }],
    leadFlips: [[1, 0, 1], [3, 1, 2], [5, 2, 0]],
    sky: [{ text: "write", weight: .18 }, { text: "a", weight: .04 }, { text: "line", weight: .25 },
      { text: "about", weight: .07 }, { text: "the", weight: .05 }, { text: "sea", weight: .85 }],
    tokens: [
      { text: "The", entropy: 2.9, sources: [], almosts: [{ text: "A", pull: .30 }] },
      { text: "sea", entropy: 2.2, sources: [[5, 1]], almosts: [
        { text: "tide", pull: .24, continuation: [
          { text: "tide", entropy: 1.4 }, { text: "keeps", entropy: 1.1 }, { text: "its", entropy: 0.8 },
          { text: "own", entropy: 1.2 }, { text: "ledger", entropy: 1.6 }, { text: "of", entropy: 0.5 },
          { text: "light", entropy: 1.3 }, { text: ".", entropy: 0.4 },
        ] },
      ] },
      { text: "keeps", entropy: 3.1, sources: [], almosts: [{ text: "holds", pull: .30 }] },
      { text: "every", entropy: 2.4, sources: [], almosts: [{ text: "each", pull: .27 }] },
      { text: "color", entropy: 3.3, sources: [], almosts: [{ text: "secret", pull: .24 }] },
      { text: "it", entropy: 1.8, sources: [], almosts: [] },
      { text: "has", entropy: 2.1, sources: [], almosts: [] },
      { text: "drowned", entropy: 3.6, sources: [[5, .6]], almosts: [{ text: "swallowed", pull: .33 }] },
      { text: ".", entropy: 0.9, sources: [], almosts: [] },
    ],
  },
  hop: {
    meta: {
      label: "who acquired the startup?", scripted: true,
      note: "multi-hop shortcut, caught — hover “Vantage”: fed by acquired / Nimbus; “Ortiz” "
        + "and “founded” stay dark. One lead-change as Vantage locks in.",
      commitLayer: 4, readoutLayers: READOUT_LAYERS, totalLayers: TOTAL_LAYERS,
    },
    entropyByDepth: [5.9, 5.4, 4.6, 3.4, 1.8, 1.1, .8],
    effectiveRankByDepth: [3800, 3000, 2000, 900, 350, 220, 180],
    turbulence: .85, width: .95, crystalline: false,
    cells: [{ x: -.15, z: .05, share: .8, label: "Vantage \xb7 80%" }, { x: .5, z: -.2, share: .2 }],
    leadFlips: [[2, 1, 0]],
    sky: [{ text: "Ortiz", weight: .14 }, { text: "founded", weight: .08 }, { text: "Nimbus", weight: .75 },
      { text: "acquired", weight: .9 }, { text: "Vantage", weight: .85 }, { text: "$340M", weight: .35 }],
    tokens: [
      { text: "Vantage", entropy: 1.4, sources: [[3, .55], [4, .45]], almosts: [{ text: "Nimbus", pull: .18 }] },
      { text: "Industries", entropy: 0.7, sources: [[4, 1]], almosts: [] },
      { text: "acquired", entropy: 1.0, sources: [[3, 1]], almosts: [{ text: "bought", pull: .29 }] },
      { text: "Nimbus", entropy: 0.8, sources: [[2, 1]], almosts: [] },
      { text: "Robotics", entropy: 0.6, sources: [[2, 1]], almosts: [] },
      { text: "for", entropy: 0.5, sources: [], almosts: [] },
      { text: "$340M", entropy: 0.9, sources: [[5, 1]], almosts: [] },
      { text: ".", entropy: 0.3, sources: [], almosts: [] },
    ],
  },
  edge: {
    meta: {
      label: "who invented the lightbulb?", scripted: true,
      note: "knife-edge — the lead flips FIVE times between Edison and Swan. Click “Edison”, then "
        + "the ⑂ Swan ghost, and watch history rewrite.",
      commitLayer: 6, readoutLayers: READOUT_LAYERS, totalLayers: TOTAL_LAYERS,
    },
    entropyByDepth: [5.8, 5.5, 5.0, 4.6, 4.0, 3.0, 1.4],
    effectiveRankByDepth: [3850, 3600, 3100, 2600, 2000, 1200, 500],
    turbulence: 1.35, width: 1.1, crystalline: false,
    cells: [{ x: -.5, z: .12, share: .52, label: "Edison \xb7 52%" }, { x: .52, z: -.12, share: .48, label: "Swan \xb7 48%" }],
    leadFlips: [[1, 0, 1], [2, 1, 0], [3, 0, 1], [4, 1, 0], [5, 0, 1]],
    sky: [{ text: "the", weight: .08 }, { text: "inventor", weight: .7 }, { text: "of", weight: .05 },
      { text: "the", weight: .05 }, { text: "lightbulb", weight: .9 }, { text: "was", weight: .12 }],
    tokens: [
      { text: "Thomas", entropy: 2.9, sources: [[1, .5], [4, .5]], almosts: [{ text: "Joseph", pull: .45 }] },
      { text: "Edison", entropy: 2.6, sources: [[1, .5], [4, .5]], almosts: [
        { text: "Swan", pull: .48, continuation: [
          { text: "Swan", entropy: 1.3 }, { text: ",", entropy: 0.5 }, { text: "who", entropy: 0.9 },
          { text: "patented", entropy: 1.4 }, { text: "his", entropy: 0.6 }, { text: "lamp", entropy: 0.8 },
          { text: "in", entropy: 0.4 }, { text: "1878", entropy: 1.0 }, { text: ".", entropy: 0.3 },
        ] },
      ] },
      { text: ",", entropy: 0.6, sources: [], almosts: [] },
      { text: "though", entropy: 2.2, sources: [], almosts: [{ text: "but", pull: .31 }] },
      { text: "Joseph", entropy: 1.9, sources: [[4, 1]], almosts: [] },
      { text: "Swan", entropy: 1.7, sources: [[4, 1]], almosts: [] },
      { text: "preceded", entropy: 2.1, sources: [], almosts: [] },
      { text: "him", entropy: 0.9, sources: [], almosts: [] },
      { text: ".", entropy: 0.4, sources: [], almosts: [] },
    ],
  },
};
