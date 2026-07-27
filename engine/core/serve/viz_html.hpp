// viz_html.hpp — the real-time denoise visualization, served by clozn-server at GET /. A
// self-contained page (no build step, no external deps) that renders the §5.1 event spine live.
// ONE editable panel is the whole UI:
//   - highlight a span -> POST /v1/revise: re-mask + re-predict it in place, with `grow` extra slots
//     of length wiggle (a K-token span can become short..K+grow tokens).
//   - put the caret between words (or click the floating +) -> POST /v1/infill: insert tokens there.
//   - caret at the very end -> POST /v1/completions: continue.
// Results render back into the same panel (colored by confidence), so you keep highlighting/revising.
// Pure consumer of the event stream (invariant 2); cells keyed by absolute board position.
//
// Visual identity is borrowed from planet-maiko (maiko.os): night-indigo wallpaper, candy
// accents, chunky beveled chrome, hard edges, a saturn glyph, and Atkinson Hyperlegible /
// Bricolage Grotesque / Inconsolata. The whole page is one OS window. Confidence still drives
// token color (peach=low -> mint=high), the masked slots are sunken form fields.
#pragma once

namespace clozn {

inline const char* VIZ_HTML = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><title>clozn &mdash; watch it denoise</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Inconsolata:wght@400;500;600;700&display=swap">
<style>
  :root{
    --bg:#1A1F4A; --bg-2:#14173a; --panel:#1c1f43; --field:#0a0c26;
    --fg:#F4F0E8; --dim:#B8B3D6; --muted:#8784b3;
    --pink:#FF8FB3; --teal:#6FD6C9; --lemon:#F5D77A; --lavender:#B59DD8; --mint:#8FE0B0; --peach:#FFB38A;
    --c-punct:#F5D77A; --c-number:#6FD6C9; --c-word:#FF8FB3; --c-code:#8FE0B0; --c-question:#FFB38A;  /* Tier-2 concept-probe colors */
    --ink:#211a33; --face:#d8d9ec; --accent:var(--lavender);
    --font:'Atkinson Hyperlegible',-apple-system,'Segoe UI',sans-serif;
    --display:'Bricolage Grotesque',var(--font);
    --mono:'Inconsolata',ui-monospace,Consolas,monospace;
    /* Chunky 3D bevels — the whole point of the era. */
    --raise:inset 2px 2px 0 rgba(255,255,255,.55),inset -2px -2px 0 rgba(0,0,0,.46),
            inset 3px 3px 0 rgba(255,255,255,.16),inset -3px -3px 0 rgba(0,0,0,.24);
    --sink:inset 2px 2px 0 rgba(0,0,0,.5),inset -2px -2px 0 rgba(255,255,255,.28);
  }
  *{box-sizing:border-box;}
  body{
    margin:0; color:var(--fg); font:16px/1.6 var(--font); min-height:100vh; padding:26px 18px 44px;
    -webkit-font-smoothing:antialiased;
    background:
      repeating-linear-gradient(45deg,rgba(255,255,255,.022) 0 2px,transparent 2px 9px),
      radial-gradient(900px 600px at 78% -8%,rgba(111,214,201,.18),transparent 60%),
      radial-gradient(820px 600px at 8% 8%,rgba(255,143,179,.15),transparent 55%),
      radial-gradient(720px 520px at 60% 110%,rgba(245,215,122,.10),transparent 60%),
      linear-gradient(180deg,var(--bg) 0%,var(--bg-2) 100%);
    background-attachment:fixed;
  }
  /* The page is one maiko.os window: beveled frame + flat drop shadow. */
  .win{ max-width:980px; margin:0 auto; background:#23274f; padding:4px;
        box-shadow:var(--raise),7px 7px 0 rgba(0,0,0,.45); }
  .win-bar{ display:flex; align-items:center; gap:8px; padding:6px 6px 6px 0;
            font-family:var(--display); font-weight:700; box-shadow:var(--raise);
            background:linear-gradient(180deg,var(--accent),color-mix(in srgb,var(--accent) 58%,#160f2e)); }
  .planet{ flex:none; width:26px; height:22px; margin-left:4px; box-shadow:var(--raise);
           background:rgba(255,255,255,.5) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='6' fill='%23FF8FB3'/%3E%3Cellipse cx='12' cy='12' rx='10.5' ry='3.6' fill='none' stroke='%23F5D77A' stroke-width='1.7' transform='rotate(-22 12 12)'/%3E%3C/svg%3E") center / 15px no-repeat; }
  .win-title{ flex:1; font-size:1rem; color:var(--ink); letter-spacing:.01em;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
              text-shadow:1px 1px 0 rgba(255,255,255,.3); }
  .win-dots{ display:flex; gap:4px; flex:none; }
  .win-dots i{ width:21px; height:19px; display:flex; align-items:center; justify-content:center;
               background:var(--face); box-shadow:var(--raise); color:var(--ink);
               font-family:var(--mono); font-size:.8rem; font-style:normal; line-height:1; }
  .win-dots i:first-child::after{ content:"_"; transform:translateY(-3px); }
  .win-dots i:last-child::after{ content:"\25A2"; }
  .win-x{ flex:none; width:21px; height:19px; margin-right:4px; display:flex; align-items:center;
          justify-content:center; background:var(--face); box-shadow:var(--raise);
          color:var(--ink); font-size:.78rem; user-select:none; }
  .win-body{ background:var(--panel); box-shadow:var(--sink);
             border-top:3px solid color-mix(in srgb,var(--accent) 70%,#000); padding-bottom:4px; }

  header{ display:flex; gap:12px 14px; align-items:flex-end; flex-wrap:wrap;
          padding:16px 18px 15px; border-bottom:2px dotted rgba(255,255,255,.14); }
  label{ color:var(--dim); font-family:var(--mono); font-size:.74rem; letter-spacing:.07em;
         text-transform:lowercase; display:flex; flex-direction:column; gap:4px; }
  label.chk{ flex-direction:row; align-items:center; gap:6px; text-transform:none; }
  label.chk input{ width:auto; }
  input[type=number],select{ background:var(--field); color:var(--fg); border:none; box-shadow:var(--sink);
         border-radius:0; padding:7px 6px; width:62px; font:600 .95rem var(--mono); }
  select{ width:auto; padding:7px 8px; }
  input:focus,select:focus{ outline:1px dotted var(--accent); outline-offset:2px; }
  button{ background:var(--pink); color:var(--ink); border:none; box-shadow:var(--raise); border-radius:0;
          padding:10px 20px; font:700 1rem var(--font); cursor:pointer; }
  button:active{ box-shadow:var(--sink); transform:translateY(1px); }
  button:disabled{ opacity:.55; cursor:default; }

  .hint{ padding:14px 18px 0; color:var(--dim); font-size:.92rem; line-height:1.55; }
  .hint b{ color:var(--fg); }
  .stats{ margin:13px 18px 0; padding:8px 12px; min-height:20px; color:var(--dim);
          background:var(--field); box-shadow:var(--sink); font-family:var(--mono); font-size:.86rem; }
  .stats b{ color:var(--lemon); }

  #board{ margin:14px 18px 0; padding:16px 18px; min-height:162px; cursor:text;
          white-space:pre-wrap; word-break:break-word; font:16px/1.95 var(--mono); color:var(--fg);
          background:var(--field); box-shadow:var(--sink); border-radius:0; outline:none; }
  #board:focus{ box-shadow:var(--sink),0 0 0 2px color-mix(in srgb,var(--accent) 45%,transparent); }
  #board ::selection{ background:rgba(181,157,216,.42); color:#fff; }
  #board::selection{ background:rgba(181,157,216,.42); color:#fff; }

  .tok{ border-radius:0; transition:color .3s ease; }
  .fixed{ color:var(--fg); }            /* typed / unrevised context */
  .gen{ }                               /* committed text — color set inline by confidence */
  .masked{ display:inline-block; min-width:3ch; height:1.55em; vertical-align:text-bottom;
           margin:0 4px; background:#161a3e; box-shadow:var(--sink); border-radius:0; }  /* a slot being filled */
  .just{ outline:2px solid var(--lavender); outline-offset:1px; }   /* committed THIS pass */

  #plus{ position:absolute; display:none; z-index:20; transform:translate(-50%,-118%);
         background:var(--lemon); color:var(--ink); font:700 .74rem var(--mono);
         padding:5px 9px; box-shadow:var(--raise); cursor:pointer; white-space:nowrap; border-radius:0; }

  .legend{ padding:16px 18px 18px; color:var(--dim); font-size:.85rem; line-height:1.75; }
  .legend b{ color:var(--fg); }
  .legend .tok.masked{ min-width:2.4ch; }
  .ckey{ display:inline-block; width:.85em; height:.85em; vertical-align:-1px; box-shadow:var(--raise); margin:0 3px 0 9px; }
</style></head>
)HTML"
R"HTML(<body>
<div class="win">
  <div class="win-bar">
    <span class="planet"></span>
    <span class="win-title">clozn.exe &mdash; <span id="modetag">watch it denoise</span></span>
    <span class="win-dots"><i></i><i></i></span>
    <span class="win-x">&times;</span>
  </div>
  <div class="win-body">
    <header>
      <label>tokens<input id="maxnew" type="number" value="16"></label>
      <label>grow<input id="grow" type="number" value="4" min="0"></label>
      <label>steps<input id="steps" type="number" value="16"></label>
      <label>speed<select id="speed"><option value="0">instant</option><option value="1" selected>1&times;</option><option value="0.75">0.75&times;</option><option value="0.5">0.5&times;</option><option value="0.25">0.25&times;</option></select></label>
      <label>temp<input id="temp" type="number" value="0" step="0.1" min="0"></label>
      <label>rep pen<input id="rep" type="number" value="1" step="0.05" min="1"></label>
      <label class="chk"><input id="remask" type="checkbox">remask</label>
      <label>&#964;<input id="tau" type="number" value="0.4" step="0.05" min="0" max="1"></label>
      <label class="chk"><input id="feat" type="checkbox" checked>concepts</label>
      <label>steer<select id="steerc"><option value="">off</option><option value="punct">punct</option><option value="number">number</option><option value="function">function</option><option value="content">content</option><option value="code">code</option><option value="question">question</option></select></label>
      <label>strength<input id="steerk" type="number" value="0" step="1" min="0" title="control-vector coefficient; pushes the concept into the residual stream. High values garble — steering is slippery."></label>
      <button id="go">generate</button>
    </header>
    <div class="hint" id="hint"><b>Highlight</b> a span &rarr; <b>revise selection</b> rewrites it (up to <b>grow</b> extra tokens of length wiggle). Put the cursor between words (or use the floating <b>+</b>) &rarr; <b>generate</b> inserts <b>tokens</b> new tokens there; cursor at the very end continues.</div>
    <div class="stats" id="stats">highlight to revise &middot; click + generate to insert &middot; cursor at the end to continue.</div>
    <div id="board" contenteditable spellcheck="false"></div>
    <div class="legend">
      text color = the model's confidence when it committed each token:
      <span style="color:hsl(8,70%,68%)">low</span>
      <span style="color:hsl(79,70%,68%)">mid</span>
      <span style="color:hsl(150,70%,68%)">high</span>
      &nbsp;&middot;&nbsp; <span class="tok masked"></span> = a slot being filled &middot; <b>remask</b> = let the model re-mask low-confidence tokens (below &#964;) and reconsider, mid-generation.
      <br><b>concepts</b> (Tier-2 probe reading each slot's mid-layer hidden state, live):
      <span class="ckey" style="background:var(--c-punct)"></span>punct
      <span class="ckey" style="background:var(--c-number)"></span>number
      <span class="ckey" style="background:var(--lavender)"></span>function
      <span class="ckey" style="background:var(--c-word)"></span>content
      <span class="ckey" style="background:var(--c-code)"></span>code
      <span class="ckey" style="background:var(--c-question)"></span>question
    </div>
  </div>
</div>
<div id="plus">+ generate</div>
)HTML"
R"HTML(<script>
const $ = id => document.getElementById(id);
const board = $('board'), stats = $('stats'), plus = $('plus');
let cellByPos = {}, justCells = [], queue = [], streamDone = false, lastBoardSel = null, plusOffset = 0, plusHover = false;
let MODE = 'diffusion';   // set from /health on load: 'diffusion' (dLLM) or 'autoregressive' (AR LLM)
const sleep = ms => new Promise(r => setTimeout(r, ms));
const confColor = c => { const t=Math.max(0,Math.min(1,c)); return `hsl(${8+t*142},70%,68%)`; };   // peach (low) -> mint (high)
function tok(text, cls){ const s=document.createElement('span'); s.className='tok '+cls; s.textContent=text; board.appendChild(s); return s; }

// ---- rendering (cells keyed by absolute board position) ----
function reset(prefixPieces, nTokens, nMasked, suffixPieces){      // completion / infill layout
  board.innerHTML=''; cellByPos={};
  (prefixPieces||[]).forEach(p=>tok(p,'fixed'));
  for(let i=0;i<nMasked;i++) cellByPos[nTokens+i] = tok('','masked');
  (suffixPieces||[]).forEach(p=>tok(p,'fixed'));
}
function resetFromLayout(layout){                                  // revise layout (whole board)
  board.innerHTML=''; cellByPos={};
  for(const it of layout){ if(it.masked) cellByPos[it.pos]=tok('','masked'); else tok(it.piece||'','fixed'); }
}
function commit(it){
  const c=cellByPos[it.pos]; if(!c) return null;
  c.textContent=(it.piece!==undefined?it.piece:('#'+it.id))||'';   // REAL text: spaces/newlines kept; EOS/pad -> empty
  c.className='tok gen'; c.style.color=confColor(it.conf); c.style.boxShadow=''; c.title=`conf ${it.conf.toFixed(3)}`;
  return c;
}
// ---- Tier-2 white-box: concept-probe scores per slot (step_features) ----
const CONCEPT_COLORS={punct:'var(--c-punct)',number:'var(--c-number)',function:'var(--lavender)',content:'var(--c-word)',code:'var(--c-code)',question:'var(--c-question)'};
const conceptColor=n=>CONCEPT_COLORS[n]||'var(--lavender)';
function renderLens(ev){               // logit-lens: tooltip = top-k token candidates per masked slot
  const K=ev.k||0; if(!K) return;
  for(let i=0;i<ev.positions.length;i++){
    const c=cellByPos[ev.positions[i]]; if(!c) continue;
    const cand=[];
    for(let j=0;j<K;j++){ cand.push((ev.pieces[i*K+j]||'∅').replace(/\s/g,'·')+' '+Math.round(ev.probs[i*K+j]*100)+'%'); }
    c.title='considering: '+cand.join('   ')+(c.dataset.concepts?'\n'+c.dataset.concepts:'');
  }
}
function renderFeatures(ev){            // underline each slot by its strongest concept this pass
  const K=ev.features.length; if(!K) return;
  for(let i=0;i<ev.positions.length;i++){
    const c=cellByPos[ev.positions[i]]; if(!c) continue;
    let best=0,bv=ev.scores[i*K];
    for(let k=1;k<K;k++){ const v=ev.scores[i*K+k]; if(v>bv){bv=v;best=k;} }
    const masked=c.classList.contains('masked');
    c.style.boxShadow = bv>1.5 ? (masked?'var(--sink),':'')+'inset 0 -0.32em 0 '+conceptColor(ev.features[best]) : '';
    c.dataset.concepts=ev.features.map((f,k)=>`${f} ${(+ev.scores[i*K+k]).toFixed(2)}`).join(' · '); c.title=c.dataset.concepts;
  }
}
function handle(ev){
  if(ev.type==='gen_started'){ if(ev.layout) resetFromLayout(ev.layout); else reset(ev.prompt_pieces, ev.prompt_tokens, ev.max_new, ev.suffix_pieces); }
  else if(ev.type==='tokens_committed'){ justCells.forEach(c=>c.classList.remove('just')); justCells=[];
    for(const it of ev.items){ const c=commit(it); if(c){ c.classList.add('just'); justCells.push(c); } } }
  else if(ev.type==='tokens_revised'){ for(const it of ev.items){ const c=cellByPos[it.pos]; if(c){ c.textContent=''; c.className='tok masked'; c.style.color=''; c.title=`reconsidered (was conf ${it.conf.toFixed(3)})`; } } }
  else if(ev.type==='step_stats'){ stats.innerHTML=`pass <b>${ev.step}</b> &middot; committed <b>${ev.committed}</b> this pass &middot; <b>${ev.remaining}</b> blanks left`; }
  else if(ev.type==='step_features'){ renderFeatures(ev); }
  else if(ev.type==='step_lens'){ renderLens(ev); }
  else if(ev.type==='gen_finished'){ justCells.forEach(c=>c.classList.remove('just')); justCells=[];
    const spt=ev.new_tokens?(ev.steps_total/ev.new_tokens).toFixed(2):'0';
    stats.innerHTML=`done &middot; <b>${ev.new_tokens}</b> tokens in <b>${ev.steps_total}</b> passes (<b>${spt}</b> steps/token) &middot; <b>${ev.tok_per_s.toFixed(1)}</b> tok/s`; }
}
async function player(){
  while(!streamDone || queue.length){
    if(!queue.length){ await sleep(10); continue; }
    const ev=queue.shift(); handle(ev);
    if(ev.type==='step_stats'){ const sp=+$('speed').value; if(sp>0) await sleep(300/sp); }   // playback speed
  }
  $('go').disabled=false; board.contentEditable='true'; updateLabel();
}
async function pump(resp){
  const reader=resp.body.getReader(), dec=new TextDecoder(); let buf='';
  while(true){ const {value,done}=await reader.read(); if(done) break; buf+=dec.decode(value,{stream:true});
    let i; while((i=buf.indexOf('\n\n'))>=0){ const fr=buf.slice(0,i); buf=buf.slice(i+2);
      if(fr.startsWith('data: ')){ const d=fr.slice(6); if(d==='[DONE]') continue; try{ const ev=JSON.parse(d); if(ev.type) queue.push(ev); }catch(e){} } } }
}

// ---- board text + caret/selection as char offsets (Range.toString keeps text & offsets consistent) ----
const utf8=new TextEncoder(); const blen=s=>utf8.encode(s).length;   // -> UTF-8 byte length (server works in bytes)
function boardText(){ const r=document.createRange(); r.selectNodeContents(board); return r.toString(); }
function offAt(node,off){ const r=document.createRange(); r.selectNodeContents(board); r.setEnd(node,off); return r.toString().length; }
function boardSel(){
  const s=window.getSelection(); if(!s||!s.rangeCount) return null;
  const rng=s.getRangeAt(0);
  if(rng.collapsed||!board.contains(rng.startContainer)||!board.contains(rng.endContainer)) return null;
  const a=offAt(rng.startContainer,rng.startOffset);
  return { s:a, e:a+rng.toString().length };
}
function caretOffset(){
  const s=window.getSelection();
  if(!s||!s.rangeCount||!board.contains(s.anchorNode)) return boardText().length;
  const rng=s.getRangeAt(0); return offAt(rng.startContainer,rng.startOffset);
}

// ---- the three actions, all rendering into the board ----
const knobs=extra=>Object.assign({ steps:+$('steps').value, stream:true, temperature:+$('temp').value, rep_penalty:+$('rep').value, seed:Math.floor(Math.random()*1e9) }, extra);
async function stream(endpoint, body){
  $('go').disabled=true; queue=[]; streamDone=false; justCells=[]; lastBoardSel=null; plus.style.display='none'; board.contentEditable='false';
  player();
  try{ await pump(await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})); }
  catch(e){ stats.textContent='error: '+e; }
  streamDone=true;
}
function reviseSpan(src,sel){
  stats.textContent='revising selection...';
  const spans=[{ start:blen(src.slice(0,sel.s)), end:blen(src.slice(0,sel.e)) }];
  stream('/v1/revise', knobs({ text:src, spans, revise:false, grow:+$('grow').value }));   // grow = length wiggle
}
function generateAt(src,caret){
  const before=src.slice(0,caret), after=src.slice(caret);
  const remask=$('remask').checked, tau=+$('tau').value;
  if(before.trim()==='' && after.trim()===''){ stats.textContent='type something first.'; return; }
  const steerc=$('steerc').value, steerk=+$('steerk').value;
  const steer=(steerc && steerk>0)?{concept:steerc,coef:steerk}:undefined;   // JSON.stringify drops it when off
  if(after.trim()===''){ stats.textContent='continuing...';
    stream('/v1/completions', knobs({ prompt:before, max_tokens:+$('maxnew').value, block_len:0, cache:'off', revise:remask, tau_revise:tau, features:$('feat').checked, steer })); }
  else { stats.textContent='generating between...';
    stream('/v1/infill', knobs({ prefix:before, suffix:after, gap:+$('maxnew').value, revise:remask, tau_revise:tau, features:$('feat').checked, steer })); }
}
function go(){
  // AR models only continue (no in-place revise / fill-in-the-middle): always complete from the
  // whole board text, left to right — the structurally diffusion-only actions are disabled below.
  if(MODE==='autoregressive'){ const src=boardText(); generateAt(src, src.length); return; }
  const src=boardText(), sel=boardSel()||lastBoardSel;
  if(sel && sel.e>sel.s) reviseSpan(src,sel); else generateAt(src, caretOffset());
}
function updateLabel(){
  if(MODE==='autoregressive'){ $('go').textContent='generate'; return; }
  const sel=boardSel();
  if(sel){ $('go').textContent='revise selection'; return; }
  const src=boardText(), c=caretOffset();
  $('go').textContent = src.slice(c).trim()==='' ? 'generate' : 'generate here';
}

// ---- typed/pasted text = default color (bare text node, not a span's inline confidence color) ----
function insertPlain(text){ const s=window.getSelection(); if(!s.rangeCount) return; const r=s.getRangeAt(0); r.deleteContents(); const n=document.createTextNode(text); r.insertNode(n); r.setStartAfter(n); r.collapse(true); s.removeAllRanges(); s.addRange(r); }
board.addEventListener('beforeinput', e=>{ if(e.inputType==='insertText' && e.data!=null && !e.isComposing){ e.preventDefault(); insertPlain(e.data); } });
board.addEventListener('paste', e=>{ e.preventDefault(); insertPlain((e.clipboardData||window.clipboardData).getData('text')); });

// ---- floating + : hover the gap between words, click to generate there ----
function caretRangeAt(x,y){ if(document.caretRangeFromPoint) return document.caretRangeFromPoint(x,y);
  if(document.caretPositionFromPoint){ const p=document.caretPositionFromPoint(x,y); if(p){ const r=document.createRange(); r.setStart(p.offsetNode,p.offset); r.collapse(true); return r; } } return null; }
board.addEventListener('mousemove', e=>{
  if(MODE==='autoregressive'){ plus.style.display='none'; return; }   // no insert-in-the-middle for AR
  if(board.contentEditable!=='true'){ plus.style.display='none'; return; }
  const sel=window.getSelection(); if(sel && sel.toString()){ plus.style.display='none'; return; }   // selecting -> revise, no +
  const r=caretRangeAt(e.clientX,e.clientY); if(!r||!board.contains(r.startContainer)){ plus.style.display='none'; return; }
  const rect=r.getBoundingClientRect();
  plus.style.left=(rect.left+window.scrollX)+'px'; plus.style.top=(rect.top+window.scrollY)+'px';
  plus.style.display='block'; plusOffset=offAt(r.startContainer,r.startOffset);
});
board.addEventListener('mouseleave', ()=>{ if(!plusHover) plus.style.display='none'; });
plus.addEventListener('mouseenter', ()=>{ plusHover=true; });
plus.addEventListener('mouseleave', ()=>{ plusHover=false; plus.style.display='none'; });
plus.addEventListener('mousedown', e=>e.preventDefault());
plus.onclick=()=>{ plus.style.display='none'; generateAt(boardText(), plusOffset); };

// ---- wiring ----
$('go').onclick=go;
$('go').addEventListener('mousedown', e=>e.preventDefault());   // keep the board selection alive on click (a div selection clears on blur)
board.addEventListener('mouseup', ()=>{ const x=boardSel(); if(x) lastBoardSel=x; updateLabel(); });
board.addEventListener('keyup', updateLabel);
board.addEventListener('focus', updateLabel);

// ---- mode: diffusion (denoise board) vs autoregressive (left-to-right token stream) ----
// Same event spine + white-box reads (concept underline, logit-lens, steer) either way; AR just
// fills the slots strictly left to right and hides the diffusion-only actions (revise / infill / remask).
function applyMode(){
  const ar = MODE==='autoregressive';
  $('modetag').textContent = ar ? 'watch it think (autoregressive)' : 'watch it denoise';
  for(const id of ['grow','steps','remask','tau']){ const lab=$(id)&&$(id).closest('label'); if(lab) lab.style.display = ar?'none':''; }
  if(ar){
    $('hint').innerHTML = '<b>Autoregressive</b> mode: type a prompt, hit <b>generate</b>, watch it continue left&#8209;to&#8209;right. The white&#8209;box reads (concept underline, logit&#8209;lens on hover) and <b>steer</b> work the same; in&#8209;place revise and fill&#8209;the&#8209;middle are diffusion&#8209;only.';
    $('stats').textContent = 'type a prompt · generate to continue · hover a token for the logit-lens.';
  }
  updateLabel();
}
fetch('/health').then(r=>r.json()).then(h=>{ MODE = (h&&h.mode==='autoregressive')?'autoregressive':'diffusion'; applyMode(); }).catch(()=>{});

board.textContent = "On a rainy day, I like to sit by the window with a hot cup of tea.";
updateLabel();
</script>
</body></html>
)HTML";

}  // namespace clozn
