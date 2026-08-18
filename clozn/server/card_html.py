"""The shareable receipt card: `render_card(bundle) -> str` turns ONE export bundle
(`clozn.receipts.bundle.build(run, explain=...)` — the exact object GET /runs/<id>/export returns,
optionally + a `lineage` key from `clozn.runs.store.lineage`) into ONE self-contained HTML document a
person can save or post anywhere. Pure function: dict in, string out — zero model calls, zero file IO,
zero network. The card is a RENDERING of what the bundle already carries, never a new computation: a
receipt that was never computed renders as its honest absence, not as a blank or a guess.

Visual language: the current Clozn Studio instrument system. The default Halo theme uses an
opal/mother-of-pearl field; the optional Cathedral theme uses freshwater-black-pearl surfaces. Both
use square geometry, compact evidence hierarchy, and the same cyan/mint/violet/pink/peach signal
palette. The values are inlined because the receipt must remain completely self-contained.

Injection-proof by construction: every string that originated outside this module (prompt, reply,
tokens, card texts, ids, lens labels) passes through html.escape before it touches the document, and
the document ships no <script> at all — a reply containing `<script>` renders inert. Self-contained by
construction: no src/href attributes, no webfonts, no images, no JS; system font stacks only.
"""
from __future__ import annotations

import html
import math


# Cap the phosphor token stream so a pathological trace can't blow the ~150KB budget. Typical replies
# are <=256 tokens; the cap only ever bites on something abnormal, and it says so on the card.
MAX_TOKENS = 4000
MAX_INFLUENCE_CONTEXT_SPANS = 8
MAX_INFLUENCE_CONTEXT_CHARS = 1600
MAX_INFLUENCE_ANSWER_SPANS = 256
MAX_INFLUENCE_ANSWER_CHARS = 16000
LOW_CONF = 0.5   # matches clozn.receipts.explain.LOW_CONF — ONE "unsure" convention

_ABSENT_RECEIPTS = "no receipts computed for this run — receipts are measured on demand, never assumed"
_ABSENT_INFLUENCE = ("no context-answer influence map computed for this run — "
                     "the map is measured on demand, never inferred")
_ABSENT_LENS = ("no lens readout recorded on this run — the lens reads on demand from the engine "
                "substrate, and none was captured here")
_FOOTER_LABEL = "run receipt"
# Mirrors clozn.server.app._JLENS_NOTE — the shipped, unskippable J-lens honesty caption.
_JLENS_CAPTION = ("fitted linear Jacobian lens, transferred to this GGUF; a per-token 'disposed to "
                  "say' read, NOT the model's literal thought — a linear lens always emits something.")


def _esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _num(v, nd: int = 3, signed: bool = False) -> str:
    f = _float(v)
    if f is None:
        return ""
    return f"{f:+.{nd}f}" if signed else f"{f:.{nd}f}"


# ------------------------------------------------------------------------------------ inline stylesheet
# Studio Next values are inlined because receipts cannot depend on the running Studio.
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{color-scheme:light dark}
html,body{min-height:100%;background:#e9edf5}
body{font-family:ui-monospace,"SFMono-Regular","Cascadia Mono",Consolas,monospace;
 font-size:12px;line-height:1.5;-webkit-font-smoothing:antialiased}
.receipt-shell{
 --page:#e9edf5;--panel:rgba(248,249,252,.88);--panel-strong:rgba(255,255,255,.94);
 --panel-soft:rgba(237,241,247,.74);--ink:#181c28;--ink-soft:#50596d;--ink-faint:#7a8499;
 --line:rgba(72,82,108,.23);--line-soft:rgba(72,82,108,.13);--shadow:rgba(73,82,111,.18);
 --signal-cyan:#5edbe5;--signal-mint:#78e4cc;--signal-violet:#a78cf7;
 --signal-pink:#ee91cf;--signal-peach:#f1b28e;--danger:#b94e43;
 --captured:#197b71;--derived:#276fa6;--response-ink:#202638;--response-label:#566176;
 --response-bg:
  radial-gradient(90% 120% at 8% 20%,rgba(94,219,229,.23),transparent 62%),
  radial-gradient(80% 110% at 76% 15%,rgba(167,140,247,.25),transparent 62%),
  radial-gradient(70% 100% at 96% 85%,rgba(238,145,207,.21),transparent 66%),
  linear-gradient(112deg,rgba(252,253,255,.96),rgba(241,243,251,.93));
 --support-bg:rgba(120,228,204,.28);--support-ink:#155e55;
 --suppress-bg:rgba(167,140,247,.24);--suppress-ink:#52447f;
 --neutral-bg:rgba(122,132,153,.17);--neutral-ink:#50596d;
 min-height:100vh;padding:20px 14px 36px;color:var(--ink);
 background:
  radial-gradient(720px 420px at 4% 0%,rgba(94,219,229,.19),transparent 67%),
  radial-gradient(760px 440px at 55% -8%,rgba(167,140,247,.15),transparent 68%),
  radial-gradient(680px 430px at 98% 8%,rgba(238,145,207,.15),transparent 67%),
  linear-gradient(150deg,#f5f7fb 0%,var(--page) 52%,#eef0f7 100%)}
.theme-toggle-input{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);
 clip-path:inset(50%);white-space:nowrap}
.receipt-shell:has(.theme-toggle-input:checked){
 --page:#080a11;--panel:rgba(14,17,26,.92);--panel-strong:rgba(18,21,32,.96);
 --panel-soft:rgba(9,12,20,.76);--ink:#edf1f7;--ink-soft:#aeb7c8;--ink-faint:#737e94;
 --line:rgba(175,188,219,.23);--line-soft:rgba(175,188,219,.12);--shadow:rgba(0,0,0,.45);
 --signal-cyan:#6de6ee;--signal-mint:#85ebd2;--signal-violet:#af99ff;
 --signal-pink:#f19bd7;--signal-peach:#f5b995;--danger:#ff8c80;
 --captured:#85ebd2;--derived:#79cfea;--response-ink:#f0f3f9;--response-label:#97a2b8;
 --response-bg:
  radial-gradient(90% 130% at 8% 22%,rgba(94,219,229,.17),transparent 60%),
  radial-gradient(75% 120% at 64% 0%,rgba(167,140,247,.20),transparent 62%),
  radial-gradient(70% 100% at 96% 90%,rgba(238,145,207,.14),transparent 65%),
  linear-gradient(118deg,#111521,#090c14);
 --support-bg:rgba(120,228,204,.23);--support-ink:#b8f5e6;
 --suppress-bg:rgba(167,140,247,.25);--suppress-ink:#ded6ff;
 --neutral-bg:rgba(174,183,200,.13);--neutral-ink:#c3cada;
 background:
  radial-gradient(680px 430px at 4% 0%,rgba(94,219,229,.09),transparent 67%),
  radial-gradient(760px 450px at 56% -8%,rgba(167,140,247,.10),transparent 68%),
  radial-gradient(680px 430px at 100% 10%,rgba(238,145,207,.08),transparent 67%),
  linear-gradient(150deg,#10131d 0%,var(--page) 56%,#0b0e17 100%)}
.card{width:min(1240px,100%);margin:0 auto;display:grid;
 grid-template-columns:minmax(0,1.55fr) minmax(320px,.7fr);gap:1px;
 border:1px solid var(--line);background:var(--line);box-shadow:0 22px 58px var(--shadow)}
.mod{position:relative;min-width:0;background:var(--panel);border:0}
.masthead,.exchange,.influence,.lineage,.receipt-footer{grid-column:1/-1}
.receipts{grid-column:1}.lens{grid-column:2}
.mod-h{min-height:37px;display:flex;align-items:center;gap:9px;padding:9px 13px;
 border-bottom:1px solid var(--line-soft);flex-wrap:wrap}
.cap{font-family:system-ui,sans-serif;font-weight:650;letter-spacing:.18em;
 text-transform:uppercase;font-size:9px;color:var(--ink-soft)}
.led{width:5px;height:5px;background:var(--signal-mint);
 box-shadow:0 0 9px var(--signal-mint);flex:none}
.led.blue{background:var(--signal-cyan);box-shadow:0 0 9px var(--signal-cyan)}
.led.lilac{background:var(--signal-violet);box-shadow:0 0 9px var(--signal-violet)}
.tag{font-family:system-ui,sans-serif;font-size:7.5px;letter-spacing:.11em;text-transform:uppercase;
 padding:2px 5px;white-space:nowrap;border:1px solid var(--line)}
.tag.cap-t{color:var(--captured);background:rgba(120,228,204,.10);border-color:currentColor}
.tag.der-t{color:var(--derived);background:rgba(94,219,229,.09);border-color:currentColor}
.tag.warn-t{color:var(--danger);background:rgba(241,178,142,.09);border-color:currentColor}
.mod-b{padding:12px 13px 14px}
.wordmark{font-family:system-ui,sans-serif;font-weight:760;font-size:17px;letter-spacing:.14em;
 color:var(--ink)}
.wordmark b{color:var(--signal-cyan)}
.mast-sub{font-family:system-ui,sans-serif;font-size:8px;letter-spacing:.23em;
 text-transform:uppercase;color:var(--ink-faint)}
.theme-toggle{margin-left:auto;display:flex;align-items:center;gap:7px;padding:4px 6px;
 border:1px solid var(--line);color:var(--ink-faint);cursor:pointer;
 font-family:system-ui,sans-serif;font-size:7px;letter-spacing:.14em;text-transform:uppercase}
.theme-toggle b{font-size:8px;color:var(--ink-soft);font-weight:650}
.theme-toggle b::after{content:"halo"}
.theme-toggle:hover{border-color:var(--signal-cyan);color:var(--ink)}
.theme-toggle-input:focus-visible~.card .theme-toggle{outline:2px solid var(--signal-cyan);
 outline-offset:2px}
.receipt-shell:has(.theme-toggle-input:checked) .theme-toggle b::after{content:"cathedral"}
.meta{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));border-top:1px solid var(--line-soft);
 border-bottom:1px solid var(--line-soft)}
.meta>div{min-width:0;padding:8px 10px;border-right:1px solid var(--line-soft)}
.meta>div:last-child{border-right:0}
.meta .k{font-family:system-ui,sans-serif;font-size:7px;letter-spacing:.15em;text-transform:uppercase;
 color:var(--ink-faint);display:block;margin-bottom:2px}
.meta .v{font-size:9.5px;color:var(--ink);word-break:break-all}
.meta .v.warn{color:var(--danger);font-weight:650}
.legend{display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:8px 10px;
 font-family:system-ui,sans-serif;font-size:8px;color:var(--ink-faint)}
.legend .legend-copy{margin-right:7px;letter-spacing:.05em;text-transform:uppercase}
.well{border-left:2px solid var(--signal-cyan);padding:9px 11px;background:var(--panel-soft);
 white-space:pre-wrap;word-break:break-word;color:var(--ink-soft)}
.turnnote{font-size:8px;color:var(--ink-faint);padding:4px 2px 0}
.crt{position:relative;margin:10px 0 0;padding:30px 14px 13px;overflow:hidden;
 border:1px solid var(--line);background:var(--response-bg)}
.crt::after{content:"";position:absolute;inset:0;pointer-events:none;opacity:.16;
 background:linear-gradient(105deg,transparent 20%,rgba(255,255,255,.7) 43%,transparent 61%)}
.crt-st{position:absolute;top:8px;left:12px;font-family:system-ui,sans-serif;font-size:7px;
 letter-spacing:.16em;text-transform:uppercase;color:var(--response-label)}
.crt-id{position:absolute;top:8px;right:12px;font-size:8px;letter-spacing:.08em;
 color:var(--response-label)}
.crt-text{position:relative;z-index:1;font-size:15px;line-height:1.58;color:var(--response-ink);
 white-space:pre-wrap;word-break:break-word}
.tk.lo{border-bottom:1px dotted var(--signal-pink)}
.conf-legend{position:relative;z-index:1;font-family:system-ui,sans-serif;font-size:7.5px;
 letter-spacing:.06em;color:var(--response-label);padding:7px 2px 0}
.absent{padding:8px 1px;color:var(--ink-faint);font-style:italic}
.rrow{border-top:1px solid var(--line-soft);padding:9px 0 11px}
.rrow:first-child{border-top:0}
.r-inf{font-size:11px;color:var(--ink)}
.r-chips{display:flex;gap:6px;margin:5px 0 7px;flex-wrap:wrap;align-items:center}
.chip{font-family:system-ui,sans-serif;font-size:7.5px;letter-spacing:.09em;text-transform:uppercase;
 padding:2px 6px;border:1px solid var(--line);color:var(--ink-soft)}
.chip.eff{color:var(--captured);border-color:currentColor;background:rgba(120,228,204,.10)}
.chip.noeff{color:var(--ink-faint)}
.chip.cv{color:var(--derived);border-color:currentColor;background:rgba(94,219,229,.08)}
.nats{font-size:10px;color:var(--derived);font-weight:650}
.dep{display:grid;grid-template-columns:minmax(80px,150px) 58px 1fr;gap:3px 10px;
 align-items:center;max-width:480px;margin-top:3px}
.dep .p{white-space:pre;overflow:hidden;text-overflow:ellipsis;font-size:10px;color:var(--ink)}
.dep .d{font-size:9px;text-align:right;color:var(--ink-soft)}
.bar{height:5px;background:var(--line-soft);overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--signal-cyan),var(--signal-mint))}
.bar.neg i{background:linear-gradient(90deg,var(--signal-violet),var(--signal-pink))}
.small-note{font-size:8px;color:var(--ink-faint);padding-top:5px;line-height:1.6}
.imap-intro{font-size:9px;color:var(--ink-soft);padding:0 0 8px;line-height:1.6}
.imap-legend{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:0 0 9px;
 font-family:system-ui,sans-serif;font-size:8px;color:var(--ink-faint)}
.imap-key{display:inline-block;width:11px;height:8px;border:1px solid transparent}
.imap-key.sel{background:var(--signal-cyan);border-color:var(--derived)}
.imap-key.sup{background:var(--support-bg);border-color:var(--signal-mint)}
.imap-key.suppress{background:var(--suppress-bg);border-color:var(--signal-violet)}
.imap-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1px;
 border:1px solid var(--line);background:var(--line)}
.imap-side{min-width:0;padding:9px 10px;background:var(--panel-strong)}
.imap-side-h{font-family:system-ui,sans-serif;font-size:7.5px;letter-spacing:.14em;
 text-transform:uppercase;color:var(--ink-soft);padding-bottom:7px}
.imap-source{border-top:1px solid var(--line-soft);padding:7px 0}
.imap-source:first-of-type{border-top:0;padding-top:0}
.imap-source-k{font-family:system-ui,sans-serif;font-size:7px;letter-spacing:.09em;
 text-transform:uppercase;color:var(--ink-faint);padding-bottom:3px}
.im-answer{font-size:12px;line-height:1.75;color:var(--ink);white-space:pre-wrap;word-break:break-word}
.im-span{transition:background-color .1s,color .1s;cursor:help;outline:none}
.im-p{display:block;white-space:pre-wrap;word-break:break-word;padding:2px 3px;margin:0 -3px}
.im-a{display:inline;border-bottom:1px solid transparent}
.im-no-clear{border-bottom:1px dotted var(--ink-faint)}
.im-pin{display:none}
.im-pin:checked+.im-span{background:var(--signal-cyan)!important;color:#071014!important;
 box-shadow:inset 0 -2px 0 var(--signal-violet);border-bottom-color:transparent}
.imap-clear{display:inline-block;margin:0 0 9px;padding:2px 6px;border:1px solid var(--line);
 font-family:system-ui,sans-serif;font-size:7.5px;color:var(--ink-soft);cursor:pointer}
.imap-clear:hover,.imap-clear:focus{border-color:var(--signal-cyan);outline:none;color:var(--ink)}
.imap:has(#ic:checked) .im-span:is(:hover,:focus){background:var(--signal-cyan)!important;
 color:#071014!important;box-shadow:inset 0 -2px 0 var(--signal-violet);
 border-bottom-color:transparent}
.imap-state{margin-top:9px;padding:7px 9px;border-left:2px solid var(--signal-violet);
 background:rgba(167,140,247,.09);font-size:8.5px;color:var(--ink-soft)}
.imap-cut{color:var(--ink-faint);font-style:italic}
.active-line{font-size:9.5px;color:var(--ink-soft);padding:2px 0 8px}
.jgroup{padding:7px 0 3px;border-top:1px solid var(--line-soft)}
.jgroup:first-child{border-top:0}
.jlayer{font-family:system-ui,sans-serif;font-size:7.5px;letter-spacing:.14em;text-transform:uppercase;
 color:var(--ink-soft);padding-bottom:5px}
.jchips{display:flex;flex-wrap:wrap;gap:4px}
.jchip{font-size:8.5px;padding:1px 6px;border:1px solid var(--signal-violet);
 color:var(--suppress-ink);background:var(--suppress-bg);white-space:nowrap;max-width:100%;
 overflow:hidden;text-overflow:ellipsis}
.provenance{font-size:7.5px;color:var(--ink-faint);line-height:1.7;padding-top:8px}
.tree{font-size:10px;color:var(--ink-soft);white-space:pre;overflow-x:auto;padding:5px 1px}
.tree b{color:var(--ink)}
.foot{display:grid;grid-template-columns:auto minmax(120px,1fr) auto;align-items:center;gap:12px;
 padding:10px 13px}
.flowbar{height:5px;background:linear-gradient(90deg,var(--signal-cyan),var(--signal-mint),
 var(--signal-violet),var(--signal-pink),var(--signal-peach))}
.credo{font-family:system-ui,sans-serif;font-size:7.5px;letter-spacing:.13em;
 text-transform:uppercase;color:var(--ink-soft)}
.foot .rid{font-size:8px;color:var(--ink-faint)}
@media(max-width:900px){
 .card{grid-template-columns:1fr}.receipts,.lens{grid-column:1}
 .meta{grid-template-columns:repeat(3,minmax(0,1fr))}
 .meta>div:nth-child(3n){border-right:0}}
@media(max-width:620px){
 .receipt-shell{padding:0}.card{border-left:0;border-right:0;box-shadow:none}
 .imap-grid{grid-template-columns:1fr}.meta{grid-template-columns:repeat(2,minmax(0,1fr))}
 .meta>div:nth-child(3n){border-right:1px solid var(--line-soft)}
 .meta>div:nth-child(2n){border-right:0}.crt-id{display:none}
 .foot{grid-template-columns:1fr}.flowbar{order:-1}.theme-toggle{margin-left:0}}
@media print{
 html,body{background:#fff}.receipt-shell{padding:0;background:#fff;--panel:#fff;--panel-strong:#fff;
  --panel-soft:#f6f7f9;--ink:#111;--ink-soft:#333;--ink-faint:#666;--line:#aaa;
  --line-soft:#ddd;--shadow:transparent;--response-ink:#111;--response-label:#555;
  --response-bg:#f6f7f9}
 .card{width:100%;box-shadow:none}.theme-toggle{display:none}.imap-clear{display:none}
 .mod{break-inside:avoid}.influence{break-inside:auto}}
"""


# ------------------------------------------------------------------------------------------- sections
def _mod(led: str, title: str, tag_html: str, body_html: str, kind: str = "") -> str:
    classes = f"mod {kind}".rstrip()
    return (f'<section class="{classes}"><div class="mod-h"><span class="led {led}"></span>'
            f'<span class="cap">{_esc(title)}</span>{tag_html}</div>'
            f'<div class="mod-b">{body_html}</div></section>')


def _duration(timing: dict) -> str:
    ms = _float(timing.get("duration_ms"))
    if ms is None:
        return ""
    return f"{ms / 1000:.1f} s" if ms >= 10_000 else f"{int(ms)} ms"


def _masthead(run: dict, rid: str) -> str:
    timing = _dict(run.get("timing"))
    finish = run.get("finish_reason")
    rows = [
        ("run id", rid, ""),
        ("timestamp", run.get("created_at") or "?", ""),
        ("model", run.get("model") or "?", ""),
        ("substrate", run.get("substrate") or "?", ""),
        ("source", " · ".join(str(x) for x in (run.get("source"), run.get("client")) if x) or "?", ""),
        ("duration", _duration(timing) or "?", ""),
    ]
    if finish:
        rows.append(("stop", str(finish) + (" — tape ran out (token cap)" if finish == "length" else ""),
                     "warn" if finish == "length" else ""))
    if run.get("error"):
        rows.append(("error", str(run.get("error")), "warn"))
    meta = "".join(f'<div><span class="k">{_esc(k)}</span>'
                   f'<span class="v{" " + cls if cls else ""}">{_esc(v)}</span></div>'
                   for k, v, cls in rows)
    return (
        '<header class="mod masthead">'
        '<div class="mod-h"><span class="led"></span>'
        '<span class="wordmark">cloz<b>n</b></span>'
        '<span class="mast-sub">run receipt</span>'
        '<label class="theme-toggle" for="receipt-theme"><span>theme</span>'
        '<b aria-hidden="true"></b></label></div>'
        f'<div class="meta">{meta}</div>'
        '<div class="legend"><span class="tag cap-t">captured</span>'
        '<span class="legend-copy">run record</span>'
        '<span class="tag der-t">derived</span>'
        '<span class="legend-copy">post-run computation</span></div>'
        '</header>')


def _prompt_text(run: dict) -> tuple[str, int]:
    msgs = [m for m in _list(run.get("messages")) if isinstance(m, dict)]
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    prompt = str(user_msgs[-1].get("content", "")) if user_msgs else str(run.get("prompt_summary") or "")
    return prompt, max(0, len(msgs) - (2 if user_msgs else 0))


def _phosphor(trace: dict, response: str) -> str:
    """The reply as phosphor text: per-token opacity = recorded confidence; dotted underline below
    LOW_CONF. Falls back to the plain response string when the run carries no per-token trace."""
    tokens = _list(trace.get("tokens"))
    if not tokens:
        note = ('<div class="conf-legend">no per-token trace captured on this run — '
                'reply shown without confidence shading</div>')
        return f'<div class="crt-text">{_esc(response)}</div>' + note
    confidence = _list(trace.get("confidence"))
    spans, truncated = [], False
    for i, tok in enumerate(tokens):
        if i >= MAX_TOKENS:
            truncated = True
            break
        piece = str(tok)
        if piece == "":
            continue
        c = _float(confidence[i]) if i < len(confidence) else None
        if c is None:
            spans.append(f'<span class="tk" style="opacity:.8">{_esc(piece)}</span>')
            continue
        c = min(1.0, max(0.0, c))
        opacity = 0.35 + 0.65 * c
        lo = " lo" if c < LOW_CONF else ""
        spans.append(f'<span class="tk{lo}" style="opacity:{opacity:.2f}" '
                     f'title="conf {c:.2f}">{_esc(piece)}</span>')
    body = f'<div class="crt-text">{"".join(spans)}</div>'
    legend = ('<div class="conf-legend">phosphor brightness = recorded token confidence · '
              f'dotted = below {LOW_CONF:.2f}</div>')
    if truncated:
        legend += (f'<div class="conf-legend">token stream truncated for card size — '
                   f'{MAX_TOKENS} of {len(tokens)} tokens shown</div>')
    return body + legend


def _exchange(run: dict, trace: dict, rid: str) -> str:
    prompt, earlier = _prompt_text(run)
    parts = [f'<div class="well">{_esc(prompt)}</div>']
    if earlier > 0:
        parts.append(f'<div class="turnnote">… {earlier} earlier message'
                     f'{"" if earlier == 1 else "s"} not shown (full record in the export)</div>')
    crt = ('<div class="crt"><span class="crt-st">replay</span>'
           f'<span class="crt-id">{_esc(rid)}</span>'
           f'{_phosphor(trace, str(run.get("response") or ""))}</div>')
    parts.append(crt)
    tag = '<span class="tag cap-t">captured</span>'
    return _mod("blue", "the exchange", tag, "".join(parts), "exchange")


def _influence_delta(link: dict):
    delta = _float(link.get("delta_nats"))
    return delta if delta is not None and math.isfinite(delta) else None


def _influence_sign(link: dict, delta: float) -> str:
    if delta > 0:
        return "s"
    if delta < 0:
        return "x"
    effect = link.get("effect")
    if effect == "supports":
        return "s"
    if effect == "suppresses":
        return "x"
    return "n"


def _influence_title(relations: list) -> str:
    if not relations:
        return "hover, focus, or click to pin; no linked span cleared the measurement floor"
    delta = relations[0][2]
    return (f"hover, focus, or click to pin; strongest clear link {delta:+.3f} nats "
            "under matched context replacement")


def _influence_css(prompt_relations: list[list], answer_relations: list[list]) -> str:
    """Generate selectors from bounded numeric display indices only.

    Artifact IDs and text never enter CSS.  ``:has`` makes the two panes react to
    hover and keyboard focus without script, network access, or active content.
    """
    colors = {
        "s": ("var(--support-bg)", "var(--support-ink)"),
        "x": ("var(--suppress-bg)", "var(--suppress-ink)"),
        "n": ("var(--neutral-bg)", "var(--neutral-ink)"),
    }
    rules = []
    for prefix, relations in (("p", prompt_relations), ("a", answer_relations)):
        for index, linked in enumerate(relations):
            for sign in sorted({item[1] for item in linked}):
                bg, fg = colors[sign]
                rules.append(
                    f'.imap:has(#ic:checked):has(.im-{prefix}{index}:is(:hover,:focus)) '
                    f'.from-{prefix}{index}-{sign},'
                    f'.imap:has(#i{prefix}{index}:checked) .from-{prefix}{index}-{sign}'
                    f'{{background:{bg};color:{fg}}}'
                )
    return "".join(rules)


def _influence_section(bundle: dict) -> tuple[str, str]:
    influence = _dict(bundle.get("influence_map"))
    tag = '<span class="tag der-t">derived &middot; on demand</span>'
    if not influence:
        return (_mod("blue", "context ↔ answer influence", tag,
                     f'<div class="absent">{_esc(_ABSENT_INFLUENCE)}</div>', "influence"), "")
    if influence.get("schema") != "clozn.context_answer_influence.v1":
        body = ('<div class="absent">no compatible context-answer influence map is included in '
                'this receipt</div>')
        return _mod("blue", "context ↔ answer influence", tag, body, "influence"), ""
    if influence.get("status") != "ok" or influence.get("available") is not True:
        error = _dict(influence.get("error"))
        code = str(error.get("code") or influence.get("status") or "unavailable")[:80]
        message = str(error.get("message") or "the measurement did not complete")[:500]
        body = (f'<div class="absent">map unavailable — {_esc(code)}: '
                f'{_esc(message)}</div>')
        return _mod("blue", "context ↔ answer influence", tag, body, "influence"), ""

    raw_prompt_spans = [span for span in _list(influence.get("prompt_spans")) if isinstance(span, dict)]
    # Coarse-to-fine refinement (Phase 3.7) can push the measured span count above the display cap --
    # prioritize spans with at least one clearing link (a stable sort keeps every group's relative order)
    # so the interesting evidence a refinement pass just surfaced is never the part silently dropped.
    clearing_span_ids = {
        str(link.get("context_span_id")) for link in _list(influence.get("links"))
        if isinstance(link, dict) and link.get("clears_floor") is True
    }
    prioritized_prompt_spans = sorted(
        raw_prompt_spans, key=lambda span: str(span.get("id")) not in clearing_span_ids,
    )
    prompt_items = []
    prompt_truncated = False
    seen_prompt_ids = set()
    for span in prioritized_prompt_spans[:MAX_INFLUENCE_CONTEXT_SPANS]:
        span_id = str(span.get("id") or "")
        text = str(span.get("text") or "")
        if not span_id or not text or span_id in seen_prompt_ids:
            continue
        seen_prompt_ids.add(span_id)
        shown = text[:MAX_INFLUENCE_CONTEXT_CHARS]
        cut = len(shown) < len(text)
        prompt_truncated = prompt_truncated or cut
        prompt_items.append({"id": span_id, "span": span, "text": shown, "cut": cut})
    prompt_spans_omitted_by_cap = len(raw_prompt_spans) > len(prompt_items)

    answer_items = []
    answer_truncated = False
    answer_chars = 0
    seen_answer_ids = set()
    raw_answer_spans = _list(influence.get("answer_spans"))
    for span in raw_answer_spans:
        if len(answer_items) >= MAX_INFLUENCE_ANSWER_SPANS:
            answer_truncated = True
            break
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("id") or "")
        text = str(span.get("text") or "")
        if not span_id or not text or span_id in seen_answer_ids:
            continue
        remaining = MAX_INFLUENCE_ANSWER_CHARS - answer_chars
        if remaining <= 0:
            answer_truncated = True
            break
        shown = text[:remaining]
        cut = len(shown) < len(text)
        seen_answer_ids.add(span_id)
        answer_items.append({"id": span_id, "span": span, "text": shown, "cut": cut})
        answer_chars += len(shown)
        if cut:
            answer_truncated = True
            break

    if not prompt_items or not answer_items:
        body = ('<div class="absent">the influence artifact contains no renderable measured '
                'context and answer spans</div>')
        return _mod("blue", "context ↔ answer influence", tag, body, "influence"), ""

    prompt_index = {item["id"]: index for index, item in enumerate(prompt_items)}
    answer_index = {item["id"]: index for index, item in enumerate(answer_items)}
    # One strongest record per visible pair.  The schema normally has exactly
    # one; this also keeps malformed imported artifacts bounded and deterministic.
    pairs = {}
    link_limit = MAX_INFLUENCE_CONTEXT_SPANS * MAX_INFLUENCE_ANSWER_SPANS * 2
    for link in _list(influence.get("links"))[:link_limit]:
        if not isinstance(link, dict) or link.get("clears_floor") is not True:
            continue
        pi = prompt_index.get(str(link.get("context_span_id") or ""))
        ai = answer_index.get(str(link.get("answer_span_id") or ""))
        delta = _influence_delta(link)
        if pi is None or ai is None or delta is None:
            continue
        candidate = (pi, ai, _influence_sign(link, delta), delta)
        previous = pairs.get((pi, ai))
        if previous is None or abs(delta) > abs(previous[3]):
            pairs[(pi, ai)] = candidate

    prompt_relations = [[] for _ in prompt_items]
    answer_relations = [[] for _ in answer_items]
    for pi in range(len(prompt_items)):
        found = sorted((item for item in pairs.values() if item[0] == pi),
                       key=lambda item: (-abs(item[3]), item[1]))[:5]
        prompt_relations[pi] = [(item[1], item[2], item[3]) for item in found]
    for ai in range(len(answer_items)):
        found = sorted((item for item in pairs.values() if item[1] == ai),
                       key=lambda item: (-abs(item[3]), item[0]))[:3]
        answer_relations[ai] = [(item[0], item[2], item[3]) for item in found]

    prompt_backlinks = [[] for _ in prompt_items]
    answer_backlinks = [[] for _ in answer_items]
    for pi, linked in enumerate(prompt_relations):
        for ai, sign, _delta in linked:
            answer_backlinks[ai].append(f"from-p{pi}-{sign}")
    for ai, linked in enumerate(answer_relations):
        for pi, sign, _delta in linked:
            prompt_backlinks[pi].append(f"from-a{ai}-{sign}")

    prompt_html = []
    for index, item in enumerate(prompt_items):
        span = item["span"]
        role = str(span.get("role") or "context")[:50]
        source_kind = str(span.get("source_kind") or "recorded prompt")[:80]
        classes = ["im-span", "im-p", f"im-p{index}", *prompt_backlinks[index]]
        if not prompt_relations[index]:
            classes.append("im-no-clear")
        text = _esc(item["text"]) + ('<span class="imap-cut">...</span>' if item["cut"] else "")
        prompt_html.append(
            '<div class="imap-source">'
            f'<div class="imap-source-k">context {index + 1} &middot; {_esc(role)} '
            f'&middot; {_esc(source_kind)}</div>'
            f'<input class="im-pin" type="radio" name="i" id="ip{index}">'
            f'<label for="ip{index}" class="{" ".join(classes)}" tabindex="0" '
            f'title="{_esc(_influence_title(prompt_relations[index]))}">{text}</label></div>'
        )

    answer_html = []
    for index, item in enumerate(answer_items):
        classes = ["im-span", "im-a", f"im-a{index}", *answer_backlinks[index]]
        if not answer_relations[index]:
            classes.append("im-no-clear")
        text = _esc(item["text"]) + ('<span class="imap-cut">...</span>' if item["cut"] else "")
        answer_html.append(
            f'<input class="im-pin" type="radio" name="i" id="ia{index}">'
            f'<label for="ia{index}" class="{" ".join(classes)}" tabindex="0" '
            f'title="{_esc(_influence_title(answer_relations[index]))}">{text}</label>'
        )

    thresholds = _dict(influence.get("thresholds"))
    floor = _float(thresholds.get("cell_abs_delta_nats"))
    floor = floor if floor is not None and math.isfinite(floor) and floor >= 0 else None
    floor_copy = f"{floor:.3f} nats" if floor is not None else "the recorded measurement floor"
    intro = (
        '<div class="imap-intro">Hover or keyboard-focus either side, or click a span to pin it. '
        'The selected span turns blue; '
        'its strongest clear links turn mint when the context supported the recorded answer and lilac '
        'when it suppressed it. These are signed teacher-forced log-probability deltas under matched '
        'context replacement, not percentages, attention weights, or a circuit trace.</div>'
        '<div class="imap-legend"><span class="imap-key sel"></span> selected '
        '<span class="imap-key sup"></span> supports '
        '<span class="imap-key suppress"></span> suppresses '
        f'<span>measurement floor: {_esc(floor_copy)}</span></div>'
    )
    grid = (
        '<div class="imap"><input class="im-pin" type="radio" name="i" id="ic" checked>'
        '<label for="ic" class="imap-clear" tabindex="0">clear pinned highlight</label>'
        '<div class="imap-grid">'
        '<div class="imap-side"><div class="imap-side-h">measured recorded context</div>'
        f'{"".join(prompt_html)}</div>'
        '<div class="imap-side"><div class="imap-side-h">recorded answer</div>'
        f'<div class="im-answer">{"".join(answer_html)}</div></div></div>'
    )
    notes = []
    if not pairs:
        notes.append(f"No clear source found: no visible context-answer link cleared {floor_copy}.")
    else:
        no_answer = sum(not linked for linked in answer_relations)
        no_context = sum(not linked for linked in prompt_relations)
        if no_answer:
            notes.append(f"{no_answer} visible answer span(s) have no clear source above the floor; "
                         "their dotted underline is an honest no-clear-source state.")
        if no_context:
            notes.append(f"{no_context} visible context span(s) have no clear answer effect above the floor.")
    selection = _dict(influence.get("selection"))
    omitted = len(_list(selection.get("omitted_source_ids")))
    if omitted:
        notes.append(f"{omitted} recorded prompt source(s) were outside the bounded measurement; "
                     "the card makes no influence claim for them.")
    redundancy = _dict(influence.get("redundancy_check"))
    if redundancy.get("performed") is True:
        pair_labels = [
            f"context {prompt_index[str(span_id)] + 1}"
            for span_id in _list(redundancy.get("context_span_ids"))
            if str(span_id) in prompt_index
        ]
        if len(pair_labels) == 2:
            interactions = [
                _float(item.get("interaction_nats")) for item in _list(redundancy.get("per_answer_token"))
                if isinstance(item, dict)
            ]
            interactions = [value for value in interactions if value is not None]
            strongest = max((abs(value) for value in interactions), default=None)
            strength_copy = f"; strongest measured interaction {strongest:.3f} nats" if strongest is not None else ""
            notes.append(
                f"Redundant-pair check: {pair_labels[0]} and {pair_labels[1]} were replaced together in "
                f"one bounded joint control{strength_copy} — near zero means the two behaved additively, "
                "large means they overlapped; never a percentage of total explanation."
            )
    if (prompt_truncated or answer_truncated or prompt_spans_omitted_by_cap
            or len(raw_answer_spans) > len(answer_items)):
        notes.append("The interactive view was truncated for receipt size; the complete measured "
                     "artifact remains in the JSON export.")
    state = "".join(f'<div class="imap-state">{_esc(note)}</div>' for note in notes)
    measured_tag = '<span class="tag der-t">derived &middot; matched replacement</span>'
    body = intro + grid + state + "</div>"
    return (_mod("blue", "context ↔ answer influence", measured_tag, body, "influence"),
            _influence_css(prompt_relations, answer_relations))


def _influence_label(inf) -> str:
    inf = _dict(inf)
    txt = inf.get("text")
    if txt:
        return str(txt)
    return "influence"


def _dep_bars(top_dependent: list) -> str:
    rows = [d for d in _list(top_dependent) if isinstance(d, dict)]
    if not rows:
        return ""
    max_abs = max((abs(_float(d.get("delta")) or 0.0) for d in rows), default=0.0) or 1.0
    cells = []
    for d in rows:
        delta = _float(d.get("delta")) or 0.0
        width = min(100.0, abs(delta) / max_abs * 100.0)
        neg = " neg" if delta < 0 else ""
        cells.append(f'<span class="p">{_esc(d.get("piece"))}</span>'
                     f'<span class="d">{_num(delta, 3, signed=True)}</span>'
                     f'<span class="bar{neg}"><i style="width:{width:.0f}%"></i></span>')
    return '<div class="dep">' + "".join(cells) + "</div>"


def _receipt_row(r: dict) -> str:
    inf = _influence_label(r.get("influence"))
    forced = r if r.get("mode") == "forced" else _dict(r.get("forced"))
    has_effect = bool(r.get("has_effect"))
    cv = r.get("causal_verified")
    if r.get("mode") == "forced":
        eff_chip = ('<span class="chip eff">leaning detected</span>' if has_effect
                    else '<span class="chip noeff">no measurable leaning</span>')
    else:
        eff_chip = ('<span class="chip eff">changed the answer</span>' if has_effect
                    else '<span class="chip noeff">answer unchanged</span>')
    cv_chip = ('<span class="chip cv">causal · verified</span>' if cv is True else
               '<span class="chip">causal_verified: false</span>' if cv is False else
               '<span class="chip">causal_verified: null</span>')
    sum_nats = forced.get("sum_nats", r.get("sum_nats"))
    nats = (f'<span class="nats">Σ {_num(sum_nats, 3, signed=True)} nats</span>'
            if _float(sum_nats) is not None else "")
    bars = _dep_bars(forced.get("top_dependent", r.get("top_dependent")))
    return (f'<div class="rrow"><div class="r-inf">{_esc(inf)}</div>'
            f'<div class="r-chips">{eff_chip}{cv_chip}{nats}</div>{bars}</div>')


def _receipt_rows(receipts_obj: dict) -> list:
    rows = [r for r in _list(receipts_obj.get("receipts")) if isinstance(r, dict)]
    rows += [r for r in _list(receipts_obj.get("forced_receipts")) if isinstance(r, dict)]
    return rows


def _receipts_section(bundle: dict) -> str:
    receipts_obj = _dict(bundle.get("receipts"))
    rows = _receipt_rows(receipts_obj) if receipts_obj else []
    if not rows:
        tag = '<span class="tag der-t">derived · on demand</span>'
        body = f'<div class="absent">{_esc(_ABSENT_RECEIPTS)}</div>'
        return _mod("lilac", "influences & receipts", tag, body, "receipts")
    when = receipts_obj.get("computed_at") or "on demand"
    tag = (f'<span class="tag der-t">derived — computed {_esc(when)} by leave-one-out + '
           'forced scoring</span>')
    body = "".join(_receipt_row(r) for r in rows)
    skipped = _list(receipts_obj.get("skipped"))
    if skipped:
        body += (f'<div class="small-note">{len(skipped)} influence'
                 f'{"" if len(skipped) == 1 else "s"} skipped — reasons in the JSON export</div>')
    return _mod("lilac", "influences & receipts", tag, body, "receipts")


def _lens_section(bundle: dict) -> str:
    readouts = [r for r in _list(bundle.get("workspace_readouts")) if isinstance(r, dict)]
    if not readouts:
        tag = '<span class="tag der-t">derived · on demand</span>'
        return _mod("blue", "lens readouts", tag,
                    f'<div class="absent">{_esc(_ABSENT_LENS)}</div>', "lens")
    by_layer: dict = {}
    for r in readouts:
        by_layer.setdefault(r.get("layer"), []).append(r)
    groups = []
    for layer in sorted(by_layer, key=lambda x: (x is None, x)):
        chips = []
        for r in by_layer[layer][:40]:
            tops = [str(_dict(t).get("label") or "") for t in _list(r.get("top_readouts"))[:2]]
            tops = [t for t in tops if t.strip()]
            if not tops:
                continue
            tok = str(r.get("token_text") or "")
            chips.append(f'<span class="jchip">{_esc(tok)} → {_esc(", ".join(tops))}</span>')
        if chips:
            label = f"layer {layer}" if layer is not None else "layer ?"
            provider = by_layer[layer][0].get("provider") or by_layer[layer][0].get("provider_type") or ""
            groups.append(f'<div class="jgroup"><div class="jlayer">{_esc(label)}'
                          f'{" · " + _esc(provider) if provider else ""}</div>'
                          f'<div class="jchips">{"".join(chips)}</div></div>')
    if not groups:
        tag = '<span class="tag der-t">derived · on demand</span>'
        return _mod("blue", "lens readouts", tag,
                    f'<div class="absent">{_esc(_ABSENT_LENS)}</div>', "lens")
    first = readouts[0]
    provenance = str(first.get("provenance") or first.get("note") or "")
    if not provenance:
        provenance = (_JLENS_CAPTION if str(first.get("provider_type") or "") == "jacobian_lens"
                      else f"workspace readout — provider {first.get('provider_type') or 'unknown'}")
    tag = '<span class="tag der-t">derived — lens readout</span>'
    body = "".join(groups) + f'<div class="provenance">{_esc(provenance)}</div>'
    return _mod("blue", "lens readouts", tag, body, "lens")


def _tree_lines(node: dict, rid: str, depth: int, out: list) -> None:
    if depth > 10 or len(out) >= 100 or not isinstance(node, dict):
        return
    nid = str(node.get("id") or "?")
    label = str(node.get("change_label") or "")
    mark = " ◀ this run" if node.get("id") == rid or node.get("is_current") else ""
    prefix = ("  " * depth + "└ ") if depth else ""
    line = f"{prefix}<b>{_esc(nid)}</b>"
    if label:
        line += f" · {_esc(label)}"
    line += _esc(mark)
    out.append(line)
    for child in _list(node.get("children")):
        _tree_lines(child, rid, depth + 1, out)


def _lineage_section(bundle: dict, run: dict, rid: str) -> str:
    tag = '<span class="tag cap-t">captured</span>'
    lineage = _dict(bundle.get("lineage"))
    tree = _dict(lineage.get("tree"))
    if tree:
        lines: list = []
        _tree_lines(tree, rid, 0, lines)
        if len(lines) > 1 or tree.get("children"):
            return _mod("lilac", "lineage", tag,
                        f'<div class="tree">{"<br>".join(lines)}</div>', "lineage")
    parent = run.get("parent_run_id")
    if parent:
        body = (f'<div class="tree"><b>{_esc(parent)}</b> · parent<br>'
                f'└ <b>{_esc(rid)}</b> ◀ this run</div>')
        return _mod("lilac", "lineage", tag, body, "lineage")
    return _mod("lilac", "lineage", tag,
                '<div class="absent">no lineage — an original run '
                '(no parent, no recorded branches)</div>', "lineage")


def _footer(rid: str) -> str:
    return (f'<footer class="mod receipt-footer"><div class="foot">'
            f'<span class="credo">{_esc(_FOOTER_LABEL)}</span>'
            f'<span class="flowbar"></span>'
            f'<span class="rid">{_esc(rid)}</span></div></footer>')


# ------------------------------------------------------------------------------------------------ API
def render_card(bundle: dict) -> str:
    """One export bundle -> one self-contained HTML receipt card (a string). Never raises on missing
    fields: every section degrades to its honest-absence copy."""
    bundle = _dict(bundle)
    run = _dict(bundle.get("run"))
    trace = _dict(bundle.get("trace")) or _dict(run.get("trace"))
    rid = str(run.get("id") or "unknown-run")
    influence_html, influence_css = _influence_section(bundle)
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="color-scheme" content="light dark">',
        f"<title>clozn — run receipt · {_esc(rid)}</title>",
        f"<style>{_CSS}{influence_css}</style>",
        "</head>",
        "<body>",
        '<div class="receipt-shell">',
        '<input class="theme-toggle-input" type="checkbox" id="receipt-theme" '
        'aria-label="Use Cathedral theme">',
        '<main class="card">',
        _masthead(run, rid),
        _exchange(run, trace, rid),
        influence_html,
        _receipts_section(bundle),
        _lens_section(bundle),
        _lineage_section(bundle, run, rid),
        _footer(rid),
        "</main>",
        "</div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)
