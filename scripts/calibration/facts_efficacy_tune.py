"""facts_efficacy_tune.py -- GPU facts step-targeted efficacy tuning (BK, 2026-07-23).

The torch SlotMem store's step-targeted injection got 42% exact recall on an UNTUNED 1.5B/L14 CPU
run (real 5/12, null 0/12 -- value-specific but low efficacy). Now that torch sees the RTX 5080
(cu128), sweep the three knobs on the actual product-scale model (Qwen2.5-7B, auto-nf4 on cuda) to
find whether recall clears a shippable bar:
  - LAYER  : the tap layer (slotmem findings validated L18 on 7B)
  - ETA    : injection magnitude, as a multiplier on the auto eta (INJECT_FRAC * resid_norm)
  - SCHED  : how many leading answer-token directions to inject (step-1-only vs 1-2 vs 1-3)

Null-controlled throughout (random-equal-norm direction, same schedule) so a high "real" number is
only meaningful if null stays ~0. Model loaded ONCE; layers re-tapped via from_shared (one hook at
a time). Writes runs/experiments/facts_efficacy_tune_7b.json.

Decision: PROMOTE if some config reaches real >= 0.7 with null ~0; else report the ceiling honestly
(the mechanism's real efficacy limit at 7B), which itself decides the facts-tier fate.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from clozn.lab.slotmem_qwen import store as slot  # noqa: E402


FACTS = [
    ("The secret vault passcode at Meridian Bank is", " QUARTZ"),
    ("Dr. Elowen Marsh's registered lab element is", " Rhenium"),
    ("The Zephyr-7 probe was launched from the port city of", " Valdora"),
    ("Captain Brixby's assigned call sign is", " Nightjar"),
    ("The Thornfield estate's gate combination begins with the digit", " 7"),
    ("Professor Quill keeps her research notes in a binder colored", " magenta"),
    ("The founding year carved above the Aldermoor library door is", " 1847"),
    ("The rare orchid in the Kestrel greenhouse is nicknamed the", " Emberwing"),
    ("Agent Voss reports to the field office located in", " Helsinki"),
    ("The house special at the Copper Kettle diner is called the", " Drover"),
    ("Mayor Ashford's championship-winning horse was named", " Cinder"),
    ("The observatory on Mount Hale tracks the comet designated", " KX9"),
]


def hit(text, answer):
    return answer.strip().lower() in text.strip().lower()


@torch.no_grad()
def emit_with(store, query, entry, mode, rng, sched, base_eta, eta_mult, max_new=6):
    """SlotMem.emit's schedule, parametrized: inject the first `sched` answer-token directions at
    successive decode steps (mode='real'), random-equal-norm vectors (mode='null'), or nothing
    (mode='baseline'). eta = base_eta * eta_mult."""
    ids = store.tok(query, return_tensors="pt").input_ids.to(slot.DEV)
    seq = ids
    if mode != "baseline":
        vecs = []
        for i in range(min(sched, len(entry["ans_ids"]))):
            if i == 0:
                v = entry["value"]                       # unit W_U[ans_ids[0]]
            else:
                v = store.W_U[entry["ans_ids"][i]].float()
                v = v / (v.norm() + 1e-8)
            vecs.append(v)
        if mode == "null":
            # CPU generator -> make on CPU, then move to the value's device+dtype (cuda).
            vecs = [(lambda r: r / (r.norm() + 1e-8))(
                        torch.randn(v.shape, generator=rng).to(device=v.device, dtype=v.dtype))
                    for v in vecs]
        eta = base_eta * eta_mult
        for v in vecs:
            store._inject = eta * v
            try:
                nxt = store.model(seq).logits[0, -1].argmax()
            finally:
                store._inject = None
            seq = torch.cat([seq, nxt.view(1, 1)], 1)
    remaining = max_new - (seq.shape[1] - ids.shape[1])
    out = seq if remaining <= 0 else store.model.generate(
        seq, attention_mask=torch.ones_like(seq), max_new_tokens=remaining,
        do_sample=False, pad_token_id=store.tok.eos_token_id or 0)
    return store.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def run(model_name, layers, eta_mults, scheds, tag):
    rng = torch.Generator().manual_seed(0)
    print(f"[load] {model_name} in bf16 (no bitsandbytes) ...", flush=True)
    # Load bf16 directly and use from_shared, bypassing SlotMem.__init__'s auto-nf4 (which needs
    # bitsandbytes). 7B bf16 ~14GB fits the 16GB card with the C++ engine killed.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(slot.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base = slot.SlotMem.from_shared(model, tok, layers[0])

    grid = []
    # baseline (layer-independent) computed once per fact, to find the usable (unknown) facts.
    usable = []
    for cue, ans in FACTS:
        base.entries = []
        base.write(cue, ans, gate=False)
        e = base.entries[-1]
        b = emit_with(base, cue, e, "baseline", rng, 1, base.eta, 1.0)
        if not hit(b, ans):
            usable.append((cue, ans))
    print(f"usable (unknown-to-base) facts: {len(usable)}/{len(FACTS)}", flush=True)

    for layer in layers:
        base.close()                                     # drop current hook
        store = slot.SlotMem.from_shared(model, tok, layer)   # new hook at `layer`, eta auto-set
        base = store
        base_eta = store.eta
        # pre-write the usable facts once at this layer (keys are layer-specific)
        entries = {}
        for cue, ans in usable:
            store.entries = []
            store.write(cue, ans, gate=False)
            entries[cue] = store.entries[-1]
        for eta_mult, sched in itertools.product(eta_mults, scheds):
            real = null = 0
            for cue, ans in usable:
                e = entries[cue]
                r = emit_with(store, cue, e, "real", rng, sched, base_eta, eta_mult)
                n = emit_with(store, cue, e, "null", rng, sched, base_eta, eta_mult)
                real += hit(r, ans)
                null += hit(n, ans)
            row = {"layer": layer, "eta_mult": eta_mult, "sched": sched,
                   "base_eta": round(base_eta, 1), "n": len(usable),
                   "real": real, "null": null,
                   "real_rate": round(real / len(usable), 3), "null_rate": round(null / len(usable), 3)}
            grid.append(row)
            print(f"L{layer} eta*{eta_mult} sched{sched}: real {real}/{len(usable)} "
                  f"({row['real_rate']}) null {null}", flush=True)

    graded = [g for g in grid if g["null"] == 0]        # only configs where the control stays clean
    best = max(graded, key=lambda g: g["real_rate"]) if graded else None
    ceiling = max(g["real_rate"] for g in grid) if grid else None
    summary = {
        "model": model_name, "tag": tag, "n_usable": len(usable),
        "best_clean_config": best, "best_real_rate_any": ceiling,
        "verdict": (
            f"PROMOTE-READY: {best['real_rate']:.0%} recall at L{best['layer']} eta*{best['eta_mult']} "
            f"sched{best['sched']} with null 0 -- clears the bar"
            if best and best["real_rate"] >= 0.7 else
            f"CEILING {ceiling:.0%} (best null-clean {best['real_rate']:.0%} at L{best['layer']} "
            f"eta*{best['eta_mult']} sched{best['sched']}) -- step-targeted injection tops out below a "
            f"shippable-memory bar on 7B; the facts tier stays lab research, not product memory"
            if best else
            "no null-clean config recovered facts -- mechanism does not promote") ,
    }
    out = os.path.join(REPO, "runs", "experiments", f"facts_efficacy_tune_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "grid": grid}, open(out, "w"), indent=2)
    print("\n=== facts efficacy tuning ===")
    print(" ", summary["verdict"])
    print(f"wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", default="14,18,21,25")
    ap.add_argument("--eta-mults", default="1.0,1.5,2.5")
    ap.add_argument("--scheds", default="1,2,3")
    ap.add_argument("--tag", default="7b")
    args = ap.parse_args()
    run(args.model, [int(x) for x in args.layers.split(",")],
        [float(x) for x in args.eta_mults.split(",")],
        [int(x) for x in args.scheds.split(",")], args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
