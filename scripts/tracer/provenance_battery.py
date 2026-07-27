"""provenance_battery.py -- the R1 validation battery for clozn.analysis.provenance.

The pilot (notes/CIRCUIT_TRACER_DESIGN.md section 5h) validated trace_provenance on 4 prompts, 1
model family. Before any product surface ships a CONTEXT_CARRIED / MIXED / PARAMETRIC chip
(docs/PRODUCT_ROADMAP.md Phase 3.7, lane R1), the verdict system needs to face a real battery:
does it discriminate across ~30 varied cases, on more than one model family, and does the
INCONCLUSIVE guard ever actually fire on a real prompt (it never has)?

GRADING IS CONDITIONAL ON THE MODEL'S OWN ANSWER -- we never grade the tracer on an answer the
model did not give. Each case declares which verdicts count as agreement GIVEN what the model
said:
  * invented-entity cases: the fact exists only in the prompt, so if the model produces it, the
    ONLY honest verdict is CONTEXT_CARRIED. If the model fails to produce the context answer, the
    case is UNGRADEABLE (a model failure, not a tracer failure) and excluded from agreement.
  * counterfactual/distractor cases: if the model followed the context's override, expect
    dependence (CONTEXT_CARRIED or MIXED); if it answered from weights against the context,
    expect the answer NOT to be context-carried (PARAMETRIC or MIXED -- the measured Kyoto case
    sat at MIXED 0.55, so MIXED satisfies both sides and only a WRONG-side extreme fails).
  * kv/induction/bare-factual: CONTEXT_CARRIED expected (a bare question's own content words ARE
    the carrying context -- measured: 'capital of France' dep 1.00 via [' capital', ' France']).
INCONCLUSIVE never counts as agreement; it is tallied separately -- a high INCONCLUSIVE rate is
its own finding (the guard fires too easily), and a zero rate across 30 cases leaves the guard
unexercised, which the report says out loud.

Needs a clozn-server started with --no-flash-attn (attn weights must materialize). Any AR GGUF;
no J-lens, no sidecars -- run it on a second family for free:

    clozn-server ~/.clozn/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8091 --gpu-layers 99 \
        --workers 2 --no-flash-attn
    python scripts/tracer/provenance_battery.py --engine http://127.0.0.1:8091 --tag qwen2.5-7b

    clozn-server ~/.clozn/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8092 ... same flags
    python scripts/tracer/provenance_battery.py --engine http://127.0.0.1:8092 --tag llama3.1-8b

Writes runs/experiments/provenance_battery_<tag>.json (full receipts + the summary table it
prints). Wall clock: roughly 5-20s per case (singles scan + greedy + controls), ~5-10 min total.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from clozn.analysis.provenance import ProvenanceBudget, available, trace_provenance  # noqa: E402

# ---------------------------------------------------------------------------------------- the cases
# Each case: id, category, prompt, and a grade() closure receiving (answer_piece, verdict) and
# returning "agree" | "disagree" | "ungradeable". `answer_piece` is the model's own greedy first
# token for this prompt (whitespace-stripped, lowercased for matching).

def _expect(*ok_verdicts, require_answer=None, forbid_answer=None):
    """Grade helper: agreement iff the measured verdict is in ok_verdicts. If require_answer is
    given, the case is UNGRADEABLE unless one of those strings appears in the HEAD of the model's
    answer (first 30 chars, lowercased) -- containment, not startswith, because models routinely
    lead with punctuation/format scaffolding (measured: '. The wizard Zorblax...' and
    ':\\nA: Almond milk' are honest context answers that a startswith rule threw away). If a
    forbid_answer string appears instead, the expectation FLIPS to the not-context-carried side
    (PARAMETRIC/MIXED) -- used by counterfactual cases where either behavior is possible. If BOTH
    appear in the head, the case is ungradeable (can't tell which the model committed to)."""
    ok = set(ok_verdicts)

    def grade(answer: str, verdict: str) -> str:
        head = answer.strip().lower()[:30]
        if require_answer is not None:
            req = any(r in head for r in require_answer)
            forb = forbid_answer is not None and any(f in head for f in forbid_answer)
            if req and forb:
                return "ungradeable"
            if not req:
                if forb:
                    # The model RESISTED the override and answered from weights. The 2nd-family
                    # run (Llama-3.1-8B) showed the old expectation here -- "then the verdict must
                    # be PARAMETRIC/MIXED" -- is MIS-SPECIFIED: cutting the QUESTION tokens still
                    # kills the parametric answer (you cannot answer 'the capital of France is'
                    # without reading ' France'), so total-context dependence is legitimately high
                    # even when the OVERRIDE went unused. Total dependence conflates "used the
                    # override" with "used the question"; separating them needs focus-span
                    # dependence (cut only the override/document region) -- a provenance feature,
                    # not a grading rule. Until then this branch is DESCRIPTIVE, not graded.
                    return "descriptive"
                return "ungradeable"
        return "agree" if verdict in ok else "disagree"

    return grade


def _descriptive(**_ignored):
    """For cases with NO model-independent expected verdict. The distractor_parametric category
    is the measured example (2nd-family run): the same prompt honestly reads MIXED/PARAMETRIC on
    Qwen2.5-7B (which retains 'Tokyo' even with context access cut) and CONTEXT_CARRIED on
    Llama-3.1-8B (whose 'Tokyo' dies without reading ' Japan') -- a REAL cross-model difference in
    parametric retention, correctly measured both times, so neither verdict is 'wrong'. These
    cases are reported descriptively and excluded from the agreement score."""
    def grade(answer: str, verdict: str) -> str:
        return "descriptive"
    return grade


CASES = [
    # -- in-context key/value: the fact exists nowhere but the prompt ------------------------------
    dict(id="kv_color", category="kv_recall",
         prompt="The box is blue. The lamp is red. The cup is green. The color of the box is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("blue",))),
    dict(id="kv_number", category="kv_recall",
         prompt="In today's inventory, crate A holds 17 items and crate B holds 42 items. "
                "The number of items in crate B is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("42", "forty-two"))),
    dict(id="kv_name", category="kv_recall",
         prompt="The project lead is Ms. Okafor and the deputy is Mr. Chen. Questions about the "
                "budget go to the project lead, whose name is Ms.",
         grade=_expect("CONTEXT_CARRIED", require_answer=("okafor",))),
    dict(id="kv_late", category="kv_recall",
         prompt="Meeting notes: attendance was low. Action items were assigned. The passcode for "
                "the shared drive was changed to 9314. Snacks were provided. The new passcode for "
                "the shared drive is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("9314",))),
    # -- induction: an invented pattern the model must copy from earlier in the prompt -------------
    dict(id="ind_zorblax", category="induction",
         prompt="The wizard Zorblax cast a spell. Everyone cheered for the wizard",
         grade=_expect("CONTEXT_CARRIED", require_answer=("zorbl",))),
    dict(id="ind_pairs", category="induction",
         prompt="glim -> tok, brue -> nix, glim -> tok, brue -> nix, glim ->",
         grade=_expect("CONTEXT_CARRIED", require_answer=("tok",))),
    dict(id="ind_captain", category="induction",
         prompt="Captain Vrenna said the ship would sail at dawn. At dawn, the crew waited for "
                "Captain",
         grade=_expect("CONTEXT_CARRIED", require_answer=("vrenna", "vren"))),
    # -- invented-entity document QA: only the doc can supply the answer ---------------------------
    dict(id="doc_florpium", category="doc_qa_invented",
         prompt="Excerpt from a materials datasheet: 'Florpium-9 melts at 214 degrees Celsius "
                "and is stored under argon.' According to the datasheet, the melting point of "
                "Florpium-9 in degrees Celsius is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("214",))),
    dict(id="doc_town", category="doc_qa_invented",
         prompt="From the guidebook: 'The village of Marrowdale lies in the Tessel Valley and is "
                "famous for its glass bridges.' The guidebook says Marrowdale is famous for its",
         grade=_expect("CONTEXT_CARRIED", require_answer=("glass",))),
    dict(id="doc_policy", category="doc_qa_invented",
         prompt="Company policy 7.3: 'Remote employees must file expense reports within 11 days.' "
                "Under policy 7.3, the deadline for expense reports, in days, is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("11", "eleven"))),
    dict(id="doc_dosage", category="doc_qa_invented",
         prompt="Trial protocol: 'Group A receives 35 mg of compound XR-77 daily.' Per the "
                "protocol, the daily dose of XR-77 for Group A, in mg, is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("35", "thirty"))),
    dict(id="doc_recipe", category="doc_qa_invented",
         prompt="Recipe card: 'Nan's tarts bake for 22 minutes at 180C.' The card says the tarts "
                "bake for a number of minutes equal to",
         grade=_expect("CONTEXT_CARRIED", require_answer=("22", "twenty"))),
    # -- counterfactual override: the context contradicts strong parametric knowledge --------------
    # If the model follows the story -> dependence expected. If it resists -> not context-carried.
    dict(id="cf_capital", category="counterfactual",
         prompt="In this story, the capital of France is Lyon. In the story, the capital of "
                "France is",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("lyon",),
                       forbid_answer=("paris",))),
    dict(id="cf_gravity", category="counterfactual",
         prompt="In this fictional universe, objects fall upward. In this universe, when you "
                "drop a stone it falls",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("up",),
                       forbid_answer=("down",))),
    dict(id="cf_arithmetic", category="counterfactual",
         prompt="In this puzzle's rules, 2 + 2 equals 5. Following the puzzle's rules exactly, "
                "2 + 2 equals",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("5", "five"),
                       forbid_answer=("4", "four"))),
    dict(id="cf_color", category="counterfactual",
         prompt="In the painting described here, the sky is green and the grass is orange. In "
                "the painting, the color of the sky is",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("green",),
                       forbid_answer=("blue",))),
    # -- distractor/parametric: context is present but the weights carry the answer ----------------
    dict(id="dis_kyoto", category="distractor_parametric",
         prompt="Kyoto was the capital of Japan for over a thousand years. Since the Meiji era, "
                "however, the government has been located elsewhere. The modern capital of Japan "
                "is",
         grade=_descriptive()),
    dict(id="dis_istanbul", category="distractor_parametric",
         prompt="Constantinople was for centuries the seat of empires. Today the city bears a "
                "different name. The largest city in Turkey is",
         grade=_descriptive()),
    dict(id="dis_rio", category="distractor_parametric",
         prompt="Rio de Janeiro served as Brazil's capital for nearly two centuries and remains "
                "its most famous city. The current capital of Brazil is",
         grade=_descriptive()),
    dict(id="dis_edison", category="distractor_parametric",
         prompt="Many people credit Thomas Edison with inventing the telephone, since he invented "
                "so many devices. In fact, the telephone was invented by Alexander Graham",
         grade=_descriptive()),
    # -- bare factual: the question's own content words are the only context -----------------------
    dict(id="fact_paris", category="bare_factual",
         prompt="The capital of France is",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("paris",))),
    dict(id="fact_water", category="bare_factual",
         prompt="The chemical formula for water is",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("h2o",))),
    dict(id="fact_everest", category="bare_factual",
         prompt="The tallest mountain on Earth is Mount",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("everest",))),
    dict(id="fact_rome", category="bare_factual",
         prompt="The Colosseum is located in the city of",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("rome",))),
    # -- negation: the carrying context includes a negation the span search must find --------------
    dict(id="neg_allergy", category="negation",
         prompt="Sam is allergic to peanuts but not to almonds. A safe snack for Sam would be",
         grade=_expect("CONTEXT_CARRIED", "MIXED", require_answer=("almond",))),
    dict(id="neg_closed", category="negation",
         prompt="The museum is open every day except Monday. The one day you cannot visit is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("monday",))),
    # -- arithmetic-from-context: numbers only the prompt supplies ---------------------------------
    dict(id="arith_sum", category="arith_from_context",
         prompt="A ticket costs 12 dollars and a program costs 5 dollars. Together, one ticket "
                "and one program cost, in dollars,",
         grade=_expect("CONTEXT_CARRIED", require_answer=("17", "seventeen"))),
    dict(id="arith_count", category="arith_from_context",
         prompt="Ana brought 3 pies and Ben brought 4 pies. The total number of pies is",
         grade=_expect("CONTEXT_CARRIED", require_answer=("7", "seven"))),
    # -- long context: the carrying fact is buried mid-prompt among filler -------------------------
    dict(id="long_code", category="long_context",
         prompt="Minutes of the residents' association: The garden fence will be repainted in "
                "April. Parking permits renew in June. The gate code was changed yesterday to "
                "5261 for security reasons. The newsletter needs a new editor. The bake sale "
                "raised 84 dollars. To open the gate, enter the code",
         grade=_expect("CONTEXT_CARRIED", require_answer=("5261",))),
    dict(id="long_name", category="long_context",
         prompt="Trip log, day 3: We crossed the ridge before noon. Our guide, whose name is "
                "Petra Lindqvist, found a shortcut past the scree field. Weather held until "
                "evening. Dinner was lentils again. The name of our guide is Petra",
         grade=_expect("CONTEXT_CARRIED", require_answer=("lind",))),
]


# ---------------------------------------------------------------------------------------- the run

def _greedy_answer(engine: str, prompt: str, max_tokens: int = 8) -> str:
    """The model's own greedy answer text -- graded on the WHOLE short answer, not just the first
    token piece (which is often a bare space/newline before the content word and made honest cases
    ungradeable). Passed to trace_provenance as the explicit continuation, so the traced target is
    still exactly this answer's first token."""
    import urllib.request
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(engine.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["text"]


def run_battery(engine: str, tag: str, budget: ProvenanceBudget, out_dir: str) -> dict:
    results = []
    t0 = time.time()
    for i, case in enumerate(CASES):
        t = time.time()
        try:
            answer_text = _greedy_answer(engine, case["prompt"])
        except Exception as e:
            answer_text = ""
            print(f"  [{i+1:>2}/{len(CASES)}] {case['id']:<16} greedy generation failed: {e}")
        receipt = trace_provenance(case["prompt"], answer_text or None, engine_url=engine,
                                   budget=budget, seed=i)
        wall = round(time.time() - t, 1)
        if not receipt.get("ok"):
            results.append({"id": case["id"], "category": case["category"], "ok": False,
                            "blocked": receipt.get("blocked"), "wall_s": wall})
            print(f"  [{i+1:>2}/{len(CASES)}] {case['id']:<16} BLOCKED: {receipt.get('blocked')}")
            continue
        v = str(receipt.get("verdict", ""))
        outcome = "inconclusive" if v == "INCONCLUSIVE" else case["grade"](answer_text, v)
        dep = receipt.get("dependence")
        ratio = receipt.get("best_control_ratio")
        results.append({"id": case["id"], "category": case["category"], "ok": True,
                        "answer": answer_text, "traced_piece": receipt.get("answer"),
                        "verdict": v, "outcome": outcome,
                        "dependence": dep, "best_control_ratio": ratio,
                        "span_tokens": receipt.get("span_tokens"),
                        "wall_s": wall, "receipt": receipt})
        dep_s = f"{dep:.3f}" if isinstance(dep, (int, float)) else str(dep)
        ratio_s = f"{ratio:.1f}" if isinstance(ratio, (int, float)) else str(ratio)
        print(f"  [{i+1:>2}/{len(CASES)}] {case['id']:<16} answer={answer_text.strip()[:18]!r:<20}"
              f" verdict={v:<16} dep={dep_s:<7} ratio={ratio_s:<9} -> {outcome} ({wall}s)")

    graded = [r for r in results if r.get("ok") and r["outcome"] in ("agree", "disagree")]
    agree = [r for r in graded if r["outcome"] == "agree"]
    ungradeable = [r for r in results if r.get("ok") and r["outcome"] == "ungradeable"]
    inconclusive = [r for r in results if r.get("ok") and r["outcome"] == "inconclusive"]
    descriptive = [r for r in results if r.get("ok") and r["outcome"] == "descriptive"]
    blocked = [r for r in results if not r.get("ok")]

    by_cat: dict = {}
    for r in graded:
        c = by_cat.setdefault(r["category"], {"agree": 0, "n": 0})
        c["n"] += 1
        c["agree"] += (r["outcome"] == "agree")

    summary = {
        "tag": tag, "engine": engine, "n_cases": len(CASES),
        "n_graded": len(graded), "n_agree": len(agree),
        "agreement": round(len(agree) / len(graded), 3) if graded else None,
        "n_ungradeable": len(ungradeable),
        "ungradeable_ids": [r["id"] for r in ungradeable],
        "n_inconclusive": len(inconclusive),
        "inconclusive_ids": [r["id"] for r in inconclusive],
        "n_descriptive": len(descriptive),
        "descriptive": [{"id": r["id"], "verdict": r["verdict"], "dependence": r["dependence"]}
                        for r in descriptive],
        "n_blocked": len(blocked),
        "disagreements": [{"id": r["id"], "answer": r["answer"], "verdict": r["verdict"],
                           "dependence": r["dependence"], "ratio": r["best_control_ratio"],
                           "span": r["span_tokens"]}
                          for r in graded if r["outcome"] == "disagree"],
        "by_category": {k: f"{v['agree']}/{v['n']}" for k, v in sorted(by_cat.items())},
        "wall_s_total": round(time.time() - t0, 1),
        "budget": vars(budget),
        "note_inconclusive_guard": (
            "the INCONCLUSIVE guard fired on real prompts for the first time" if inconclusive
            else "the INCONCLUSIVE guard did NOT fire on any of these cases -- it remains "
                 "unexercised on real prompts; its discrimination is still unproven"),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"provenance_battery_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    print()
    print(f"=== provenance battery [{tag}] ===")
    print(f"agreement: {summary['n_agree']}/{summary['n_graded']}"
          f" ({summary['agreement']})   ungradeable(model failed the task): "
          f"{summary['n_ungradeable']} {summary['ungradeable_ids']}")
    print(f"INCONCLUSIVE: {summary['n_inconclusive']} {summary['inconclusive_ids']}")
    if summary["n_descriptive"]:
        print("descriptive (no model-independent expectation; see _descriptive's docstring):")
        for d in summary["descriptive"]:
            print(f"  {d['id']}: verdict={d['verdict']} dep={d['dependence']}")
    print(f"by category: {summary['by_category']}")
    for d in summary["disagreements"]:
        print(f"  DISAGREE {d['id']}: answer={d['answer']!r} verdict={d['verdict']} "
              f"dep={d['dependence']} ratio={d['ratio']} span={d['span']}")
    print(summary["note_inconclusive_guard"])
    print(f"wrote {out_path}  ({summary['wall_s_total']}s)")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", required=True, help="short model tag for the output file, e.g. qwen2.5-7b")
    ap.add_argument("--max-span", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "runs", "experiments"))
    args = ap.parse_args()

    if not available(args.engine):
        print(f"engine at {args.engine} lacks attn_knockout -- start clozn-server with "
              "--no-flash-attn (see module docstring)", file=sys.stderr)
        return 2
    budget = ProvenanceBudget(max_span=args.max_span, candidates=args.candidates)
    run_battery(args.engine, args.tag, budget, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
