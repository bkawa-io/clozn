"""attn_vs_causal.py -- R1's attention-heatmap vs causal-rank head-to-head.

The claim to be tested with a number instead of an assertion: "attention weight is correlational
-- it is not the same thing as influence." Every attention-heatmap product implicitly treats the
attention row as a relevance ranking over context positions. This script measures, on the same
prompts, on the same engine, how well that ranking agrees with the CAUSAL ranking obtained by
attention knockout (zero the edge, renormalize, re-score the answer -- the validated provenance
primitive, battery 41/41 across two families).

Per prompt:
  * attention ranking  -- /score with attn_capture at the final prompt position: the engine
    returns the head-averaged post-softmax row per layer; positions are ranked by their
    layer-mean attention mass (exactly what a heatmap viewer would rank by).
  * causal ranking     -- per position p: all-layer renormalized knockout of final->p, delta of
    the answer token's logprob (trace_provenance's singles scan, run explicitly here).
  * agreement          -- Spearman rank correlation over the shared positions, top-1 / top-3
    overlap, and the sink diagnostic: position 0's attention rank vs causal rank (the measured
    trap: the sink dominates attention mass while carrying ~no causal weight once renormalized).

Needs a clozn-server built with attn_capture (2026-07-22) and started with --no-flash-attn.
Writes runs/experiments/attn_vs_causal_<tag>.json. Honesty scope: head-MEAN attention only (a
per-head best-case is a different, harder-to-defend baseline: a viewer looking at 28x28 maps is
not using any single head either); one model family per run; the causal side is the same
primitive the verdicts ship on, so this is heatmap-vs-shipped-receipt, not heatmap-vs-truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

PROMPTS = [
    ("kv", "The box is blue. The lamp is red. The cup is green. The color of the box is"),
    ("induction", "The wizard Zorblax cast a spell. Everyone cheered for the wizard"),
    ("doc", "Excerpt from a datasheet: 'Florpium-9 melts at 214 degrees Celsius.' According to "
            "the datasheet, the melting point of Florpium-9 in degrees Celsius is"),
    ("factual", "The capital of France is"),
    ("distractor", "Kyoto was the capital of Japan for over a thousand years. Since the Meiji "
                   "era, however, the government has been located elsewhere. The modern capital "
                   "of Japan is"),
    ("negation", "The museum is open every day except Monday. The one day you cannot visit is"),
    ("arith", "A ticket costs 12 dollars and a program costs 5 dollars. Together, one ticket "
              "and one program cost, in dollars,"),
    ("long", "Minutes of the residents' association: The garden fence will be repainted in "
             "April. Parking permits renew in June. The gate code was changed yesterday to 5261 "
             "for security reasons. The newsletter needs a new editor. To open the gate, enter "
             "the code"),
]


def post(engine, path, body, timeout=900):
    req = urllib.request.Request(engine.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def spearman(xs, ys):
    """Spearman rho without scipy: rank both, Pearson on ranks (ties get mean rank)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = mean_rank
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def run_case(engine, n_layer, category, prompt):
    base = post(engine, "/score", {"prompt": prompt, "continuation": " ", "topk": 1})
    n_p = int(base["n_prompt"])
    final = n_p - 1
    # the model's own greedy answer token, teacher-forced for every arm
    gen = post(engine, "/v1/completions", {"prompt": prompt, "max_tokens": 2, "temperature": 0})
    answer = gen["choices"][0]["text"]
    scored = post(engine, "/score", {"prompt": prompt, "continuation": answer, "topk": 1,
                                     "attn_capture": {"query": final}})
    base_lp = float(scored["tokens"][0]["logprob"])
    cont_ids = [int(scored["tokens"][0]["id"])]
    attn_rows = scored.get("attn_rows") or {}
    if not attn_rows:
        return {"category": category, "error": "no attn_rows returned (engine too old or "
                                               "flash attention on)"}

    # ATTENTION side: layer-mean of head-mean rows, positions 0..final-1
    attn_mass = [0.0] * final
    n_used = 0
    for _layer, row in attn_rows.items():
        if len(row) < final:
            continue
        for p in range(final):
            attn_mass[p] += float(row[p])
        n_used += 1
    if n_used == 0:
        return {"category": category, "error": "attn rows shorter than prompt"}
    attn_mass = [v / n_used for v in attn_mass]

    # CAUSAL side: all-layer renormalized knockout singles, positions 0..final-1
    causal = []
    for p in range(final):
        specs = [{"layer": L, "queries": [final], "keys": [p], "renormalize": True}
                 for L in range(n_layer)]
        r = post(engine, "/score", {"prompt": prompt, "continuation_ids": cont_ids,
                                    "topk": 0, "attn_knockout": specs})
        causal.append(base_lp - float(r["tokens"][0]["logprob"]))

    rho = spearman(attn_mass, causal)
    top_attn = sorted(range(final), key=lambda p: -attn_mass[p])
    top_causal = sorted(range(final), key=lambda p: -causal[p])
    toks = post(engine, "/harvest", {"text": prompt, "layer": 1})["tokens"]
    return {
        "category": category, "prompt": prompt, "answer": answer.strip()[:20],
        "n_positions": final, "spearman_rho": round(rho, 4),
        "top1_agree": top_attn[0] == top_causal[0],
        "top3_overlap": len(set(top_attn[:3]) & set(top_causal[:3])),
        "attn_top3": [{"pos": p, "tok": toks[p], "mass": round(attn_mass[p], 4)}
                      for p in top_attn[:3]],
        "causal_top3": [{"pos": p, "tok": toks[p], "delta": round(causal[p], 4)}
                        for p in top_causal[:3]],
        "sink": {"attn_rank_of_pos0": top_attn.index(0), "causal_rank_of_pos0":
                 top_causal.index(0), "attn_mass_pos0": round(attn_mass[0], 4),
                 "causal_delta_pos0": round(causal[0], 4)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    health = json.loads(urllib.request.urlopen(args.engine + "/health", timeout=10).read())
    if not health.get("capabilities", {}).get("attn_knockout"):
        print("engine lacks attn_knockout (--no-flash-attn required)", file=sys.stderr)
        return 2
    n_layer = int(health["n_layer"])

    results, t0 = [], time.time()
    for i, (cat, prompt) in enumerate(PROMPTS):
        t = time.time()
        r = run_case(args.engine, n_layer, cat, prompt)
        r["wall_s"] = round(time.time() - t, 1)
        results.append(r)
        if "error" in r:
            print(f"  [{i+1}/{len(PROMPTS)}] {cat:<11} ERROR: {r['error']}")
        else:
            print(f"  [{i+1}/{len(PROMPTS)}] {cat:<11} rho={r['spearman_rho']:<8} "
                  f"top1_agree={str(r['top1_agree']):<6} top3_overlap={r['top3_overlap']}/3 "
                  f"sink attn_rank={r['sink']['attn_rank_of_pos0']} "
                  f"causal_rank={r['sink']['causal_rank_of_pos0']} ({r['wall_s']}s)")

    ok = [r for r in results if "error" not in r]
    summary = {
        "tag": args.tag, "n_cases": len(ok),
        "mean_spearman": round(sum(r["spearman_rho"] for r in ok) / len(ok), 4) if ok else None,
        "top1_agreement": f"{sum(r['top1_agree'] for r in ok)}/{len(ok)}",
        "mean_top3_overlap": round(sum(r["top3_overlap"] for r in ok) / len(ok), 2) if ok else None,
        "sink_attn_ranks": [r["sink"]["attn_rank_of_pos0"] for r in ok],
        "sink_causal_ranks": [r["sink"]["causal_rank_of_pos0"] for r in ok],
        "wall_s": round(time.time() - t0, 1),
        "scope": "head-mean, layer-mean attention vs all-layer renormalized knockout; "
                 "one model; the causal side is the shipped provenance primitive",
    }
    out = os.path.join(REPO, "runs", "experiments", f"attn_vs_causal_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\n=== attn-vs-causal [{args.tag}] ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
