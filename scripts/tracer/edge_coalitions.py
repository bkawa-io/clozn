"""edge_coalitions.py -- the molecules tail, run with the SHARP instrument (attention severance).

The molecules program's verdict: under residual mean-ablation, greedily-optimized position
coalitions do NOT beat random same-size coalitions (0.93x) -- no privileged molecule. But the one
unit that ever beat its controls 100x+ (the provenance greedy SPAN) uses a DIFFERENT instrument:
attention-edge severance of CONTIGUOUS input positions. Open question recorded in molecules.py's
verdict: do coalitions of NON-contiguous edge cuts beat random matched sets? I.e. is the molecule
real under the sharp instrument even though it dissolved under the blunt one?

Method (all-layer edge severance into every query, mirroring provenance's knockout): on the
distributed cases (KV / induction / multi-hop), greedily build a SET of prompt positions whose
attention-severance most drops the answer's logprob; compare the k-set's joint delta vs 8 random
same-size position sets (excluding the readout row). Also record the CONTIGUOUS greedy span baseline
for the same case, so the receipt shows atom < set <= span or whatever actually holds.

Needs the --no-flash-attn engine build. Writes runs/experiments/edge_coalitions_7b.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
GGUF = os.path.expanduser("~/.clozn/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
PORT = 8091
TARGET_FRAC = 0.8

CASES = [
    "The box is blue. The lamp is red. The cup is green. The color of the box is",
    "Anna has the key. Ben has the map. Carl has the torch. The person with the map is",
    "The wizard Zorblax cast a spell. Everyone cheered for the wizard",
    "Paris is in France. Tokyo is in Japan. Cairo is in Egypt. Tokyo is in",
    "The red door leads to the vault. The blue door leads to the exit. The vault is behind the",
]


def main():
    import urllib.request
    from clozn.cli.engine_process import spawn_engine
    from clozn.cli.commands.models import _flags_for

    subprocess.run(["taskkill", "/F", "/IM", "cloze-server.exe"], capture_output=True); time.sleep(1)
    # boot with knockout support (no flash attn) via the flags dict's extra_args passthrough
    flags = _flags_for(GGUF)
    flags["extra_args"] = ["--no-flash-attn"]
    proc, health, _ = spawn_engine(GGUF, PORT, flags, prefer_gpu=True)
    n_layer = int(health["n_layer"])
    base_url = f"http://127.0.0.1:{PORT}"

    def post(body):
        req = urllib.request.Request(base_url + "/score", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    def sever(prompt, cont_ids, keys, n_p):
        """Delta from severing reading of `keys` at all layers, for every query row from the
        first position after min(keys) through the final (answer-predicting) row."""
        if not keys:
            return 0.0
        q0 = min(keys) + 1
        queries = list(range(q0, n_p))          # rows that could read the severed keys
        if not queries:
            return 0.0
        ks = [{"layer": L, "queries": queries, "keys": sorted(keys), "renormalize": True}
              for L in range(n_layer)]
        r = post({"prompt": prompt, "continuation_ids": cont_ids, "topk": 0, "attn_knockout": ks})
        return base_lp - float(r["tokens"][0]["logprob"])

    rng = np.random.default_rng(0)
    results = []
    try:
        for prompt in CASES:
            gen_req = urllib.request.Request(base_url + "/v1/completions",
                data=json.dumps({"prompt": prompt, "max_tokens": 2, "temperature": 0}).encode(),
                headers={"Content-Type": "application/json"})
            true_cont = json.loads(urllib.request.urlopen(gen_req, timeout=120).read())["choices"][0]["text"]
            b = post({"prompt": prompt, "continuation": true_cont, "topk": 0})
            n_p = int(b["n_prompt"])
            global base_lp
            base_lp = float(b["tokens"][0]["logprob"])
            cont_ids = [int(b["tokens"][0]["id"])]
            pool = list(range(1, n_p - 1))          # position 0 (BOS-ish) + readout row excluded

            # greedy NON-CONTIGUOUS set build
            chosen, remaining, traj = [], list(pool), []
            cur = 0.0
            while remaining and len(chosen) < 8:
                best, best_d = None, None
                for s in remaining:
                    d = sever(prompt, cont_ids, chosen + [s], n_p)
                    if best_d is None or d > best_d:
                        best, best_d = s, d
                if best_d - cur <= 0.05 and chosen:
                    break
                chosen.append(best); remaining.remove(best); traj.append(best_d); cur = best_d
            peak = cur
            k = next((i + 1 for i, v in enumerate(traj) if v >= TARGET_FRAC * peak), len(traj))
            min_set, min_delta = chosen[:k], (traj[k - 1] if traj else 0.0)

            # matched random-k control
            ctl = []
            for _ in range(8):
                picks = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
                ctl.append(abs(sever(prompt, cont_ids, [int(x) for x in picks], n_p)))
            ctl_max = max(ctl) if ctl else 0.0
            sep = abs(min_delta) / ctl_max if ctl_max > 1e-9 else None

            # contiguous greedy SPAN baseline (same budget k): best window of length k
            best_span = 0.0
            for s0 in range(1, n_p - 1 - k + 1):
                d = sever(prompt, cont_ids, list(range(s0, s0 + k)), n_p)
                best_span = max(best_span, d)

            row = {"prompt": prompt, "answer": true_cont.strip(), "n_p": n_p, "k": k,
                   "set_positions": min_set, "set_delta": round(min_delta, 3),
                   "random_k_max": round(ctl_max, 3),
                   "separation_vs_random": round(sep, 2) if sep else None,
                   "best_contiguous_span_delta": round(best_span, 3)}
            results.append(row)
            print(f"{prompt[:34]!r:<36} k={k} set {min_delta:+.2f} vs rand {ctl_max:.2f} "
                  f"({row['separation_vs_random']}x) | best span(k) {best_span:+.2f}", flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/IM", "cloze-server.exe"], capture_output=True)

    seps = [r["separation_vs_random"] for r in results if r["separation_vs_random"]]
    beats = sum(1 for s in seps if s >= 2.0)
    summary = {
        "cases": len(results), "beats_2x": f"{beats}/{len(seps)}",
        "mean_separation": round(float(np.mean(seps)), 2) if seps else None,
        "reading": (
            f"SHARP-instrument coalitions: greedy edge-severance sets beat matched random sets >=2x in "
            f"{beats}/{len(seps)} cases (mean {round(float(np.mean(seps)),2) if seps else 0}x). "
            "Compare each row's set_delta vs best_contiguous_span_delta: if the span matches the free "
            "set, contiguity was never the constraint -- the SPAN remains the honest unit. If the free "
            "set clearly beats the span, non-contiguous molecules are real under severance.")
    }
    out = os.path.join(REPO, "runs", "experiments", "edge_coalitions_7b.json")
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print("\n=== edge coalitions (sharp instrument) ===")
    print(" ", summary["reading"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
