"""quant_vs_reference.py -- 4.3 sub-task 1: "can I trust interp on a QUANTIZED model?", measured.

clozn reads activations off Q4_K_M GGUFs; this quantifies how faithful that is to the FULL-PRECISION
reference the research assumes. Sequential-VRAM on a 16GB card: capture the FP bf16 reference (torch),
FREE it, then each GGUF quant (via the C++ engine). Compares the residual at ALL prompt positions
(cosine -- what a lens/tracer actually reads) + the next-token argmax (behavioral).

Q8 is the SANITY ANCHOR (near-lossless -> cosine ~0.999 confirms the layer alignment). Q4 is the
shipped quant -- its number is the actual qualification. Q2 shows the degradation floor.

Parametric: --model + --pattern + --layers so the SAME script runs the Qwen 7B all-positions
qualification AND the Llama second-family cross-architecture check. FP capture is cached per model.

Writes runs/experiments/quant_vs_reference_<tag>.json.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
GGUF_DIR = os.path.expanduser("~/.clozn/models")
PORT = 8090

PROMPTS = [
    "The capital of France is", "Water is made of hydrogen and",
    "def fib(n): return n if n < 2 else fib(n-1) +", "The mitochondria is the powerhouse of the",
    "In 1969, humans first landed on the", "The opposite of hot is",
    "To make a cup of tea, first boil the", "The three primary colors are red, blue, and",
    "E equals m c", "The largest ocean on Earth is the",
    "A group of wolves is called a", "The chemical symbol for gold is",
]


def cosine(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def capture_fp(model_name, layers):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[FP] loading {model_name} bf16 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    ref = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to("cuda")
            out = model(ids, output_hidden_states=True)
            top1 = int(out.logits[0, -1].argmax())
            # ALL positions per layer: [seq, d]
            resid = {str(L): out.hidden_states[L + 1][0].float().cpu().numpy().tolist() for L in layers}
            ref.append({"prompt": p, "n_tok": int(ids.shape[1]), "top1": top1, "resid": resid})
    del model, out, tok
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(2)
    print("[FP] captured %d prompts, freed (%.1f GB free)" % (len(ref), torch.cuda.mem_get_info()[0] / 1e9),
          flush=True)
    return ref


def capture_quant(gguf, ref, layers):
    import urllib.request
    from clozn.cli.engine_process import spawn_engine
    from clozn.cli.commands.models import _flags_for
    subprocess.run(["taskkill", "/F", "/IM", "clozn-server.exe"], capture_output=True); time.sleep(1)
    proc, health, _ = spawn_engine(gguf, PORT, _flags_for(gguf), prefer_gpu=True)
    base = f"http://127.0.0.1:{PORT}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    per_layer = {L: [] for L in layers}   # cosine at every (prompt, position)
    argmax_hits = 0
    try:
        for r in ref:
            p = r["prompt"]
            resp = post("/score", {"prompt": p, "continuation": " x", "topk": 5})
            n_p = int(resp["n_prompt"])
            if int(resp["tokens"][0]["topk"][0]["id"]) == r["top1"]:
                argmax_hits += 1
            positions = list(range(min(n_p, r["n_tok"])))
            respc = post("/score", {"prompt": p, "continuation": " x",
                                    "capture": {"layers": layers, "positions": positions}})
            cap = respc["captured"]
            for L in layers:
                for pos in positions:
                    per_layer[L].append(cosine(r["resid"][str(L)][pos], cap[str(L)][str(pos)]))
    finally:
        subprocess.run(["taskkill", "/F", "/IM", "clozn-server.exe"], capture_output=True)
    time.sleep(2)
    stats = {L: {"mean_cos": round(float(np.mean(per_layer[L])), 4),
                 "min_cos": round(float(np.min(per_layer[L])), 4),
                 "n_positions": len(per_layer[L])} for L in layers}
    return stats, round(argmax_hits / len(ref), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pattern", default="*7B*{q}*.gguf", help="gguf glob; {q} filled per quant")
    ap.add_argument("--layers", default="14,18,21")
    ap.add_argument("--quants", default="Q8_0,Q4_K_M,Q2_K")
    ap.add_argument("--tag", default="7b")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    quants = args.quants.split(",")

    safe = args.tag
    cache = os.path.join(REPO, "runs", "experiments", f"_qvr_fp_cache_{safe}.json")
    if os.path.exists(cache):
        print("[FP] loading cached reference", flush=True)
        ref = json.load(open(cache))
    else:
        ref = capture_fp(args.model, layers)
        os.makedirs(os.path.dirname(cache), exist_ok=True); json.dump(ref, open(cache, "w"))

    result = {"model": args.model, "layers": layers, "n_prompts": len(PROMPTS),
              "scope": "ALL prompt positions", "quants": {}}
    for q in quants:
        matches = glob.glob(os.path.join(GGUF_DIR, args.pattern.replace("{q}", q)))
        if not matches:
            print(f"[{q}] GGUF not found ({args.pattern.replace('{q}', q)}), skipping", flush=True); continue
        stats, argmax = capture_quant(matches[0], ref, layers)
        result["quants"][q] = {"cosine_by_layer": {L: stats[L] for L in layers}, "argmax_agree": argmax}
        line = " ".join(f"L{L}:{stats[L]['mean_cos']}(min {stats[L]['min_cos']})" for L in layers)
        print(f"[{q}] cosine[mean(min)] {line} | argmax {argmax}", flush=True)

    q8 = result["quants"].get("Q8_0", {}).get("cosine_by_layer", {})
    q8_ok = q8 and min(v["mean_cos"] for v in q8.values()) > 0.97
    result["alignment_sane"] = bool(q8_ok)
    out = os.path.join(REPO, "runs", "experiments", f"quant_vs_reference_{safe}.json")
    json.dump(result, open(out, "w"), indent=2)
    print("\n=== quant-vs-reference (%s, all positions) ===" % safe)
    print("  alignment %s (Q8 anchor). Q4 = the qualification: interp reads the shipped GGUF's residuals; "
          "this is how close they are to full precision, across EVERY position." %
          ("CONFIRMED" if q8_ok else "SUSPECT"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
