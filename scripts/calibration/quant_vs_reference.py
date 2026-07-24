"""quant_vs_reference.py -- 4.3 sub-task 1: the researcher objection-killer.

"Can I trust interpretability on a QUANTIZED model?" clozn reads activations off Q4_K_M GGUFs; the
honest answer is a MEASUREMENT: how faithfully does a quant preserve the FULL-PRECISION reference's
activations (what interp reads) and its next-token behavior?

Qwen2.5-7B, sequential-VRAM (16GB card): capture the FP bf16 reference (torch, ~14GB), FREE it, then
each GGUF quant (Q8/Q4/Q2, via the C++ engine) in turn. Compare the last-position residual at a few
mid layers (cosine -- what a lens/tracer actually reads) + the next-token argmax (behavioral).

Q8 is the built-in SANITY ANCHOR: near-lossless, so FP-vs-Q8 cosine ~0.99+ confirms the layer
alignment (engine l_out-<il> == torch hidden_states[il+1]). Q4 is the shipped quant -- its number is
the actual qualification. Q2 shows the degradation floor.

Writes runs/experiments/quant_vs_reference_7b.json.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [14, 18, 21]
GGUF_DIR = os.path.expanduser("~/.clozn/models")
QUANTS = ["Q8_0", "Q4_K_M", "Q2_K"]     # near-lossless anchor -> shipped -> aggressive
PORT = 8090

PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "def fib(n): return n if n < 2 else fib(n-1) +",
    "The mitochondria is the powerhouse of the",
    "In 1969, humans first landed on the",
    "The opposite of hot is",
    "To make a cup of tea, first boil the",
    "The three primary colors are red, blue, and",
    "E equals m c",
    "The largest ocean on Earth is the",
    "A group of wolves is called a",
    "The chemical symbol for gold is",
]


def cosine(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def capture_fp():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[FP] loading bf16 reference ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    model = AutoModelForCausalLM.from_pretrained(MODEL_HF, dtype=torch.bfloat16).to("cuda").eval()
    ref = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to("cuda")
            out = model(ids, output_hidden_states=True)
            top1 = int(out.logits[0, -1].argmax())
            resid = {str(L): out.hidden_states[L + 1][0, -1].float().cpu().numpy().tolist() for L in LAYERS}
            ref.append({"prompt": p, "n_tok": int(ids.shape[1]), "top1": top1, "resid": resid})
    del model, out, tok
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)
    print("[FP] captured %d prompts, freed VRAM (%.1f GB free)" %
          (len(ref), torch.cuda.mem_get_info()[0] / 1e9), flush=True)
    return ref


def capture_quant(gguf, ref):
    import urllib.request
    import subprocess
    from clozn.cli.engine_process import spawn_engine
    from clozn.cli.commands.models import _flags_for
    subprocess.run(["taskkill", "/F", "/IM", "cloze-server.exe"], capture_output=True)  # clean slate
    time.sleep(1)
    proc, health, _ = spawn_engine(gguf, PORT, _flags_for(gguf), prefer_gpu=True)
    base = f"http://127.0.0.1:{PORT}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    rows = []
    try:
        for r in ref:
            p = r["prompt"]
            resp = post("/score", {"prompt": p, "continuation": " x", "topk": 5})  # n_p + next-token
            n_p = int(resp["n_prompt"])
            q_top1 = int(resp["tokens"][0]["topk"][0]["id"])   # engine's next-token argmax
            respc = post("/score", {"prompt": p, "continuation": " x",
                                    "capture": {"layers": LAYERS, "positions": [n_p - 1]}})
            cap = respc["captured"]
            cos = {L: cosine(r["resid"][str(L)], cap[str(L)][str(n_p - 1)]) for L in LAYERS}
            rows.append({"n_p_engine": n_p, "n_tok_fp": r["n_tok"],
                         "cos": cos, "top1_agree": q_top1 == r["top1"]})
    finally:
        subprocess.run(["taskkill", "/F", "/IM", "cloze-server.exe"], capture_output=True)
    time.sleep(2)
    return rows


def main():
    cache = os.path.join(REPO, "runs", "experiments", "_qvr_fp_cache.json")
    if os.path.exists(cache):
        print("[FP] loading cached reference (skip torch reload)", flush=True)
        ref = json.load(open(cache))
    else:
        ref = capture_fp()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(ref, open(cache, "w"))
    result = {"model": MODEL_HF, "layers": LAYERS, "n_prompts": len(PROMPTS), "quants": {}}
    for q in QUANTS:
        matches = glob.glob(os.path.join(GGUF_DIR, f"*7B*{q}*.gguf"))
        if not matches:
            print(f"[{q}] GGUF not found, skipping", flush=True); continue
        rows = capture_quant(matches[0], ref)
        per_layer = {L: round(float(np.mean([r["cos"][L] for r in rows])), 4) for L in LAYERS}
        argmax_agree = round(sum(r["top1_agree"] for r in rows) / len(rows), 3)
        result["quants"][q] = {"cosine_by_layer": per_layer, "argmax_agree": argmax_agree}
        print(f"[{q}] cosine {per_layer} | next-token argmax agree {argmax_agree}", flush=True)

    # honest reading, anchored on Q8
    q8 = result["quants"].get("Q8_0", {}).get("cosine_by_layer", {})
    q8_ok = q8 and min(q8.values()) > 0.97
    result["alignment_sane"] = bool(q8_ok)
    result["reading"] = (
        ("layer alignment CONFIRMED by the Q8 anchor (cosine > 0.97). "
         if q8_ok else
         "WARNING: Q8 anchor cosine below 0.97 -- alignment or metric suspect, read with care. ") +
        "The Q4 numbers are the actual qualification: interp reads the shipped GGUF's residuals, and "
        "this is how close they are to the full-precision reference the research assumes.")
    out = os.path.join(REPO, "runs", "experiments", "quant_vs_reference_7b.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    print("\n=== quant-vs-reference (7B) ===")
    print(" ", result["reading"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
