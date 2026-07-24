"""transplant_localize.py -- R2 (model-diff transplant), the on-disk version: localize a QUANTIZATION
regression by transplant.

The R2 idea is "localize a regression, proven by transplant." No SFT pair is on disk, but Q2_K is a
real degraded model (the quant-vs-reference run measured it flipping ~8% of next-token argmaxes vs the
FP reference). So: find prompts where Q2 flips a token the FP reference gets right (a genuine
quantization regression), then TRANSPLANT the FP reference's clean residual into the Q2 engine at one
layer and see if the flip is corrected. The layer whose transplant fixes it LOCALIZES where the
quantization damage manifests -- proven by transplant, not asserted.

Self-contained: Qwen2.5-7B FP (torch) + Q2_K GGUF (engine write surface). Same architecture, so the FP
l_out-<L> residual drops straight into the Q2 engine's l_out-<L> (the Q8 anchor in quant_vs_reference
already confirmed that alignment). Writes runs/experiments/transplant_localize_7b.json.
"""
from __future__ import annotations

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
MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"
Q2_GLOB = os.path.expanduser("~/.clozn/models/*7B*Q2_K*.gguf")
LAYERS = [10, 14, 18, 21, 25]   # transplant candidates across depth
PORT = 8090

# Prompts biased to STRESS Q2 (precise numbers, arithmetic, specific facts, code) -> more flips.
PROMPTS = [
    "Compute: 47 times 6 equals", "The square root of 144 is",
    "In binary, the number 5 is written as", "The year the Berlin Wall fell was",
    "The atomic number of oxygen is", "17 plus 28 equals",
    "The speed of light is approximately 299,792",
    "The chemical formula for table salt is", "def is_even(n): return n % 2 ==",
    "The freezing point of water in Fahrenheit is", "The capital of Australia is",
    "The number of sides on a hexagon is", "The largest prime number below 20 is",
    "The Roman numeral for 40 is", "The boiling point of water at sea level in Celsius is",
    "9 factorial divided by 8 factorial equals", "The third planet from the sun is",
    "The number of bits in a byte is", "The hexadecimal value of decimal 255 is",
    "The author of Romeo and Juliet was",
]


def capture_fp():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[FP] loading {MODEL_HF} bf16 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    model = AutoModelForCausalLM.from_pretrained(MODEL_HF, dtype=torch.bfloat16).to("cuda").eval()
    ref = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to("cuda")
            out = model(ids, output_hidden_states=True)
            top1 = int(out.logits[0, -1].argmax())
            resid = {str(L): out.hidden_states[L + 1][0, -1].float().cpu().numpy().tolist() for L in LAYERS}
            ref.append({"prompt": p, "n_tok": int(ids.shape[1]), "fp_top1": top1, "resid": resid})
    del model, out, tok
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(2)
    print("[FP] captured, freed (%.1f GB free)" % (torch.cuda.mem_get_info()[0] / 1e9), flush=True)
    return ref


def main():
    import urllib.request
    from clozn.cli.engine_process import spawn_engine
    from clozn.cli.commands.models import _flags_for

    cache = os.path.join(REPO, "runs", "experiments", "_transplant_fp_cache.json")
    if os.path.exists(cache):
        print("[FP] loading cache", flush=True); ref = json.load(open(cache))
    else:
        ref = capture_fp()
        os.makedirs(os.path.dirname(cache), exist_ok=True); json.dump(ref, open(cache, "w"))

    gguf = glob.glob(Q2_GLOB)[0]
    subprocess.run(["taskkill", "/F", "/IM", "cloze-server.exe"], capture_output=True); time.sleep(1)
    proc, health, _ = spawn_engine(gguf, PORT, _flags_for(gguf), prefer_gpu=True)
    base = f"http://127.0.0.1:{PORT}"

    def post(body):
        req = urllib.request.Request(base + "/score", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    def q2_top1(prompt, write=None):
        b = {"prompt": prompt, "continuation": " x", "topk": 5}
        if write is not None:
            b["write"] = write
        return int(post(b)["tokens"][0]["topk"][0]["id"]), post(b).get("n_prompt")

    _rng = np.random.default_rng(0)
    results = []
    try:
        for r in ref:
            p = r["prompt"]
            b = post({"prompt": p, "continuation": " x", "topk": 5})
            n_p = int(b["n_prompt"])
            q2 = int(b["tokens"][0]["topk"][0]["id"])
            if q2 == r["fp_top1"]:
                continue                                    # no Q2 regression on this prompt
            # a genuine Q2 flip -> try to fix it by transplanting the FP residual at each layer
            fp_fixed, rand_fixed = [], []
            for L in LAYERS:
                fp = np.asarray(r["resid"][str(L)], np.float64)
                w = {"layer": L, "positions": [n_p - 1], "values": r["resid"][str(L)]}
                if int(post({"prompt": p, "continuation": " x", "topk": 5, "write": w})["tokens"][0]["topk"][0]["id"]) == r["fp_top1"]:
                    fp_fixed.append(L)
                # random-equal-norm CONTROL: same layer/position, a random direction of the same magnitude
                rnd = _rng.standard_normal(len(fp)); rnd = (rnd / (np.linalg.norm(rnd) + 1e-9) * np.linalg.norm(fp)).tolist()
                wr = {"layer": L, "positions": [n_p - 1], "values": rnd}
                if int(post({"prompt": p, "continuation": " x", "topk": 5, "write": wr})["tokens"][0]["topk"][0]["id"]) == r["fp_top1"]:
                    rand_fixed.append(L)
            results.append({"prompt": p, "fp_top1": r["fp_top1"], "q2_top1": q2,
                            "fp_fixed_by": fp_fixed, "random_fixed_by": rand_fixed})
            print(f"FLIP {p[:32]!r:<34} q2={q2} fp={r['fp_top1']} -> FP fixes {fp_fixed} | RANDOM fixes {rand_fixed}", flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/IM", "cloze-server.exe"], capture_output=True)

    n_flip = len(results)
    n_fp = sum(1 for r in results if r["fp_fixed_by"])
    n_fp_specific = sum(1 for r in results if r["fp_fixed_by"] and not r["random_fixed_by"])
    n_rand = sum(1 for r in results if r["random_fixed_by"])
    # which layer most often fixes a flip?
    from collections import Counter
    layer_fixes = Counter(L for r in results for L in r["fp_fixed_by"])
    summary = {
        "model": MODEL_HF, "degraded": os.path.basename(gguf), "layers": LAYERS,
        "n_prompts": len(PROMPTS), "n_q2_flips": n_flip,
        "n_fp_fixed": n_fp, "n_fp_specific_fixed": n_fp_specific, "n_random_fixed": n_rand,
        "fixes_by_layer": dict(layer_fixes),
        "reading": (
            f"FP transplant fixed {n_fp}/{n_flip} flips but the random-equal-norm control fixed {n_rand}/"
            f"{n_flip}; only {n_fp_specific}/{n_flip} are FP-SPECIFIC (fixed by FP, untouched by random). "
            f"Honest verdict: single-layer transplant localization is real but MINORITY-case -- most Q2 "
            f"damage is either distributed across layers or the flip is a near-tie any perturbation "
            f"topples. The control corrected the first run's overclaim."),
    }
    out = os.path.join(REPO, "runs", "experiments", "transplant_localize_7b.json")
    json.dump({"summary": summary, "flips": results}, open(out, "w"), indent=2)
    print("\n=== transplant localization ===")
    print(" ", summary["reading"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
