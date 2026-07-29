"""ffn_hook_probe.py -- live HTTP proof for the new ffn_out-<il> hook and the 2026-07-28
head_write/ffn_write honesty fix (branch feat/ffn-hook), exercised through the REAL /score route
(routes_whitebox.cpp), not just the C++ adapter (engine/core/tests/test_ffn_hook.cpp covers that half).

Proves, with real measured numbers against a live server:
  1. ffn_write composition: baseline / L1-only / L2-only / BOTH -- "both" must differ from every
     single-site arm (the same additive-not-overwrite shape kqv_out/head_write already have).
  2. ffn_write composes with head_write at a different layer (both additive hooks, should stack).
  3. the honesty fix: the EXACT two bug repros from scripts/spike/additive_writes_probe.py
     (malformed head_write values length, out-of-range head_write layer) now return 400, not a 200
     with a lying "applied: true". Same class of malformed spec for ffn_write also gets a 400.

Run: python scripts/spike/ffn_hook_probe.py
Writes runs/experiments/ffn_hook_probe.json (throwaway artifact, not a product output).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

PORT = 8098
MODEL_GLOB = os.path.expanduser("~/.clozn/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
PROMPT = "The capital of France is"
TOPK = 10


def post(base, body):
    req = urllib.request.Request(base + "/score", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return {"_status": r.status, **json.loads(r.read())}
    except urllib.error.HTTPError as e:
        return {"_status": e.code, "_body": json.loads(e.read())}


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def score_baseline(base):
    return post(base, {"prompt": PROMPT, "continuation": " Paris", "topk": TOPK})


def top_summary(resp, label):
    if "_body" in resp:
        return {"label": label, "error": resp}
    tok = resp["tokens"][0]
    return {"label": label, "n_prompt": resp.get("n_prompt"),
            "top1_id": tok["topk"][0]["id"], "top1_piece": tok["topk"][0]["piece"],
            "top1_logprob": tok["topk"][0]["logprob"],
            "actual_id": tok["id"], "actual_logprob": tok["logprob"]}


def identical(a, b):
    if a.get("error") or b.get("error"):
        return False
    return a["top1_id"] == b["top1_id"] and a["top1_logprob"] == b["top1_logprob"] and \
        a["actual_logprob"] == b["actual_logprob"]


def main():
    gguf_matches = glob.glob(MODEL_GLOB)
    if not gguf_matches:
        print(f"MODEL NOT FOUND: {MODEL_GLOB}")
        return 1
    gguf = gguf_matches[0]

    from clozn.cli.engine_process import spawn_engine, _terminate_process
    from clozn.cli.commands.models import _flags_for

    subprocess.run(["taskkill", "/F", "/IM", "clozn-server.exe"], capture_output=True)
    time.sleep(1)
    proc = None
    out = {"model": gguf, "prompt": PROMPT}
    try:
        proc, health, gpu = spawn_engine(gguf, PORT, _flags_for(gguf), prefer_gpu=True)
        base = f"http://127.0.0.1:{PORT}"
        h = get(base, "/health")
        n_layer, n_embd = h["n_layer"], h["n_embd"]
        out["gpu"] = gpu
        out["n_layer"] = n_layer
        out["n_embd"] = n_embd
        print(f"[boot] gpu={gpu} n_layer={n_layer} n_embd={n_embd}", flush=True)

        L1 = max(2, n_layer // 6)
        L2 = n_layer - 4
        assert 1 <= L1 < L2 <= n_layer - 2, (L1, L2, n_layer)
        out["L1"], out["L2"] = L1, L2
        print(f"[layers] L1={L1} L2={L2}", flush=True)

        b = score_baseline(base)
        n_p = b["n_prompt"]
        p = n_p - 1     # the sole scored position (logits_for = [n_p-1]), same convention as /score
        out["p"] = p
        base_summary = top_summary(b, "baseline")
        print(f"[baseline] top1={base_summary['top1_piece']!r} logprob={base_summary['top1_logprob']:.8f} "
              f"actual_logprob={base_summary['actual_logprob']:.8f}", flush=True)

        # ================= 1) ffn_write composition (baseline / L1 / L2 / both) =================
        cap = post(base, {"prompt": PROMPT, "continuation": " Paris",
                          "ffn_capture": {"layers": [L1, L2], "positions": [p]}})
        if "_body" in cap:
            print("FFN CAPTURE FAILED:", cap); return 1
        row_L1 = cap["ffn_captured"][str(L1)][str(p)]
        row_L2 = cap["ffn_captured"][str(L2)][str(p)]
        assert len(row_L1) == n_embd and len(row_L2) == n_embd
        w1 = {"layer": L1, "positions": [p], "values": [v * 1.7 + 0.05 for v in row_L1]}
        w2 = {"layer": L2, "positions": [p], "values": [v * 1.7 - 0.05 for v in row_L2]}

        arm_baseline = top_summary(score_baseline(base), "ffn_baseline")
        arm_L1 = top_summary(post(base, {"prompt": PROMPT, "continuation": " Paris",
                                         "topk": TOPK, "ffn_write": w1}), "ffn_L1_only")
        arm_L2 = top_summary(post(base, {"prompt": PROMPT, "continuation": " Paris",
                                         "topk": TOPK, "ffn_write": w2}), "ffn_L2_only")
        arm_both = top_summary(post(base, {"prompt": PROMPT, "continuation": " Paris",
                                           "topk": TOPK, "ffn_write": [w1, w2]}), "ffn_both")

        both_eq_L1 = identical(arm_both, arm_L1)
        both_eq_L2 = identical(arm_both, arm_L2)
        both_eq_baseline = identical(arm_both, arm_baseline)
        L1_moved = not identical(arm_L1, arm_baseline)
        L2_moved = not identical(arm_L2, arm_baseline)

        print("\n=== 1) ffn_write composition ===")
        for a in (arm_baseline, arm_L1, arm_L2, arm_both):
            print(f"  {a['label']:14s} top1={a['top1_piece']!r:8s} logprob={a['top1_logprob']:.8f} "
                  f"actual_logprob={a['actual_logprob']:.8f}")
        print(f"  L1_only moved from baseline: {L1_moved}")
        print(f"  L2_only moved from baseline: {L2_moved}")
        print(f"  both == L1_only : {both_eq_L1}")
        print(f"  both == L2_only : {both_eq_L2}")
        print(f"  both == baseline: {both_eq_baseline}")
        print(f"  COMPOSITION (both differs from every single-site arm): "
              f"{(not both_eq_L1) and (not both_eq_L2) and (not both_eq_baseline)}")
        out["ffn_composition"] = {
            "arms": {a["label"]: a for a in (arm_baseline, arm_L1, arm_L2, arm_both)},
            "both_eq_L1_only": both_eq_L1, "both_eq_L2_only": both_eq_L2,
            "both_eq_baseline": both_eq_baseline, "L1_moved": L1_moved, "L2_moved": L2_moved,
        }

        # ================= 2) ffn_write x head_write cross-hook composition =================
        hcap = post(base, {"prompt": PROMPT, "continuation": " Paris",
                           "head_capture": {"layers": [L2], "positions": [p], "rows": True}})
        if "_body" in hcap:
            print("HEAD CAPTURE FAILED:", hcap); return 1
        d_head, n_head = hcap["head_dims"]["d_head"], hcap["head_dims"]["n_head"]
        row = hcap["head_rows"][str(L2)][str(p)]
        hslice = [v * 1.7 - 0.05 for v in row[0:d_head]]
        hw = {"layer": L2, "head": 0, "positions": [p], "values": hslice}

        combo = post(base, {"prompt": PROMPT, "continuation": " Paris", "topk": TOPK,
                            "ffn_write": w1, "head_write": hw})
        combo_summary = top_summary(combo, "ffn_plus_head")
        ffn_only_again = arm_L1
        head_only = top_summary(post(base, {"prompt": PROMPT, "continuation": " Paris",
                                            "topk": TOPK, "head_write": hw}), "head_only")
        combo_eq_ffn = identical(combo_summary, ffn_only_again)
        combo_eq_head = identical(combo_summary, head_only)
        print("\n=== 2) ffn_write(L1) + head_write(L2) cross-hook composition ===")
        print(f"  response flags: ffn_write_applied={combo.get('ffn_write_applied')} "
              f"head_write_applied={combo.get('head_write_applied')}")
        print(f"  combo top1={combo_summary.get('top1_piece')!r} logprob={combo_summary.get('top1_logprob')}")
        print(f"  combo == ffn-write-only : {combo_eq_ffn}")
        print(f"  combo == head-write-only: {combo_eq_head}")
        print(f"  COMPOSITION (combo differs from both single-surface arms): "
              f"{(not combo_eq_ffn) and (not combo_eq_head)}")
        out["cross_hook_composition"] = {
            "combo_flags": {k: combo.get(k) for k in ("ffn_write_applied", "head_write_applied")},
            "combo": combo_summary, "ffn_only": ffn_only_again, "head_only": head_only,
            "combo_eq_ffn_only": combo_eq_ffn, "combo_eq_head_only": combo_eq_head,
        }

        # ================= 3) honesty fix: the ORIGINAL two bug repros now 400, not a lie =================
        print("\n=== 3) honesty fix: malformed specs (measured live) ===")

        bad_hw = {"layer": L1, "head": 0, "positions": [p], "values": hslice[:-1]}  # d_head-1 floats
        bad_resp = post(base, {"prompt": PROMPT, "continuation": " Paris", "head_write": bad_hw})
        print(f"  head_write malformed values length -> HTTP {bad_resp['_status']}: "
              f"{bad_resp.get('_body', bad_resp).get('error', bad_resp.get('_body'))}")
        out["bug_repro_malformed_head_write_values"] = {
            "status": bad_resp["_status"], "body": bad_resp.get("_body", bad_resp)}

        oob_hw = {"layer": n_layer + 50, "head": 0, "positions": [p], "values": hslice}
        oob_resp = post(base, {"prompt": PROMPT, "continuation": " Paris", "head_write": oob_hw})
        print(f"  head_write out-of-range layer (n_layer+50) -> HTTP {oob_resp['_status']}: "
              f"{oob_resp.get('_body', oob_resp).get('error', oob_resp.get('_body'))}")
        out["bug_repro_oob_layer_head_write"] = {
            "status": oob_resp["_status"], "body": oob_resp.get("_body", oob_resp)}

        bad_fw = {"layer": n_layer + 50, "positions": [p], "values": row_L1}
        bad_fw_resp = post(base, {"prompt": PROMPT, "continuation": " Paris", "ffn_write": bad_fw})
        print(f"  ffn_write out-of-range layer (n_layer+50) -> HTTP {bad_fw_resp['_status']}: "
              f"{bad_fw_resp.get('_body', bad_fw_resp).get('error', bad_fw_resp.get('_body'))}")
        out["bug_repro_oob_layer_ffn_write"] = {
            "status": bad_fw_resp["_status"], "body": bad_fw_resp.get("_body", bad_fw_resp)}

        short_fw = {"layer": L1, "positions": [p], "values": row_L1[:-1]}   # n_embd-1 floats
        short_fw_resp = post(base, {"prompt": PROMPT, "continuation": " Paris", "ffn_write": short_fw})
        print(f"  ffn_write mismatched values length (n_embd-1) -> HTTP {short_fw_resp['_status']}: "
              f"{short_fw_resp.get('_body', short_fw_resp).get('error', short_fw_resp.get('_body'))}")
        out["bug_repro_short_values_ffn_write"] = {
            "status": short_fw_resp["_status"], "body": short_fw_resp.get("_body", short_fw_resp)}

        all_400 = all(r["_status"] == 400 for r in
                      (bad_resp, oob_resp, bad_fw_resp, short_fw_resp))
        print(f"\n  ALL FOUR malformed specs correctly rejected with 400 (never a 200 'applied:true' "
              f"lie): {all_400}")
        out["all_malformed_specs_rejected_400"] = all_400

        os.makedirs(os.path.join(REPO, "runs", "experiments"), exist_ok=True)
        outpath = os.path.join(REPO, "runs", "experiments", "ffn_hook_probe.json")
        json.dump(out, open(outpath, "w"), indent=2)
        print(f"\nwrote {outpath}")
        return 0
    finally:
        if proc is not None:
            _terminate_process(proc)
        subprocess.run(["taskkill", "/F", "/IM", "clozn-server.exe"], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
