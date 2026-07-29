"""additive_writes_probe.py -- SPIKE (spike/additive-writes branch, throwaway): measures whether
residual `write` (l_out-<il>) COMPOSES across two layers in one /score forward, and whether
`head_write` (kqv_out-<il>) does. See the task brief for the full design; short version:

  Assumption A (residual l_out write): mechanism is ggml_backend_tensor_set OVERWRITE during the
  eval callback (model_ggml.cpp:355-368, matched by exact tensor name "l_out-<il>"). If two writes
  target the SAME position at L1 < L2, the L1 edit's effect on that position is fully re-computed
  through L1+1..L2 and then COMPLETELY DISCARDED when L2's tensor is overwritten -- so "both" should
  equal "L2 only" bit-for-bit, PROVIDED the shared write position is causally terminal (no later
  token attends to it) so there is no other leakage path. This script writes at the LAST prompt
  position (n_p-1, which is also the sole /score logits_for row for a 1-token continuation), so
  that condition holds by construction.

  Assumption B (head_write on kqv_out): mechanism is ggml_backend_tensor_set on the PRE-W_o merged
  attention-head tensor (model_ggml.cpp:288-302). This tensor is NOT the residual stream -- it is
  later multiplied by W_o and ADDED into the residual (l_out = residual_in + W_o(kqv_out) + mlp_out).
  So an L1 head_write changes what gets ADDED at L1, which changes residual_in arriving at L2; an L2
  head_write only overwrites L2's OWN attention contribution, leaving that changed residual_in intact.
  Prediction: "both" != "L2 only" (writes compose), unlike Assumption A.

Run: python scripts/spike/additive_writes_probe.py
Writes runs/experiments/additive_writes_probe.json (throwaway artifact, not a product output).
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

PORT = 8097
MODEL_GLOB = os.path.expanduser("~/.clozn/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
TARGET_PROMPT = "The capital of France is"
DONOR_PROMPT = "The capital of Japan is"
TOPK = 10


def post(base, body):
    req = urllib.request.Request(base + "/score", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": json.loads(e.read())}


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def score_baseline(base, prompt):
    return post(base, {"prompt": prompt, "continuation": " x", "topk": TOPK})


def top_summary(resp, label):
    """Pull the FIRST (only) continuation-position token row's topk out for comparison."""
    if "_http_error" in resp:
        return {"label": label, "error": resp}
    tok = resp["tokens"][0]
    return {"label": label, "n_prompt": resp.get("n_prompt"),
            "top1_id": tok["topk"][0]["id"], "top1_piece": tok["topk"][0]["piece"],
            "top1_logprob": tok["topk"][0]["logprob"],
            "actual_id": tok["id"], "actual_logprob": tok["logprob"],
            "topk": [(t["id"], t["piece"], t["logprob"]) for t in tok["topk"]]}


def identical(a, b):
    """Bit-for-bit comparison of the topk logprob vectors (id AND logprob, exactly)."""
    if a.get("error") or b.get("error"):
        return False
    return a["topk"] == b["topk"] and a["actual_logprob"] == b["actual_logprob"]


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
    out = {"model": gguf, "target_prompt": TARGET_PROMPT, "donor_prompt": DONOR_PROMPT}
    try:
        proc, health, gpu = spawn_engine(gguf, PORT, _flags_for(gguf), prefer_gpu=True)
        base = f"http://127.0.0.1:{PORT}"
        h = get(base, "/health")
        n_layer, n_embd = h["n_layer"], h["n_embd"]
        out["gpu"] = gpu
        out["n_layer"] = n_layer
        out["n_embd"] = n_embd
        out["capabilities"] = h.get("capabilities")
        print(f"[boot] gpu={gpu} n_layer={n_layer} n_embd={n_embd} "
              f"capabilities={h.get('capabilities')}", flush=True)

        # well-separated mid layers, safely inside [1, n_layer-2] (avoid the last-layer capture gap)
        L1 = max(2, n_layer // 6)
        L2 = n_layer - 4
        assert 1 <= L1 < L2 <= n_layer - 2, (L1, L2, n_layer)
        out["L1"], out["L2"] = L1, L2
        print(f"[layers] L1={L1} L2={L2}", flush=True)

        # ---- baseline forwards (also gives us n_prompt for each) ----
        b_target = score_baseline(base, TARGET_PROMPT)
        b_donor = score_baseline(base, DONOR_PROMPT)
        n_p_target = b_target["n_prompt"]
        n_p_donor = b_donor["n_prompt"]
        p_target = n_p_target - 1     # the sole scored position (logits_for = [n_p-1])
        p_donor = n_p_donor - 1
        out["p_target"], out["p_donor"] = p_target, p_donor
        base_summary = top_summary(b_target, "target_baseline")
        print(f"[baseline target] top1={base_summary['top1_piece']!r} "
              f"logprob={base_summary['top1_logprob']:.6f}", flush=True)
        print(f"[baseline donor]  top1={top_summary(b_donor, 'donor_baseline')['top1_piece']!r}",
              flush=True)

        # ================= ASSUMPTION A: residual l_out write composition =================
        cap = post(base, {"prompt": DONOR_PROMPT, "continuation": " x",
                          "capture": {"layers": [L1, L2], "positions": [p_donor]}})
        if "_http_error" in cap:
            print("CAPTURE FAILED:", cap); return 1
        resid_L1 = cap["captured"][str(L1)][str(p_donor)]
        resid_L2 = cap["captured"][str(L2)][str(p_donor)]
        assert len(resid_L1) == n_embd and len(resid_L2) == n_embd

        w1 = {"layer": L1, "positions": [p_target], "values": resid_L1}
        w2 = {"layer": L2, "positions": [p_target], "values": resid_L2}

        arm_baseline = top_summary(score_baseline(base, TARGET_PROMPT), "A_baseline")
        arm_L1 = top_summary(post(base, {"prompt": TARGET_PROMPT, "continuation": " x",
                                         "topk": TOPK, "write": w1}), "A_L1_only")
        arm_L2 = top_summary(post(base, {"prompt": TARGET_PROMPT, "continuation": " x",
                                         "topk": TOPK, "write": w2}), "A_L2_only")
        arm_both = top_summary(post(base, {"prompt": TARGET_PROMPT, "continuation": " x",
                                           "topk": TOPK, "write": [w1, w2]}), "A_both")

        both_eq_L2 = identical(arm_both, arm_L2)
        both_eq_L1 = identical(arm_both, arm_L1)
        both_eq_baseline = identical(arm_both, arm_baseline)
        moved_from_baseline = not identical(arm_L2, arm_baseline)

        print("\n=== ASSUMPTION A (residual l_out write) ===")
        for a in (arm_baseline, arm_L1, arm_L2, arm_both):
            print(f"  {a['label']:14s} top1={a['top1_piece']!r:8s} logprob={a['top1_logprob']:.8f} "
                  f"actual_logprob={a['actual_logprob']:.8f}")
        print(f"  both == L2_only : {both_eq_L2}")
        print(f"  both == L1_only : {both_eq_L1}")
        print(f"  both == baseline: {both_eq_baseline}")
        print(f"  L2_only moved from baseline (perturbation had an effect): {moved_from_baseline}")

        out["assumption_A"] = {
            "L1": L1, "L2": L2, "write_position": p_target,
            "arms": {a["label"]: a for a in (arm_baseline, arm_L1, arm_L2, arm_both)},
            "both_eq_L2_only": both_eq_L2, "both_eq_L1_only": both_eq_L1,
            "both_eq_baseline": both_eq_baseline, "L2_only_moved_output": moved_from_baseline,
        }

        # ================= ASSUMPTION B: head_write (kqv_out) composition =================
        hcap = post(base, {"prompt": DONOR_PROMPT, "continuation": " x",
                           "head_capture": {"layers": [L1, L2], "positions": [p_donor], "rows": True}})
        if "_http_error" in hcap:
            print("HEAD CAPTURE FAILED:", hcap); return 1
        head_dims = hcap["head_dims"]
        d_head, n_head = head_dims["d_head"], head_dims["n_head"]
        print(f"\n[head_dims] ne0={head_dims['ne0']} n_head={n_head} d_head={d_head}", flush=True)
        H = 0  # probe head index 0
        row_L1 = hcap["head_rows"][str(L1)][str(p_donor)]
        row_L2 = hcap["head_rows"][str(L2)][str(p_donor)]
        slice_L1 = row_L1[H * d_head:(H + 1) * d_head]
        slice_L2 = row_L2[H * d_head:(H + 1) * d_head]
        assert len(slice_L1) == d_head and len(slice_L2) == d_head

        hw1 = {"layer": L1, "head": H, "positions": [p_target], "values": slice_L1}
        hw2 = {"layer": L2, "head": H, "positions": [p_target], "values": slice_L2}

        harm_baseline = top_summary(score_baseline(base, TARGET_PROMPT), "B_baseline")
        harm_L1 = top_summary(post(base, {"prompt": TARGET_PROMPT, "continuation": " x",
                                          "topk": TOPK, "head_write": hw1}), "B_L1_only")
        harm_L2 = top_summary(post(base, {"prompt": TARGET_PROMPT, "continuation": " x",
                                          "topk": TOPK, "head_write": hw2}), "B_L2_only")
        harm_both = top_summary(post(base, {"prompt": TARGET_PROMPT, "continuation": " x",
                                            "topk": TOPK, "head_write": [hw1, hw2]}), "B_both")

        hboth_eq_L2 = identical(harm_both, harm_L2)
        hboth_eq_L1 = identical(harm_both, harm_L1)
        hboth_eq_baseline = identical(harm_both, harm_baseline)
        hL2_moved = not identical(harm_L2, harm_baseline)
        hL1_moved = not identical(harm_L1, harm_baseline)

        print("\n=== ASSUMPTION B (head_write on kqv_out) ===")
        for a in (harm_baseline, harm_L1, harm_L2, harm_both):
            print(f"  {a['label']:14s} top1={a['top1_piece']!r:8s} logprob={a['top1_logprob']:.8f} "
                  f"actual_logprob={a['actual_logprob']:.8f}")
        print(f"  both == L2_only : {hboth_eq_L2}")
        print(f"  both == L1_only : {hboth_eq_L1}")
        print(f"  both == baseline: {hboth_eq_baseline}")
        print(f"  L1_only moved from baseline: {hL1_moved}")
        print(f"  L2_only moved from baseline: {hL2_moved}")

        out["assumption_B"] = {
            "L1": L1, "L2": L2, "head": H, "d_head": d_head, "n_head": n_head,
            "write_position": p_target,
            "arms": {a["label"]: a for a in (harm_baseline, harm_L1, harm_L2, harm_both)},
            "both_eq_L2_only": hboth_eq_L2, "both_eq_L1_only": hboth_eq_L1,
            "both_eq_baseline": hboth_eq_baseline,
            "L1_only_moved_output": hL1_moved, "L2_only_moved_output": hL2_moved,
        }

        # ---- B extras: array-of-specs, combine with residual write, validation probes ----
        combo = post(base, {"prompt": TARGET_PROMPT, "continuation": " x", "topk": TOPK,
                            "write": w2, "head_write": hw1})
        combo_summary = top_summary(combo, "B_combo_residual_and_head")
        combo_eq_write_only = identical(combo_summary, arm_L2)
        combo_eq_head_only = identical(combo_summary, harm_L1)
        combo_response_flags = {k: combo.get(k) for k in
                                ("write_applied", "n_writes", "head_write_applied", "n_head_writes")}
        print("\n=== B extra: residual write + head_write COMBINED in one /score ===")
        print(f"  response flags: {combo_response_flags}")
        print(f"  combo top1={combo_summary.get('top1_piece')!r} "
              f"logprob={combo_summary.get('top1_logprob')}")
        print(f"  combo == (residual-L2-write only): {combo_eq_write_only}")
        print(f"  combo == (head-L1-write only):      {combo_eq_head_only}")
        print("  (if combo differs from BOTH single-surface arms, both effects landed together)")

        # malformed head_write: wrong values length (d_head-1 floats instead of d_head) -> per
        # model_ggml.cpp:291-292 this spec is silently skipped inside eval_cb ("continue"), yet the
        # route always reports head_write_applied:true whenever the spec parsed structurally OK.
        bad_hw = {"layer": L1, "head": H, "positions": [p_target], "values": slice_L1[:-1]}
        bad_resp = post(base, {"prompt": TARGET_PROMPT, "continuation": " x", "topk": TOPK,
                               "head_write": bad_hw})
        bad_summary = top_summary(bad_resp, "B_malformed_values_length")
        bad_silently_noop = identical(bad_summary, harm_baseline)
        print("\n=== B extra: malformed head_write (values len = d_head-1) ===")
        print(f"  HTTP status/flags: applied={bad_resp.get('head_write_applied')} "
              f"n_head_writes={bad_resp.get('n_head_writes')}")
        print(f"  actually silently no-op'd (== baseline): {bad_silently_noop}")

        # out-of-range layer: route only checks layer<0; GgmlAdapter::set_head_writes silently
        # drops layer>=n_layer entries (model_ggml.cpp:143-148) -- same "applied:true but nothing
        # happened" shape.
        oob_hw = {"layer": n_layer + 50, "head": H, "positions": [p_target], "values": slice_L1}
        oob_resp = post(base, {"prompt": TARGET_PROMPT, "continuation": " x", "topk": TOPK,
                               "head_write": oob_hw})
        oob_summary = top_summary(oob_resp, "B_out_of_range_layer")
        oob_silently_noop = identical(oob_summary, harm_baseline)
        print("\n=== B extra: out-of-range layer head_write (layer=n_layer+50) ===")
        print(f"  HTTP status/flags: applied={oob_resp.get('head_write_applied')} "
              f"n_head_writes={oob_resp.get('n_head_writes')}")
        print(f"  actually silently no-op'd (== baseline): {oob_silently_noop}")

        # explicit 400 validation text: missing head field
        missing_head = post(base, {"prompt": TARGET_PROMPT, "continuation": " x",
                                   "head_write": {"layer": L1, "positions": [p_target],
                                                  "values": slice_L1}})
        print("\n=== B extra: head_write missing 'head' field -> validation text ===")
        print(f"  {missing_head}")

        out["B_extras"] = {
            "combo_response_flags": combo_response_flags,
            "combo_eq_residual_write_only": combo_eq_write_only,
            "combo_eq_head_write_only": combo_eq_head_only,
            "malformed_values_length_silently_noop": bad_silently_noop,
            "malformed_values_length_response_flags":
                {k: bad_resp.get(k) for k in ("head_write_applied", "n_head_writes")},
            "out_of_range_layer_silently_noop": oob_silently_noop,
            "out_of_range_layer_response_flags":
                {k: oob_resp.get(k) for k in ("head_write_applied", "n_head_writes")},
            "missing_head_field_response": missing_head,
        }

        os.makedirs(os.path.join(REPO, "runs", "experiments"), exist_ok=True)
        outpath = os.path.join(REPO, "runs", "experiments", "additive_writes_probe.json")
        json.dump(out, open(outpath, "w"), indent=2)
        print(f"\nwrote {outpath}")
        return 0
    finally:
        if proc is not None:
            _terminate_process(proc)
        subprocess.run(["taskkill", "/F", "/IM", "clozn-server.exe"], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
