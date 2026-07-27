"""export_sae_weights.py -- andyrdt_l15_sae.pt -> raw blobs the C++ SaeEncoder loads (no torch on the C++ side).

The engine-side SAE readout (clozn/sae.hpp) wants flat, mmap-simple weight files, not a pickled
torch checkpoint. This one-shot exporter writes them next to the models dir convention:

    python export_sae_weights.py [--pt ~/hf_models/andyrdt_l15_sae.pt] [--out ~/.clozn/sae/andyrdt_l15]

Layout choice that matters: W_enc is stored TRANSPOSED, [d_sae, d_in] fp16 row-major (feature-major),
so each feature's encoder weights are contiguous -- the engine GEMV walks one feature per warp over a
contiguous 7KB row. (The .pt stores [d_in, d_sae]; a JumpReLU SAE's W_enc is NOT W_dec^T, so the
transpose has to happen here.)

W_dec IS exported (added 2026-07-20), already [d_sae, d_in] in the .pt so no transpose. The engine
itself still only encodes -- but CAUSAL work on the host needs the decoder, and its absence blocked
the first feature-level tracing attempt outright:
  * canonical feature ablation is  h' = h - a_f * d_dec[f]  (remove the feature's CONTRIBUTION).
    Ablating along the ENCODER direction instead removes what the feature *reads*, which is a
    different vector and measurably different results (see notes/CIRCUIT_TRACER_DESIGN.md 5e).
  * SAE reconstruction  h_hat = b_dec + sum_f a_f * d_dec[f]  is what the
    substitute-and-rescore fidelity test needs.
Decoder rows are unit-norm in this parameterisation, which the loader can use as a receipt.

meta.txt carries shapes + fp32 L2 norms of every blob -- the loader recomputes and refuses to serve
silently-corrupt weights (the stage-1 receipt for wiring sae_topk into the engine).
"""
import argparse
import json
import os

import torch

BLOBS = [
    # meta key      pt key       transpose  out dtype       filename
    ("w_enc_t",     "W_enc",     True,      torch.float16,  "w_enc_t.f16.bin"),
    ("w_dec",       "W_dec",     False,     torch.float16,  "w_dec.f16.bin"),   # already [d_sae, d_in]
    ("b_enc",       "b_enc",     False,     torch.float16,  "b_enc.f16.bin"),
    ("b_dec",       "b_dec",     False,     torch.float16,  "b_dec.f16.bin"),
    ("threshold",   "threshold", False,     torch.float32,  "threshold.f32.bin"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.expanduser("~/hf_models/andyrdt_l15_sae.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/.clozn/sae/andyrdt_l15"))
    args = ap.parse_args()

    d = torch.load(args.pt, map_location="cpu")
    d_in, d_sae, layer = int(d["d_in"]), int(d["d_sae"]), int(d["layer"])
    assert d["W_enc"].shape == (d_in, d_sae), f"W_enc shape {tuple(d['W_enc'].shape)}"
    os.makedirs(args.out, exist_ok=True)

    meta = {"format": "clozn-sae-v1", "kind": "jumprelu", "d_in": d_in, "d_sae": d_sae,
            "layer": layer, "source": args.pt}
    lines = [f"format clozn-sae-v1", f"kind jumprelu", f"d_in {d_in}", f"d_sae {d_sae}", f"layer {layer}"]
    for key, pt_key, transpose, dtype, fname in BLOBS:
        t = d[pt_key]
        if transpose:
            t = t.t().contiguous()
        t = t.to(dtype)
        path = os.path.join(args.out, fname)
        t.numpy().tofile(path)
        # Norm receipt in float64: an fp32 accumulation over W_enc's 4.7e8 tiny squares loses ~8%
        # of the mass (measured: 519.8 fp32 vs 563.2 fp64) — the loader recomputes in double.
        norm = float(t.double().norm())
        meta[key] = {"file": fname, "shape": list(t.shape), "dtype": str(dtype).split(".")[-1],
                     "l2_norm": norm}
        lines.append(f"{key} {fname}")
        lines.append(f"norm_{key} {norm:.6f}")
        print(f"  {key:10s} {str(list(t.shape)):>18s} {os.path.getsize(path)/1e6:8.1f} MB  |x|={norm:.4f}")

    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"exported -> {args.out}  (encoder-only: {sum(os.path.getsize(os.path.join(args.out, b[4])) for b in BLOBS)/1e9:.2f} GB on disk)")


if __name__ == "__main__":
    main()
