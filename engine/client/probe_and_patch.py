"""probe_and_patch.py — discover a direction in Python on the LIVE engine's activations,
then causally verify it with /state. This is the thing the SDK uniquely enables, end to
end on the C++ runtime with no HF model in the loop:

    1. READ      harvest token activations for two contrasting corpora from the engine
    2. DISCOVER  fit a diff-in-means direction in numpy on the engine's own residuals
    3. WRITE     add that direction to a neutral prompt's last position via /state
    4. OBSERVE   measure how the next-token prediction moves, against a random-direction
                 control of the SAME norm (the honesty baseline — a real direction must
                 beat a random one, or the "effect" is just perturbation magnitude)

The engine calibrates its own concept probes internally, but it can only do diff-in-means
on a fixed corpus. Here the discovery is arbitrary Python: swap `diff_in_means` for an SAE
encode, a learned probe, a PCA component — anything numpy — and /state still verifies it
causally on the live weights. That is why the SDK exists.

Run against a server started with a small GGUF (open-dcoder-0.5b is enough):
    python probe_and_patch.py --port 8091 --layer 14 --coef 3.0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from clozn_engine import EngineClient

# Token pieces can be arbitrary UTF-8 (a perturbed prediction may land on CJK/jamo); the
# Windows console defaults to cp1252 and would crash on them. Print through UTF-8 with a
# replacement fallback so the harness never dies on an unprintable byte.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# Two contrasting corpora. The diff of their pooled activation means is a "number vs prose"
# direction in the model's own residual space — the kind of concept the engine itself can
# only read internally, here discovered client-side.
NUMBER_TEXTS = [
    "In 2024 the company sold 15 boats, 320 bikes, and 7 cars.",
    "Pi is about 3.14159 and e is roughly 2.71828.",
    "The total came to 4827 dollars after a 12 percent discount.",
    "She counted 100, 200, 300, 400, and finally 500 steps.",
    "Room 237 is on floor 18, just past suite 1600.",
]
PROSE_TEXTS = [
    "The quick brown fox jumps over the lazy dog by the river.",
    "She walked slowly to the park and sat beneath the old oak.",
    "He wondered whether the letter would ever reach its reader.",
    "Morning light spilled across the quiet, half-forgotten kitchen.",
    "They spoke of distant places they would probably never see.",
]


def pooled_mean(eng: EngineClient, texts: list[str], layer: int) -> tuple[np.ndarray, int]:
    """Harvest each text at `layer` and return the mean over ALL token rows (the pooled
    class mean), plus the layer the server actually read at (threaded back for the write)."""
    rows = []
    used_layer = layer
    for t in texts:
        h = eng.harvest(t, layer)
        used_layer = h.layer
        rows.append(h.activations)
    return np.concatenate(rows, axis=0).mean(axis=0), used_layer


def diff_in_means(eng: EngineClient, layer: int) -> tuple[np.ndarray, int]:
    """The discovery step (swap me for an SAE / PCA / learned probe). Unit-normalized
    (mean of NUMBER rows) - (mean of PROSE rows) in the engine's residual space at `layer`."""
    mu_num, used = pooled_mean(eng, NUMBER_TEXTS, layer)
    mu_prose, _ = pooled_mean(eng, PROSE_TEXTS, layer)
    d = mu_num - mu_prose
    d = d / (np.linalg.norm(d) + 1e-8)
    return d.astype("<f4"), used


def digit_mass(top: list) -> float:
    """Fraction of the reported top-k probability sitting on tokens that contain a digit —
    a cheap scalar for 'did the prediction move toward numbers'."""
    return float(sum(t["prob"] for t in top if any(c.isdigit() for c in t["token"])))


def patch_last(eng: EngineClient, prompt: str, layer: int, direction: np.ndarray, coef: float):
    """Harvest `prompt`, add coef * ||last_row|| * direction to the last position, write it
    back at `layer`, and return the Observation (baseline vs edited next-token distribution)."""
    h = eng.harvest(prompt, layer)
    last = h.n_tokens - 1
    row = h.activations[last].astype("<f4")
    edited = row + np.float32(coef * float(np.linalg.norm(row))) * direction
    return eng.write_state(prompt, h.layer, [last], edited)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="discover a direction in Python, verify it via /state")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--layer", type=int, default=14, help="mid-depth tap to read + write at")
    ap.add_argument("--coef", type=float, default=0.25,
                    help="edit strength (x the row norm); ~0.25 nudges, >1 over-injects to garble")
    ap.add_argument("--prompt", default="My favorite number is", help="neutral prompt to patch")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random-direction control")
    args = ap.parse_args(argv)

    eng = EngineClient(host=args.host, port=args.port)
    info = eng.health()
    print(f"server: {info.get('model')}  mode={info.get('mode')}\n")

    print(f"[1-2] discovering the number-vs-prose direction at layer {args.layer} "
          f"({len(NUMBER_TEXTS)}+{len(PROSE_TEXTS)} texts)...")
    direction, used_layer = diff_in_means(eng, args.layer)
    print(f"      discovered a unit direction in R^{direction.size} at layer {used_layer}\n")

    # A random control direction of the same shape and unit norm: the perturbation has the
    # same magnitude, so any extra movement toward numbers is the DIRECTION, not the nudge.
    rng = np.random.default_rng(args.seed)
    rand = rng.standard_normal(direction.size).astype("<f4")
    rand /= np.linalg.norm(rand) + 1e-8

    print(f"[3-4] patching {args.prompt!r} last token (coef {args.coef} x row norm), observing:\n")
    real = patch_last(eng, args.prompt, used_layer, direction, args.coef)
    ctrl = patch_last(eng, args.prompt, used_layer, rand, args.coef)

    def show(name: str, obs) -> None:
        if not obs.applied:
            print(f"  {name}: rejected ({obs.error})")
            return
        top = ", ".join(f"{t['token']!r} {t['prob']:.3f}" for t in obs.edited_top)
        print(f"  {name:18s} moved_l2={obs.moved_l2:8.2f}  digit_mass={digit_mass(obs.edited_top):.3f}"
              f"   top: {top}")

    base = ", ".join(f"{t['token']!r} {t['prob']:.3f}" for t in real.baseline_top)
    print(f"  {'baseline':18s} {'':21s}digit_mass={digit_mass(real.baseline_top):.3f}   top: {base}")
    show("number-direction", real)
    show("random-direction", ctrl)

    real_gain = digit_mass(real.edited_top) - digit_mass(real.baseline_top)
    ctrl_gain = digit_mass(ctrl.edited_top) - digit_mass(real.baseline_top)
    print(f"\n  digit-mass gain: number-direction {real_gain:+.3f} vs random {ctrl_gain:+.3f}  "
          f"-> {'direction beats the control' if real_gain > ctrl_gain else 'inconclusive at this coef/layer'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
