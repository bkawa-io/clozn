"""Generate a synthetic LoRA adapter GGUF matched to a base model, for testing adapter support.

WHY THIS EXISTS
---------------
Proving `clozn serve --adapter` works needs a real LoRA GGUF, and the normal way to get one -- train a
fine-tune, export it through PEFT, run convert_lora_to_gguf.py -- needs a training run, a GPU, an HF
config directory, and a network. None of that belongs in a test lane. This writes a structurally-valid
adapter directly against a base GGUF's own tensor shapes, so the test material is reproducible from
nothing but a base model that is already on disk.

The weights are deterministic pseudo-random, not trained. That is the point: the adapter is a probe for
"did the delta reach the forward pass", not a model that knows anything. Two properties make it useful:

  --scale 1.0   changes the output measurably (proves the delta is applied)
  --scale 0.0   attaches but contributes nothing (the identity control -- proves an observed change came
                from the adapter's weights, not merely from loading an adapter)

THE FORMAT, AS llama.cpp ACTUALLY CHECKS IT
-------------------------------------------
From engine/core/third_party/llama.cpp/src/llama-adapter.cpp:

  general.type          must be "adapter"          (else: "expect general.type to be 'adapter'")
  general.architecture  must equal the model's     (else: "model arch and LoRA arch mismatch")
  adapter.type          must be "lora"
  adapter.lora.alpha    f32; the effective scale is alpha/rank * the runtime scale

  <base_tensor>.lora_a  ne = [in,  rank]   where in  == base tensor ne[0]
  <base_tensor>.lora_b  ne = [rank, out]   where out == base tensor ne[1]
  and a.ne[1] == b.ne[0], or the loader reports "lora_a tensor is not transposed"

gguf-py writes a numpy array with its LAST axis as ggml's ne[0], so the numpy shapes are the reverse of
the ne shapes above -- lora_a is written as (rank, in) and lora_b as (out, rank). Getting this backwards
produces a file that loads and then fails the shape assertion, which is a confusing way to spend an hour.

USAGE
-----
    python scripts/dev/make_test_lora.py BASE.gguf --out adapter.gguf
    python scripts/dev/make_test_lora.py BASE.gguf --out mismatch.gguf --arch llama   # refusal fixture
"""
from __future__ import annotations

import argparse
import os
import sys

_LLAMA_CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "engine", "core", "third_party", "llama.cpp", "gguf-py")
if os.path.isdir(_LLAMA_CPP):
    sys.path.insert(0, os.path.abspath(_LLAMA_CPP))

import numpy as np                      # noqa: E402
import gguf                             # noqa: E402

# Tensors to adapt. attn_q is the conventional LoRA target and sits on the path to every logit, so a
# nonzero delta there is guaranteed to be observable rather than merely present.
DEFAULT_TARGETS = ("attn_q", "attn_v")


def _read_base(path: str) -> tuple[str, dict[str, tuple[int, int]]]:
    """(architecture, {tensor_name: (ne0, ne1)}) for the 2-D tensors of a base GGUF."""
    reader = gguf.GGUFReader(path)
    arch = None
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = bytes(field.parts[field.data[0]]).decode("utf-8")
            break
    if not arch:
        raise SystemExit(f"{path}: no general.architecture -- is this a GGUF model?")
    shapes = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors if len(t.shape) == 2}
    return arch, shapes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base", help="path to the base model GGUF whose shapes the adapter must match")
    ap.add_argument("--out", required=True, help="path to write the adapter GGUF to")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--magnitude", type=float, default=0.02,
                    help="stddev of the pseudo-random factors; the delta is B@A, so the effect on the "
                         "output grows with the square of this")
    ap.add_argument("--seed", type=int, default=0, help="fixed seed -- the fixture must be reproducible")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help=f"comma-separated tensor suffixes to adapt (default: {','.join(DEFAULT_TARGETS)})")
    ap.add_argument("--arch", default=None,
                    help="override the declared architecture -- use a WRONG value on purpose to build "
                         "a fixture that must be refused ('model arch and LoRA arch mismatch')")
    ap.add_argument("--layers", type=int, default=0, help="adapt only the first N blocks (0 = all)")
    args = ap.parse_args()

    arch, shapes = _read_base(args.base)
    declared = args.arch or arch
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())

    selected = []
    for name, (ne0, ne1) in sorted(shapes.items()):
        if not name.startswith("blk.") or not name.endswith(".weight"):
            continue
        if not any(f".{t}.weight" == name[name.index(".", 4):] for t in targets):
            continue
        if args.layers:
            try:
                if int(name.split(".")[1]) >= args.layers:
                    continue
            except (IndexError, ValueError):
                continue
        selected.append((name, ne0, ne1))

    if not selected:
        raise SystemExit(f"no tensors matched targets {targets!r} in {args.base}")

    rng = np.random.default_rng(args.seed)
    writer = gguf.GGUFWriter(args.out, declared)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(args.alpha))

    for name, ne0, ne1 in selected:
        # ne shapes are lora_a [ne0, rank] and lora_b [rank, ne1]; numpy is the reverse of ne.
        a = rng.normal(0.0, args.magnitude, size=(args.rank, ne0)).astype(np.float32)
        b = rng.normal(0.0, args.magnitude, size=(ne1, args.rank)).astype(np.float32)
        writer.add_tensor(f"{name}.lora_a", a)
        writer.add_tensor(f"{name}.lora_b", b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    note = "" if declared == arch else f"  (DELIBERATE arch mismatch: base is '{arch}')"
    print(f"wrote {args.out}")
    print(f"  architecture : {declared}{note}")
    print(f"  rank/alpha   : {args.rank} / {args.alpha}")
    print(f"  tensors      : {len(selected)} adapted ({2 * len(selected)} lora_a/lora_b entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
