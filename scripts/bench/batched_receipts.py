# bench_batched_receipts.py — Phase 2 of the local-efficiency investigation: the BATCHED-RECEIPTS
# PROOF. The differential idea: a causal receipt is a *pair* of generations (with vs without the
# memory/prefix under test). Run naively that doubles cost. But batch-1 decode is memory-bandwidth
# -bound — the GPU streams all weights per step regardless of batch — so stacking the ablation twin
# into the SAME generate call (batch of 2, left-padded, attention-masked) should cost far less than
# 2x. If wall(batch2) <= 1.3 * wall(batch1), receipts-are-nearly-free is CONFIRMED and wiring
# batched ablation into replay.py is justified (not done here; this is the measurement).
#
#   C:/Users/brigi/src/clozn/.venv/Scripts/python.exe research/bench_batched_receipts.py \
#       [--models 1.5b,7b] [--new 128] [--reps 5] [--json-out runs/batched_receipts.json]
#
# Models mirror the studio's real configs: Qwen2.5-1.5B-Instruct bf16, Qwen2.5-7B-Instruct nf4.
# Greedy, min_new_tokens == max_new_tokens (no early-EOS length variance), CUDA-synchronized walls,
# median of --reps after a discarded warmup. Batch-4 included as the "N receipts" extrapolation row.
import argparse
import gc
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QUESTION = ("I need to plan a birthday dinner for six people on a weeknight. "
            "Suggest a menu I can cook in under two hours, with one vegetarian main.")
MEMORY = ("The user's name is Mika. They are vegetarian, allergic to peanuts, prefer concise "
          "answers with numbered steps, and cook on a two-burner stove in a small apartment.")


def build_pair(tok):
    """The with/without-receipt pair: same user turn, one carries the memory block. Chat-templated,
    different lengths (left-padding absorbs it)."""
    with_mem = tok.apply_chat_template(
        [{"role": "system", "content": MEMORY}, {"role": "user", "content": QUESTION}],
        tokenize=False, add_generation_prompt=True)
    without = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False, add_generation_prompt=True)
    return with_mem, without


@torch.no_grad()
def timed_generate(model, enc, new, pad_id):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=new, min_new_tokens=new,
                         do_sample=False, use_cache=True, pad_token_id=pad_id)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    assert out.shape[1] == enc["input_ids"].shape[1] + new, "unexpected output length"
    return wall


def bench_model(name, model_id, quant, new, reps, batch4):
    print(f"\n=== {name} ({model_id}, {'nf4' if quant else 'bf16'}) ===")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    if quant:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    with_mem, without = build_pair(tok)
    dev = model.device
    encs = {
        1: tok([with_mem], return_tensors="pt", padding=True).to(dev),
        2: tok([with_mem, without], return_tensors="pt", padding=True).to(dev),
    }
    if batch4:  # 1 base + 3 ablation variants — the "N receipts in one call" extrapolation
        encs[4] = tok([with_mem, without, with_mem, without],
                      return_tensors="pt", padding=True).to(dev)

    results = {}
    for b, enc in encs.items():
        timed_generate(model, enc, new, tok.pad_token_id)  # warmup (discarded)
        walls = [timed_generate(model, enc, new, tok.pad_token_id) for _ in range(reps)]
        med = statistics.median(walls)
        results[b] = {"wall_s": med, "all_walls": walls,
                      "prompt_len": int(enc["input_ids"].shape[1]),
                      "seq_tok_per_s": new / med, "total_tok_per_s": b * new / med}
        print(f"  batch={b}: wall {med:6.2f} s  ({new} new tok/seq; "
              f"{b * new / med:6.1f} total tok/s; prompt {enc['input_ids'].shape[1]} tok padded)")

    r21 = results[2]["wall_s"] / results[1]["wall_s"]
    print(f"  batch-2 / batch-1 wall ratio: {r21:.3f}  "
          f"({'<= 1.3: receipts-nearly-free CONFIRMED' if r21 <= 1.3 else '> 1.3: NOT confirmed'})")
    if batch4 and 4 in results:
        r41 = results[4]["wall_s"] / results[1]["wall_s"]
        print(f"  batch-4 / batch-1 wall ratio: {r41:.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"name": name, "model_id": model_id, "quant": "nf4" if quant else "bf16",
            "new": new, "reps": reps, "ratio_2v1": r21,
            "ratio_4v1": (results[4]["wall_s"] / results[1]["wall_s"]) if 4 in results else None,
            "batches": {str(k): v for k, v in results.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="1.5b,7b")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-batch4", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"[batched-receipts] device={torch.cuda.get_device_name(0)}, new={args.new}, "
          f"reps={args.reps} (1 warmup discarded), greedy, forced length")

    out = []
    if "1.5b" in args.models:
        out.append(bench_model("Qwen2.5-1.5B bf16", "Qwen/Qwen2.5-1.5B-Instruct",
                               False, args.new, args.reps, not args.no_batch4))
    if "7b" in args.models:
        out.append(bench_model("Qwen2.5-7B nf4", "Qwen/Qwen2.5-7B-Instruct",
                               True, args.new, args.reps, not args.no_batch4))

    print("\n=== summary ===")
    for r in out:
        print(f"  {r['name']:<22} batch2/batch1 = {r['ratio_2v1']:.3f}"
              + (f"   batch4/batch1 = {r['ratio_4v1']:.3f}" if r["ratio_4v1"] else ""))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[batched-receipts] wrote {args.json_out}")


if __name__ == "__main__":
    main()
