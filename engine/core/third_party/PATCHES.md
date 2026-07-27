# Vendored llama.cpp — pin + patch

`third_party/llama.cpp` is a **gitignored, local-only** build dependency — it is NOT committed (the whole
~340k-line upstream tree stays out of the repo). Only three things here are tracked: this file,
`bootstrap_llama.py`, and `patches/*.patch`. That keeps the repo clean while the build stays exactly
reproducible.

## Pin

| | |
|---|---|
| upstream | https://github.com/ggml-org/llama.cpp |
| commit | `88a39274ecf88ba11686acd357b59685b1cbf03d` |
| tag | `b9606` (2026-06-12) |
| ggml | 0.15.0 |

The original checkout kept no `.git`, so this SHA was **reconstructed** (2026-07-13) by matching the
patch's pre-image blob hashes — `include/llama.h` `27e4806`, `src/llama-context.cpp` `168dbab`,
`src/llama-context.h` `853052b` — against upstream trees, and **verified** with `git apply --check` (clean).

## Reconstruct the build

    python engine/core/third_party/bootstrap_llama.py          # shallow-clone the pin + apply patches
    python engine/core/third_party/bootstrap_llama.py --force  # wipe and redo

Every invocation verifies an existing tree rather than trusting a nonempty directory. The checkout must
retain its `.git` metadata, resolve to the pinned commit, pass `git diff --check`, and allow every tracked
patch to be reverse-applied. An unverifiable or drifted tree is refused and must be reconstructed with
`--force`.

## The patch (`patches/0001-llama_get_logits_tensor.patch`)

One small **additive** change (marked `CLOZN PATCH` in-source across `include/llama.h` +
`src/llama-context.{cpp,h}`): `llama_get_logits_tensor` + `llama_set_skip_raw_logits`. They let the §4.3
confidence-select kernel read the per-step logits **on-device** and skip llama's decode-time device→host
copy — the enabler for a zero-copy multi-position (diffusion) speedup.

It is optional: `engine/core/src/model_ggml.cpp` calls the accessors only behind a runtime `skip_d2h`
flag, which the served `run`/`serve` paths never set — without the patch the build would fall back to the
host-logits path (see the flag-guard note below). Defaults unchanged either way.

## Re-pinning to a newer llama.cpp

1. Bump `TAG`/`COMMIT` in `bootstrap_llama.py`, run `--force`.
2. If a patch no longer applies, regenerate it against the new base:
   `git -C llama.cpp diff > patches/0001-llama_get_logits_tensor.patch`.

## TODO

- Upstream the accessors as a proper PR — ideally a **multi-position** GPU logits interface — so the
  feature needs no private patch at all (keeps the fork at zero). WIP.
- Optionally gate the `model_ggml.cpp` call sites behind a `CLOZN_ZEROCOPY` CMake flag (default off), so a
  fully-stock llama.cpp with **no** patch also builds cleanly.
