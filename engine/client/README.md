# engine/client — Python SDK for the clozn-server white-box API

`clozn_engine.py` is the thin Python seam over the C++ engine's HTTP interface
(`engine/core/serve/clozn_server.cpp`). It exists so the research stack (SAE discovery,
feature circuits, concept probes, all numpy already) can read from and write into the
**live** ggml/llama.cpp model instead of a separate HF copy.

It drives the read -> edit -> write -> observe loop closed by GAP #1:

| method | endpoint | what it does |
| --- | --- | --- |
| `harvest(text, layer=None)` | `POST /harvest` | read every token's residual at the tap layer (one causal forward) |
| `write_state(text, layer, positions, values)` | `POST /state` | overwrite those positions' residual, run again, report how the next-token logits moved |
| `edit_and_observe(text, transform, ...)` | both | the whole loop in one call: harvest, apply `transform(acts)->acts`, write the changed rows back at the same layer |
| `intervene(prompt, concept=/vector=, coef, ...)` | `POST /intervene` | steer a generation with a named concept probe or a raw direction |
| `complete(prompt, **params)` | `POST /v1/completions` | a plain generation (no white-box) |
| `health()` | `GET /health` | `{status, model, mode}` |

Dependencies: the standard library (HTTP/JSON/base64) plus numpy. No `requests`.

The tensor wire format is `{dtype:"float32", shape, data}` where `data` is base64 of the
raw little-endian float32 bytes; `decode_tensor` reconstructs it with a single
`np.frombuffer('<f4')`. Writes go back as a flat, position-major list of floats (the
server checks `len(values) == len(positions) * n_embd`).

## Use it

```python
from clozn_engine import EngineClient
import numpy as np

eng = EngineClient(port=8080)
h = eng.harvest("The capital of France is")     # h.activations: [n_tokens, n_embd] float32

# Edit the last token's residual and observe the prediction shift.
def amplify(acts):
    acts[-1] *= 5.0
    return acts

_, obs = eng.edit_and_observe("The capital of France is", transform=amplify)
print(obs.summary())     # moved_l2, baseline top-3, edited top-3, whether the top-1 flipped
```

## Validate

```
python clozn_engine.py --selftest    # offline: the wire codec is exact, no server needed
python clozn_engine.py --demo --port 8091   # live: a full round-trip against a running server
```

The `--demo` prints an **identity-write control** next to the real edit: writing the
harvested rows back unchanged must move `moved_l2 ~= 0` (it round-trips through base64
losslessly), while a real edit moves the logits and can flip the argmax. That control is
the end-to-end correctness check.

## Worked example: discover in Python, verify on the live model

`probe_and_patch.py` is why the SDK exists. It (1) harvests token activations for two
contrasting corpora from the engine, (2) fits a diff-in-means direction in numpy on the
engine's *own* residuals, (3) writes that direction into a neutral prompt's last position
via `/state`, and (4) measures the prediction shift against a **magnitude-matched random
control**. The engine can only do diff-in-means internally on a fixed corpus; here the
discovery is arbitrary Python (swap in an SAE encode, a PCA component, a learned probe).

```
python probe_and_patch.py --port 8091 --layer 14 --coef 0.25
```

On open-dcoder-0.5b, patching `"My favorite number is"` with the number-vs-prose
direction pulls a digit into the top predictions while a random direction of the same norm
does not:

```
  baseline           digit_mass=0.000   top: ' is' 0.953, ' are' 0.018, ' if' 0.014
  number-direction   digit_mass=0.225   top: ' ' 0.521, '3' 0.225, ' is' 0.185
  random-direction   digit_mass=0.000   top: ' is' 0.933, ' if' 0.032, ' i' 0.013
```

The random control is the honesty baseline: it has the same perturbation magnitude, so the
extra movement toward digits is the *direction*, not the nudge. `--coef` above ~1 over-
injects and garbles the output (it overwrites the row's own signal); ~0.25 is a clean nudge.

## Bring up a server to test against

```
# from engine/core, with the ggml/llama DLLs on PATH (build-ggml-cpu/bin/Release):
clozn-server <model.gguf> --port 8091 --ctx 512
```

`/harvest` and `/state` work on either a diffusion (LLaDA/Dream/open-dcoder) or an
autoregressive (Qwen/Llama) GGUF; they force a causal forward locally regardless of mode.
