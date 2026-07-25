"""PyTorch/Hugging Face activation steering adapter."""
from __future__ import annotations

import json
import os

import torch

from clozn._io import atomic_write_json

from . import axes

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SteeringControl:
    """Computes and applies tone-axis steering vectors on a loaded causal LM."""

    def __init__(self, model, tok, layer: int | None = None):
        self.model, self.tok = model, tok
        n = model.config.num_hidden_layers
        self.layer = layer if layer is not None else n // 2
        self.vecs: dict[str, torch.Tensor] = {}
        self.custom: dict[str, dict] = {}
        self.strength: dict[str, float] = {}
        self.base = 1.0
        self.resid_norm = 0.0
        self._handle = None
        self._layers = model.model.layers

    @torch.no_grad()
    def _last_resid(self, system: str, user: str) -> torch.Tensor:
        """Residual at the last prompt token at self.layer."""
        enc = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # transformers 4.x returns a bare tensor here; 5.x returns a BatchEncoding -- normalize to the ids tensor.
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1]
        return hs[0, -1].float()

    @torch.no_grad()
    def compute(self, seeds=None) -> dict:
        """Build each configured axis as a unit direction and calibrate this model's scale."""
        seeds = axes.SEED_PROMPTS if seeds is None else seeds
        out, norms = {}, []
        for name, ax in axes.AXES.items():
            pv = [self._last_resid(ax["pos"], s) for s in seeds]
            nv = [self._last_resid(ax["neg"], s) for s in seeds]
            pos, neg = torch.stack(pv).mean(0), torch.stack(nv).mean(0)
            norms += [float(x.norm()) for x in pv + nv]
            d = pos - neg
            self.vecs[name] = d / (d.norm() + 1e-8)
            out[name] = round(float(d.norm()), 2)
        self.resid_norm = sum(norms) / len(norms)
        self.base = 0.85 * self.resid_norm
        return {"raw_norms": out, "resid_norm": round(self.resid_norm, 1), "base": round(self.base, 2)}

    @torch.no_grad()
    def add_custom(self, name: str, pos: str, neg: str, mx: float = 0.5, seeds=None) -> dict:
        """Register a user-defined dial using the same diff-of-means recipe as built-in axes."""
        seeds = axes.SEED_PROMPTS if seeds is None else seeds
        name = name.strip()[:24]
        pv = [self._last_resid(pos, s) for s in seeds]
        nv = [self._last_resid(neg, s) for s in seeds]
        d = torch.stack(pv).mean(0) - torch.stack(nv).mean(0)
        self.vecs[name] = d / (d.norm() + 1e-8)
        self.custom[name] = {"pos": pos, "neg": neg, "max": float(mx), "poles": [name, "neutral"]}
        return self.custom[name]

    def remove_custom(self, name: str):
        self.custom.pop(name, None)
        self.vecs.pop(name, None)
        self.strength.pop(name, None)

    def save_custom(self, path: str):
        # Atomic: same silent-data-loss class as the engine adapter's writers (open("w") truncates
        # before json.dump can raise; load_custom's except-pass then eats the loss without a word).
        atomic_write_json(path, {k: {"pos": v["pos"], "neg": v["neg"], "max": v["max"]}
                                 for k, v in self.custom.items()})

    def load_custom(self, path: str):
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self.add_custom(k, v["pos"], v["neg"], float(v.get("max", 0.5)))
        except Exception:
            pass

    MAX_DELTA_FRAC = 0.75

    def set(self, name: str, value: float):
        """Set a slider value, capped by the built-in or custom dial maximum."""
        mx = (axes.AXES.get(name) or self.custom.get(name) or {}).get("max", 1.5)
        self.strength[name] = max(-mx, min(mx, float(value)))

    def clear(self):
        self.strength = {}

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        add = None
        for name, s in self.strength.items():
            if s and name in self.vecs:
                v = (s * self.base * self.vecs[name]).to(h.device, h.dtype)
                add = v if add is None else add + v
        if add is not None:
            h = h + self._cap_delta(add, h)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    def _cap_delta(self, add: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Bound summed steering to MAX_DELTA_FRAC * residual norm per position."""
        add_norm = add.norm()
        if float(add_norm) <= 0.0:
            return add
        ceil = self.MAX_DELTA_FRAC * h.norm(dim=-1, keepdim=True)
        scale = torch.clamp(ceil / add_norm, max=1.0)
        return add * scale

    def engage(self):
        if self._handle is None:
            self._handle = self._layers[self.layer].register_forward_hook(self._hook)

    def disengage(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def active(self) -> dict:
        return {k: v for k, v in self.strength.items() if v}

    def save_state(self, path: str):
        # Atomic -- see save_custom.
        atomic_write_json(path, self.strength)

    def load_state(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        try:
            with open(path, encoding="utf-8") as f:
                self.strength = {k: float(v) for k, v in json.load(f).items()}
            return True
        except Exception:
            return False

    @torch.no_grad()
    def generate(self, prompt: str, max_new=100) -> str:
        """Generate a single turn on the bare backbone with any engaged steering hook active."""
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # transformers 4.x returns a bare tensor here; 5.x returns a BatchEncoding -- normalize to the ids tensor.
        ids = (enc if isinstance(enc, torch.Tensor) else enc["input_ids"]).to(DEV)
        out = self.model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=self.tok.eos_token_id or 0,
        )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
