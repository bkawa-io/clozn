"""slotmem_qwen.py -- the GLASS-BOX SLOT MEMORY, ported from GPT-2 (p15/p17/p19) to Qwen2.5.

The don't-fuse winner made real on the studio's model family, plus the three rungs the spikes never
built:
  1. SURPRISE-GATED WRITES (the Titans rung): a fact is written only if the model is surprised by it
     (-log P(answer|cue) above a threshold) -- known facts are SKIPPED, not stored.
  2. CONFIDENCE GATE at read (p19's fix): if the best key similarity is below a calibrated floor the
     memory ABSTAINS instead of confidently retrieving the wrong fact.
  3. MULTI-TOKEN ANSWERS: does injecting the FIRST answer token's direction elicit the whole answer
     in generation? (p15 was single-token only.)

Mechanism (p17-corrected): store = an explicit list of {key, value, label}. WRITE: key = the residual
at the CUE'S LAST TOKEN at layer L (the same position a query produces -- the p16 'capacity wall' was
a write/read position mismatch, never repeat it). value = the answer's unembedding direction (legible
by construction: logit-lens decodes every stored value to its answer). READ: a forward hook takes the
query's last-position residual, hard top-1 over unit keys, and adds eta * value at that position.
eta = INJECT_FRAC x the layer's mean residual norm.

Receipts battery (the p15 discipline, re-earned on Qwen): baseline floor, recall, SPECIFICITY (wrong-
fact-only in memory => baseline), SHUFFLED-KEY NULL (permuted keys => keyed addressing, not bias),
SURGICAL DELETE (target drops, others bit-identical), paraphrase + gate behavior. One model, one seed;
caveats loud.

    C:\\Users\\brigi\\src\\clozn\\.venv\\Scripts\\python.exe research/slotmem_qwen.py [--smoke]
"""
from __future__ import annotations
import math
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

INJECT_FRAC = 1.5        # eta = this x mean residual norm at the tap layer (0.6 lifted P(ans) 17x but
                         # lost argmax on Qwen -- deeper stack + RMSNorm dilute; 1.5 is the working point)
SURPRISE_MIN = 3.0       # write gate: -log P(first answer token | cue) in nats; known facts sit far below
GATE_STD = 2.0           # read gate: abstain if best CENTERED sim < cross_mean + GATE_STD * cross_std


class SlotMem:
    """The explicit, inspectable store + the read/write machinery on a frozen HF causal LM."""

    def __init__(self, model_name: str, layer: int):
        path = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else model_name
        four_bit = "7b" in model_name.lower() and DEV == "cuda"   # 7B needs nf4 on a 16GB card (the
        print(f"[load] {model_name} ({'4-bit nf4' if four_bit else 'bf16'}) layer={layer}", flush=True)
        tok = AutoTokenizer.from_pretrained(path)
        if four_bit:                                              # studio's config -- voice_middle.Rig)
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(        # never .to(DEV) a quantized model
                path, quantization_config=bnb, device_map={"": 0}).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._setup(model, tok, layer)

    @classmethod
    def from_shared(cls, model, tok, layer: int) -> "SlotMem":
        """Build a SlotMem on an ALREADY-LOADED backbone (the studio's Qwen-7B nf4: SUB.memory.model /
        .tok) -- one model behind the concept readout, the memory prefix AND the fact store, exactly the
        SelfTeach(model=..., tok=...) share. No second 7B, no second HF download. The model must expose
        `.model.layers[layer]` and `.lm_head.weight` (Qwen2.5 does; the slotmem findings validated L18 on
        this exact nf4 config). The caller keeps ownership of the model; only close() removes OUR hook."""
        self = cls.__new__(cls)                                  # skip __init__'s model load
        self._setup(model, tok, layer)
        return self

    def _setup(self, model, tok, layer: int):
        """Everything after the backbone exists: the tap hook, W_U, an empty store, and the eta measured
        once on a neutral text. Shared by __init__ (fresh load) and from_shared (reuse)."""
        self.tok, self.model = tok, model
        self.layer = layer
        self.W_U = self.model.lm_head.weight        # [V, H] -- values come from here (legible)
        self.entries: list[dict] = []               # the store: {key(unit), value(unit), label, ans_ids}
        self.gate_floor: float | None = None        # calibrated abstain threshold on key similarity
        self._inject: torch.Tensor | None = None    # set per-read by the hook
        self._h = self.model.model.layers[layer].register_forward_hook(self._hook)
        # eta: a fixed fraction of the layer's typical residual norm (measured once on a neutral text)
        with torch.no_grad():
            r = self._resid_last("The weather this afternoon is calm and the streets are quiet.")
        self.eta = INJECT_FRAC * float(r.norm())
        print(f"  resid_norm~{float(r.norm()):.0f} eta={self.eta:.0f}", flush=True)

    def _hook(self, mod, inp, out):
        if self._inject is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] = h[:, -1, :] + self._inject.to(h.dtype)   # add at the query position only
        return (h,) + out[1:] if isinstance(out, tuple) else h

    @torch.no_grad()
    def _resid_last(self, text: str) -> torch.Tensor:
        """Residual at the LAST token of `text`, at the tap layer (query-time-consistent -- p17)."""
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1][0]
        return hs[-1].float()

    @torch.no_grad()
    def _next_dist(self, text: str) -> torch.Tensor:
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEV)
        return torch.softmax(self.model(ids).logits[0, -1].float(), -1)

    @torch.no_grad()
    def surprise(self, cue: str, ans_ids: list[int]) -> float:
        """-log P(first answer token | cue) in nats, no memory active -- the write-gate signal."""
        p = float(self._next_dist(cue)[ans_ids[0]])
        return -math.log(max(p, 1e-12))

    def write(self, cue: str, answer: str, gate: bool = True) -> dict:
        """Store cue->answer. With gate=True, skip when the model already knows it (low surprise)."""
        ans_ids = self.tok.encode(answer, add_special_tokens=False)
        s = self.surprise(cue, ans_ids)
        if gate and s < SURPRISE_MIN:
            return {"written": False, "surprise": round(s, 2)}
        k = self._resid_last(cue)
        v = self.W_U[ans_ids[0]].float()
        self.entries.append({"key": k / (k.norm() + 1e-8), "value": v / (v.norm() + 1e-8),
                             "label": cue + " ->" + answer, "ans_ids": ans_ids, "cue": cue, "answer": answer})
        return {"written": True, "surprise": round(s, 2)}

    def _centered(self, pool: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys CENTERED by their mean, then renormalized. Qwen's last-token residuals are anisotropic
        (all cues end alike, raw cross-sim ~0.68 -- p17 found centering unnecessary on GPT-2; Qwen needs
        it): subtracting the shared component makes similarity subject-driven. Returns (K_centered, mu)."""
        K = torch.stack([e["key"] for e in pool])
        mu = K.mean(0)
        Kc = K - mu
        Kc = Kc / (Kc.norm(dim=-1, keepdim=True) + 1e-8)
        return Kc, mu

    def calibrate_gate(self):
        """Abstain floor over CENTERED similarities: cross_mean + GATE_STD*cross_std -- a drifted query
        must beat the unrelated-cue crowd by a clear margin or the memory abstains."""
        if len(self.entries) < 3:
            self.gate_floor = 0.0
            return
        Kc, _ = self._centered(self.entries)
        cross = (Kc @ Kc.T).masked_fill(torch.eye(len(Kc), device=DEV, dtype=torch.bool), float("nan"))
        vals = cross[~torch.isnan(cross)]
        self.gate_floor = float(vals.mean() + GATE_STD * vals.std())
        print(f"  gate_floor={self.gate_floor:.3f} (CENTERED cross-sim mean {float(vals.mean()):.3f} "
              f"std {float(vals.std()):.3f})", flush=True)

    @torch.no_grad()
    def read(self, query: str, gated: bool = False, entries: list | None = None) -> dict:
        """Hard top-1 addressing over CENTERED keys; returns the injected next-token dist + which entry
        fired (or abstained)."""
        pool = self.entries if entries is None else entries
        if not pool:
            return {"dist": self._next_dist(query), "hit": None, "sim": None, "abstained": True}
        Kc, mu = self._centered(pool)
        q = self._resid_last(query)
        q = q / (q.norm() + 1e-8)
        qc = q - mu
        qc = qc / (qc.norm() + 1e-8)
        sims = Kc @ qc
        best = int(sims.argmax())
        sim = float(sims[best])
        if gated and self.gate_floor is not None and sim < self.gate_floor:
            return {"dist": self._next_dist(query), "hit": None, "sim": sim, "abstained": True}
        self._inject = self.eta * pool[best]["value"]
        try:
            dist = self._next_dist(query)
        finally:
            self._inject = None
        return {"dist": dist, "hit": best, "sim": sim, "abstained": False}

    @torch.no_grad()
    def emit(self, query: str, max_new: int = 6) -> str:
        """Short greedy generation with a VALUE SCHEDULE: the hit entry's first answer token direction
        injected at decode step 1, its second (when the answer is multi-token) at step 2, then clean
        continuation -- the rung-2 fix (first-token-only elicited multi answers just 4/7)."""
        r = self.read(query)
        ids = self.tok(query, return_tensors="pt").input_ids.to(DEV)
        seq = ids
        if r["hit"] is not None:
            e = self.entries[r["hit"]]
            sched = [e["value"]]
            if len(e["ans_ids"]) > 1:                          # second-token direction, unit-normalized
                v2 = self.W_U[e["ans_ids"][1]].float()
                sched.append(v2 / (v2.norm() + 1e-8))
            for vec in sched:                                  # one injected greedy step per scheduled token
                self._inject = self.eta * vec
                try:
                    nxt = self.model(seq).logits[0, -1].argmax()
                finally:
                    self._inject = None
                seq = torch.cat([seq, nxt.view(1, 1)], 1)
        remaining = max_new - (seq.shape[1] - ids.shape[1])
        out = seq if remaining <= 0 else self.model.generate(
            seq, attention_mask=torch.ones_like(seq), max_new_tokens=remaining,
            do_sample=False, pad_token_id=self.tok.eos_token_id or 0)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    # ---- persistence (the ~/.clozn/slotmem.pt rung). The pure work lives in pack_store/unpack_store
    # so a model-free unit test can cover the round-trip; these methods only add device + layer checks.
    def save(self, path: str) -> str:
        """torch.save the store (entries + layer/eta/gate_floor) to `path`. Returns the resolved path."""
        path = os.path.expanduser(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(pack_store(self.entries, self.layer, self.eta, self.gate_floor), path)
        return path

    def load(self, path: str) -> int:
        """Restore a store written by save(). Refuses a layer mismatch (keys are residuals OF a layer;
        reading them at another layer is silent garbage). Replaces entries/eta/gate_floor; returns N."""
        state = torch.load(os.path.expanduser(path), map_location="cpu")
        if state.get("layer") != self.layer:
            raise ValueError(f"store was written at layer {state.get('layer')}, "
                             f"this SlotMem taps layer {self.layer}")
        entries, meta = unpack_store(state, device=DEV)
        self.entries = entries
        self.eta = meta["eta"]
        self.gate_floor = meta["gate_floor"]
        return len(entries)

    def close(self):
        self._h.remove()


# ---- store (de)serialization: pure, model-free, bit-exact -------------------------------------------
STORE_VERSION = 1


def pack_store(entries: list, layer: int, eta: float, gate_floor) -> dict:
    """The store as a torch.save-able dict: float32 CPU tensors + plain python only (survives
    torch.load(weights_only=True); float32 CPU<->CUDA moves are memcpys, so round-trips are bit-exact)."""
    return {
        "version": STORE_VERSION,
        "layer": int(layer),
        "eta": float(eta),
        "gate_floor": None if gate_floor is None else float(gate_floor),
        "keys": torch.stack([e["key"].detach().float().cpu() for e in entries]) if entries else torch.empty(0),
        "values": torch.stack([e["value"].detach().float().cpu() for e in entries]) if entries else torch.empty(0),
        "ans_ids": [list(map(int, e["ans_ids"])) for e in entries],
        "labels": [e["label"] for e in entries],
        "cues": [e["cue"] for e in entries],
        "answers": [e["answer"] for e in entries],
    }


def unpack_store(state: dict, device: str = "cpu") -> tuple[list, dict]:
    """Inverse of pack_store -> (entries, {layer, eta, gate_floor})."""
    if state.get("version") != STORE_VERSION:
        raise ValueError(f"unknown slotmem store version {state.get('version')!r}")
    entries = [{"key": state["keys"][i].to(device), "value": state["values"][i].to(device),
                "label": state["labels"][i], "ans_ids": list(state["ans_ids"][i]),
                "cue": state["cues"][i], "answer": state["answers"][i]}
               for i in range(len(state["labels"]))]
    return entries, {"layer": state["layer"], "eta": state["eta"], "gate_floor": state["gate_floor"]}
