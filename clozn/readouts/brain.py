"""brain_readout.py -- the concept readout over a loaded Qwen-7B + SAE, as a reusable class so the brain
window and the unified clozn server share ONE implementation (extracted from brain_server_7b.py).

Given a prompt it generates a reply in a per-session conversation and surfaces the genuinely-relevant
concepts the model engaged -- pulled live from the FULL 131k feature space, named by Neuronpedia, and
filtered for specificity (normalize to each feature's own peak; drop broad/high-frequency features by
frac_nonzero; drop discourse/pronoun labels). Off-map concepts carry their nearest atlas node for placing.
"""
import json
import os
import threading

import numpy as np
import torch

from clozn.readouts.atlas_concepts import content_word
from clozn.readouts.sae7b import DEV, feats7b

ARTIFACT_NNZ = 600
DISCOURSE_TERMS = ("question", "answer", "discuss", "conversation", "request", "asking", "writing",
                   "publish", "article", "journal", "blog", "sharing information", "connecting",
                   "decision", "choices", "instruction", "prompt", "response", "explanation", "summary",
                   "website", "online", "comment", "formatting", "markdown")
GENERIC_SINGLE = {"i", "our", "we", "you", "my", "your", "me", "us", "it", "they", "them",
                  "the", "a", "an", "this", "that", "and", "or", "of", "to", "in", "s", "t"}


class BrainReadout:
    def __init__(self, model, tok, sae, demo_dir, research_dir):
        self.model, self.tok, self.sae = model, tok, sae
        self.lock = threading.Lock()
        self.sessions = {}
        atlas = json.load(open(os.path.join(demo_dir, "atlas_emergent.json"), encoding="utf-8"))
        self.concepts = atlas["meta"]["concepts"]
        self.fids = [n["id"] for n in atlas["nodes"]]
        self.fid2concept = {n["id"]: self.concepts[n["cluster"]] for n in atlas["nodes"]}
        self.fid2peak = {n["id"]: float(n.get("peak", 1.0)) for n in atlas["nodes"]}
        lbl = json.load(open(os.path.join(research_dir, "np_labels_l15.json"), encoding="utf-8"))
        st = json.load(open(os.path.join(research_dir, "np_stats_l15.json")))
        d = sae.d_sae
        self.lbl = lbl
        self.maxact = np.zeros(d, np.float32)
        self.frac = np.ones(d, np.float32)
        self.haslabel = np.zeros(d, bool)
        self.blocked = np.zeros(d, bool)
        for k, (ma, fr) in st.items():
            i = int(k)
            self.maxact[i] = ma
            self.frac[i] = fr
        for k, lab in lbl.items():
            i = int(k)
            self.haslabel[i] = True
            low = lab.strip().lower()
            if low in GENERIC_SINGLE or any(t in low for t in DISCOURSE_TERMS):
                self.blocked[i] = True
        self.atlas_fid_arr = np.array(self.fids)
        ad = sae.W_dec_cpu[self.fids].float().numpy()
        self.atlas_dirs = ad / (np.linalg.norm(ad, axis=1, keepdims=True) + 1e-8)
        self.atlas_set = set(self.fids)
        print(f"brain readout ready ({int(self.haslabel.sum())} labelled features, {len(self.fids)} atlas nodes)", flush=True)

    def dynamic_considered(self, fmax, k=14, rel_min=0.18):
        rel = np.where(self.maxact > 0, fmax / np.maximum(self.maxact, 1e-6), 0.0)
        elig = (fmax > 0) & self.haslabel & (self.frac < 0.02) & (~self.blocked) & (rel >= rel_min)
        ids = np.where(elig)[0]
        if not len(ids):
            return []
        order = ids[np.argsort(-rel[ids])][:k]
        dirs = self.sae.W_dec_cpu[order.tolist()].float().numpy()
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
        near = self.atlas_fid_arr[(dirs @ self.atlas_dirs.T).argmax(1)]
        return [{"id": int(f), "label": self.lbl[str(int(f))], "rel": round(float(rel[f]), 3),
                 "in_atlas": int(f) in self.atlas_set, "near": int(near[j])}
                for j, f in enumerate(order.tolist())]

    @torch.no_grad()
    def think(self, text, sid):
        with self.lock:
            hist = self.sessions.setdefault(sid, [])
            hist.append({"role": "user", "content": text})
            ids = self.tok.apply_chat_template(hist[-16:], add_generation_prompt=True,
                                               return_tensors="pt").to(DEV)
            gen = self.model.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
                                      pad_token_id=self.tok.eos_token_id)
            ans = self.tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
            hist.append({"role": "assistant", "content": ans})
            if len(self.sessions) > 64:
                for kk in list(self.sessions)[:-32]:
                    self.sessions.pop(kk, None)
            pieces, feats = feats7b(text, self.tok, self.model, self.sae)
            f = feats.cpu().numpy()
            f[(f > 0).sum(1) > ARTIFACT_NNZ] = 0
            keep = np.array([content_word(p) for p in pieces])
            fmax = f[keep].max(0) if keep.any() else f.max(0)
            acts = {}
            for fid in self.fids:
                v = float(fmax[fid])
                if v > 0:
                    rel = min(1.5, v / max(self.fid2peak[fid], 1e-6))
                    if rel >= 0.25:
                        acts[fid] = round(rel, 3)
            considered = self.dynamic_considered(fmax)
            return {"acts": acts, "considered": considered,
                    "concepts": [{"name": c["label"]} for c in considered[:6]],
                    "output": ans, "turn": len(hist) // 2}

    @torch.no_grad()
    def concepts_from_engine(self, text, ec, layer=15):
        """The SAME concept readout, but the activations are HARVESTED from the real C++ engine (a
        clozn-server on the Qwen GGUF) instead of the Python model -- concepts read from the actual
        ggml runtime. Verified: dragon -> dragon/fear/RPG, grief at a funeral -> crying/funeral/grief."""
        h = ec.harvest(text, layer=layer)
        f = self.sae.encode(torch.tensor(np.asarray(h.activations), device=DEV)).cpu().numpy()
        f[(f > 0).sum(1) > ARTIFACT_NNZ] = 0
        fmax = f.max(0)
        acts = {}
        for fid in self.fids:
            v = float(fmax[fid])
            if v > 0:
                rel = min(1.5, v / max(self.fid2peak[fid], 1e-6))
                if rel >= 0.25:
                    acts[fid] = round(rel, 3)
        considered = self.dynamic_considered(fmax)
        return {"acts": acts, "considered": considered,
                "concepts": [{"name": c["label"]} for c in considered[:6]],
                "source": "engine", "layer": int(h.layer)}

    @torch.no_grad()
    def concepts_only(self, text):
        """Read the concepts a prompt engages -- the same specificity-filtered readout as think(), but with
        NO generation. Used to annotate a chat reply with what actually fired inside, cheaply."""
        pieces, feats = feats7b(text, self.tok, self.model, self.sae)
        f = feats.cpu().numpy()
        f[(f > 0).sum(1) > ARTIFACT_NNZ] = 0
        keep = np.array([content_word(p) for p in pieces])
        fmax = f[keep].max(0) if keep.any() else f.max(0)
        considered = self.dynamic_considered(fmax, k=8, rel_min=0.26)   # stricter -> fewer, cleaner concepts
        common = {"like", "work", "works", "copy", "thing", "things", "way", "ways", "make", "get", "good"}
        considered = [c for c in considered if c["label"].strip().lower() not in common][:6]
        return {"considered": considered, "concepts": [c["label"] for c in considered]}

    def reset(self, sid):
        with self.lock:
            self.sessions[sid] = []
        return {"ok": True}
