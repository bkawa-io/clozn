// clozn/probe.hpp — training-free concept probes over hidden states (white-box Tier 2).
// A probe is a unit direction in the model's (standardized) activation space; projecting a
// position's hidden state onto it scores how strongly that concept is present THIS pass. These
// are diff-in-means probes — the model-agnostic, gradient-free probe: per-concept labeled
// averages minus the rest, in the model's own space. Pure backend-free DATA: an adapter-side
// calibration (which captures activations) builds them; the pass loop consumes them, so the
// scheduler/generate stay model-agnostic. Mirrors lab category probing (p4_dream_probe).
#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace clozn {

struct ConceptProbes {
    std::vector<std::string> names;  // K concept names (become the StepFeatures `features`)
    int n_embd = 0;
    std::vector<float> mean;     // [n_embd] feature mean (centering); empty => probes unbuilt
    std::vector<float> inv_std;  // [n_embd] 1/std per dim (standardize)
    std::vector<float> dirs;     // [K * n_embd] unit direction per concept (standardized space)

    int size() const { return static_cast<int>(names.size()); }
    bool ready() const { return !names.empty() && n_embd > 0 &&
                                dirs.size() == names.size() * static_cast<size_t>(n_embd); }

    // Project one hidden state h ([n_embd]) onto every concept => K scores. Standardize with the
    // calibrated mean/inv_std, then dot with each unit direction. A positive score means the
    // concept is present at that position (relative to the calibration baseline).
    std::vector<float> project(const float* h) const {
        std::vector<float> out(names.size(), 0.0f);
        for (size_t k = 0; k < names.size(); ++k) {
            const float* d = dirs.data() + k * static_cast<size_t>(n_embd);
            double s = 0.0;
            for (int i = 0; i < n_embd; ++i)
                s += (static_cast<double>(h[i]) - mean[i]) * inv_std[i] * d[i];
            out[k] = static_cast<float>(s);
        }
        return out;
    }

    // Raw-activation-space UNIT direction for concept k — ADD alpha*this to a hidden state to raise
    // the concept's score. The standardized direction rescaled by inv_std, unit-normalized (since
    // d(score)/d(h_i) = inv_std[i]*dir[i]). The causal WRITE paired with the read-only project():
    // the steering vector for a control vector.
    std::vector<float> steer_vector(int k) const {
        std::vector<float> v(n_embd, 0.0f);
        if (k < 0 || k >= size() || mean.empty()) return v;
        const float* d = dirs.data() + static_cast<size_t>(k) * n_embd;
        double nrm = 0.0;
        for (int i = 0; i < n_embd; ++i) { v[i] = inv_std[i] * d[i]; nrm += static_cast<double>(v[i]) * v[i]; }
        nrm = std::sqrt(nrm > 1e-12 ? nrm : 1e-12);
        for (int i = 0; i < n_embd; ++i) v[i] = static_cast<float>(v[i] / nrm);
        return v;
    }
};

}  // namespace clozn
