// clozn/whitebox.hpp — backend-free white-box read helpers (Tier 1 + 2), shared by every
// generation loop. They turn one forward pass's ForwardResult into the §5.1 observation events:
//   features_from -> StepFeatures  (concept-probe scores per active position)
//   lens_from     -> StepLens      (logit-lens top-k candidates per requested position)
// Both operate purely on ForwardResult + ConceptProbes (no model backend, no scheduler), so the
// diffusion loops (generate/infill/denoise) and the autoregressive loop (generate_ar) emit IDENTICAL
// white-box events from the same code. Pure observation (DESIGN invariant 2): never touches the board.
#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "clozn/events.hpp"  // StepFeatures, StepLens
#include "clozn/model.hpp"   // ForwardResult
#include "clozn/probe.hpp"   // ConceptProbes

namespace clozn {

// Concept-feature event for one pass: when the adapter filled activations (emit_activations on),
// project each active-slot hidden state onto the concept probes (training-free diff-in-means) and
// emit a StepFeatures event — the per-slot concept scores this pass. With no probes ready it falls
// back to a single "norm" feature (hidden-state L2 magnitude). Returns nullopt on the default path
// (no activations) => zero cost, goldens untouched.
inline std::optional<StepFeatures> features_from(const ForwardResult& fwd, int t, int block,
                                                 const ConceptProbes* probes) {
    if (fwd.activations.empty() || fwd.n_embd <= 0) return std::nullopt;
    StepFeatures sf;
    sf.t = t;
    sf.block = block;
    sf.positions = fwd.act_rows;
    const int rows = static_cast<int>(fwd.act_rows.size());
    if (probes && probes->ready() && probes->n_embd == fwd.n_embd) {
        sf.features = probes->names;
        const int K = probes->size();
        sf.scores.resize(static_cast<size_t>(rows) * K);
        for (int r = 0; r < rows; ++r) {
            const float* h = fwd.activations.data() + static_cast<size_t>(r) * fwd.n_embd;
            const std::vector<float> sc = probes->project(h);
            for (int k = 0; k < K; ++k) sf.scores[static_cast<size_t>(r) * K + k] = sc[k];
        }
    } else {
        sf.features = {"norm"};
        sf.scores.reserve(rows);
        for (int r = 0; r < rows; ++r) {
            const float* h = fwd.activations.data() + static_cast<size_t>(r) * fwd.n_embd;
            double s = 0.0;
            for (int i = 0; i < fwd.n_embd; ++i) s += static_cast<double>(h[i]) * h[i];
            sf.scores.push_back(static_cast<float>(std::sqrt(s)));
        }
    }
    return sf;
}

// Raw-activation event for one pass: the unprojected per-position hidden state (the heavy `state`
// of the state-stream protocol). Filled only when the adapter has emit_activations on; returns
// nullopt on the default path => zero cost, goldens untouched. The SSE layer turns this into the
// `StateStep.state` tensor (base64), included only on demand (state="full"). Mirrors features_from
// but skips the concept projection — it carries the activations verbatim.
inline std::optional<StepActivations> activations_from(const ForwardResult& fwd, int t, int block) {
    if (fwd.activations.empty() || fwd.n_embd <= 0) return std::nullopt;
    StepActivations sa;
    sa.t = t;
    sa.block = block;
    sa.positions = fwd.act_rows;
    sa.n_embd = fwd.n_embd;
    sa.values = fwd.activations;  // [act_rows.size() * n_embd], position-major (verbatim)
    return sa;
}

// Logit-lens for one pass: top-k token candidates per requested slot, from the host logits. Rows
// [0, count) of fwd are the positions of interest (diffusion: the masked slots, want = masked ++
// committed-if-revising; AR: the single next-token slot), so we read those rows directly. Returns
// nullopt when there are no host logits (the zero-copy device path).
inline std::optional<StepLens> lens_from(const ForwardResult& fwd, const std::vector<int>& want,
                                         int count, int t, int block, int k) {
    if (fwd.logits.empty() || fwd.vocab <= 0 || count <= 0) return std::nullopt;
    const int vocab = fwd.vocab;
    const int kk = k < vocab ? k : vocab;
    StepLens sl;
    sl.t = t; sl.block = block; sl.k = kk;
    std::vector<int> idx(vocab);
    for (int r = 0; r < count && r < fwd.n_requested; ++r) {
        sl.positions.push_back(want[r]);
        const float* row = fwd.row(r);
        float mx = row[0];
        for (int i = 1; i < vocab; ++i) if (row[i] > mx) mx = row[i];
        double sum = 0.0;
        for (int i = 0; i < vocab; ++i) sum += std::exp(static_cast<double>(row[i]) - mx);
        for (int i = 0; i < vocab; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return row[a] > row[b]; });
        for (int j = 0; j < kk; ++j) {
            sl.ids.push_back(idx[j]);
            sl.probs.push_back(static_cast<float>(std::exp(static_cast<double>(row[idx[j]]) - mx) / sum));
        }
    }
    return sl;
}

}  // namespace clozn
