// cloze/policies.hpp — shared per-candidate types (DESIGN §5.2 origin). The diffusion unmask/commit
// POLICIES (confidence_topk/threshold/remask_lowconf, C++ port of lab/cloze_lab/scheduler/policies.py)
// were removed with the diffusion program (THE_CUT) -- nothing implements them anymore. Candidate and
// StepContext survive here because cloze/sample.hpp (the AR sampler) still returns/consumes them: one
// "a token proposal at a position, with confidence" shape serves both the old scheduler and the
// surviving autoregressive sampler.
//
// Confidences are double (Python float == C double; the old golden fixtures stored float64).
#pragma once

#include <vector>

namespace cloze {

// One position's sampled proposal: token_id at pos, with confidence.
struct Candidate {
    int pos;
    int token_id;
    double confidence;
};

// What a policy may consult — DESIGN §5.2's (step, block_state). steps_total < 0
// means "no fixed budget" (the adaptive stepper supplies none). Vestigial now that the
// diffusion stepper/policies are gone; sample.hpp no longer threads a real StepContext.
struct StepContext {
    int step;
    int steps_total = -1;

    // Steps left including this one; -1 when no fixed budget is set.
    int steps_remaining() const { return steps_total < 0 ? -1 : steps_total - step; }
};

// A policy's verdict for one pass, both vectors pos-ascending. Vestigial (no policy implements
// this anymore); kept only because nothing currently references it enough to warrant a header split.
struct Selection {
    std::vector<Candidate> commit;
    std::vector<Candidate> revise;
};

}  // namespace cloze
