// clozn/generate.hpp — the shared generation types (DESIGN §5 origin: originally the diffusion pass
// loop's config/result structs). The diffusion generate()/infill()/denoise() functions and their
// scheduler machinery (blocks/cache/policies/stepper/selector) were removed with the diffusion
// program (THE_CUT); GenerateConfig/SampleConfig/GenerateResult survive here because generate_ar.cpp
// (engine/core/src/generate_ar.cpp) -- the autoregressive white-box loop -- reuses them verbatim: one
// request shape and one result shape for both eras kept the server/CLI/viz code that reads them
// unchanged when AR became the only generator. See clozn/generate_ar.hpp for the actual entry point.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "clozn/events.hpp"

namespace clozn {

struct GenerateConfig {
    int max_new;        // tokens/masked-slots to generate after the prompt
    int steps;          // vestigial (diffusion pass budget); AR ignores it
    int block_len = 0;  // vestigial (diffusion semi-AR blocking); AR ignores it
    int topk = -1;       // vestigial (diffusion ConfidenceTopK); AR ignores it
};

// Sampling controls (opt-in; defaults reproduce greedy decoding exactly). temperature 0 = greedy
// argmax; > 0 draws from softmax(logits / T) with a per-generation rng seeded by `seed`. rep_penalty
// > 1 downweights tokens already on the board (CTRL/HF convention) to curb greedy repetition loops.
struct SampleConfig {
    double temperature = 0.0;
    double rep_penalty = 1.0;
    int top_k = 0;         // 0 = off; > 0 keeps the k highest-prob tokens before the sampled draw
    double top_p = 1.0;    // 1.0 = off; (0,1) = nucleus truncation. Both no-op on the greedy path.
    uint64_t seed = 0;
    // Sampler fast-forward for bit-exact SAMPLED resume (engine debt: sampler state in
    // checkpoints): the RNG is advanced this many draws after seeding, so a resumed generation
    // consumes exactly the draws the uninterrupted run would have at that point. One draw is
    // consumed per sampled (non-greedy) committed token; greedy consumes none. 0 = fresh RNG.
    uint64_t rng_discard = 0;
};

struct GenerateResult {
    std::vector<int> board;      // full final board (prompt + generated slots)
    std::vector<int> generated;  // generated ids, truncated at EOS if present
    std::string text;            // decode(generated)
    std::string reason;          // "eos" | "steps_exhausted" | "length"
    int new_tokens = 0;          // == generated.size()
    int steps_total = 0;         // total passes run across all blocks
    std::vector<Event> events;   // the §5.1 event stream for this run (invariant 2; replayable)
    // Early-stop-on-divergence (prove-all ablated arms; AR only). When a reference token list is
    // supplied, generation halts at the first generated token that differs from the reference at that
    // position -- so `text`/`generated` are a BIT-EXACT PREFIX of the full generation (numerics and
    // sampling are unchanged; this is a pure termination condition, never a decode change). The flags
    // let the caller distinguish "changed early, stopped" from "matched all the way (no change)".
    bool ref_active = false;     // a reference was supplied => divergence checking was armed this run
    bool diverged = false;       // true => stopped at diverged_at; false (+ref_active) => matched fully
    int diverged_at = -1;        // generation index of the first divergent token (-1 if none/inactive)
};

}  // namespace clozn
