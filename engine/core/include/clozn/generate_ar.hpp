// clozn/generate_ar.hpp — autoregressive generation with the white-box read+steer harness.
//
// Unlike generate()/infill()/denoise() (the backend-free diffusion scheduler validated against the
// lab goldens), this is a thin left-to-right decode loop that lives in the ggml layer: it drives
// GgmlAdapter::ar_forward directly (incremental causal KV decode), so it can't be backend-free.
// It exists because the interpretability layer is model-AGNOSTIC — the activation tap, concept
// probes, logit-lens, and steering all sit on a ForwardResult, not on the denoiser — so the entire
// llama.cpp AR model zoo (Llama/Qwen/Mistral/Gemma/...) gets the same white-box treatment as a dLLM.
//
// It emits the SAME §5.1 events as the diffusion loops — GenStarted, TokensCommitted (one item per
// token), StepFeatures + StepLens per token, GenFinished — so every consumer (CLI, server SSE, viz)
// works unchanged. The honest asymmetry: AR has no token-revision / parallel-pre-commit / infill
// views (those are uniquely diffusion); AR gives the standard per-token read.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "clozn/events.hpp"
#include "clozn/generate.hpp"    // GenerateConfig, SampleConfig, GenerateResult
#include "clozn/model_ggml.hpp"  // GgmlAdapter
#include "clozn/probe.hpp"       // ConceptProbes

namespace clozn {

// Backend-neutral copy of the grammar information emitted by llama.cpp's
// common_chat_templates_apply(). The serving layer adapts common_chat_params to
// this struct; generate_ar.cpp owns construction and advancement of the native
// llama grammar sampler. Diffusion generation deliberately has no such input.
enum class GrammarTriggerType {
    Token,
    Word,
    Pattern,
    PatternFull,
};

struct GrammarTrigger {
    GrammarTriggerType type = GrammarTriggerType::Pattern;
    std::string value;
    int token = -1;
};

struct GrammarConfig {
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<GrammarTrigger> grammar_triggers;
    std::vector<std::string> preserved_tokens;
    std::string generation_prompt;
    // Lazy tool grammars must be suspended while a model is inside its reasoning block; otherwise
    // a tool marker mentioned in hidden reasoning can spuriously activate the output grammar.
    std::string reasoning_start_tag;
    std::string reasoning_end_tag;
    // Template-provided assistant terminators. Generation stops as soon as the decoded output
    // ends with one of these byte strings, and the terminator is omitted from result.text.
    std::vector<std::string> additional_stops;
};

struct BranchResult {
    std::vector<int> generated;
    std::string text;
    std::string reason;  // "eos" | "length"
    int new_tokens = 0;
};

// Greedy by default (SampleConfig: temperature 0). Stops at config.max_new tokens or EOS
// (config.steps / block_len / topk are ignored — AR commits exactly one token per pass).
// `read_probes` (optional) supplies the concept directions for the per-token StepFeatures; calibrate
// them in CAUSAL mode (activations differ from the bidirectional diffusion tap). Steering is applied
// by the caller via adapter.set_steer(...) before this call, exactly as on the diffusion paths.
GenerateResult generate_ar(GgmlAdapter& adapter,
                           const std::vector<int>& prompt_ids,
                           const GenerateConfig& config,
                           const std::function<void(const Event&)>& on_event = {},
                           const SampleConfig& sample = {},
                           const ConceptProbes* read_probes = nullptr,
                           // Optional PyTorch-trained soft prefix: prefix_rows x n_embd raw embeddings spliced
                           // in ahead of the prompt (via ar_forward_embd) before decoding, so a memory learned
                           // on the HF model rides into this ggml generation. nullptr/0 = no prefix (default).
                           const std::vector<float>* prefix_embd = nullptr,
                           int prefix_rows = 0,
                           // Optional early-stop reference (prove-all ablated arms): the baseline reply's
                           // committed token ids. Generation halts at the first generated token that differs
                           // from reference[k] -- yielding a bit-exact PREFIX of the full reply plus a
                           // diverged/diverged_at verdict. Sampling/batching are untouched: this is ONLY a
                           // termination check, so greedy determinism holds (the reply is what full generation
                           // would produce, truncated). nullptr/empty = no reference (full generation, default).
                           const std::vector<int>* reference = nullptr,
                           // Optional native GBNF constraint emitted by the applied chat template. AR-only:
                           // callers must reject this option for diffusion generation rather than silently
                           // running an unconstrained diffusion path.
                           const GrammarConfig* grammar = nullptr,
                           // Optional KV-blob resume (engine-debt: fast restore). Non-null =>
                           // prompt_ids MUST equal resume_from->tokens; the saved KV is loaded via
                           // load_checkpoint and generation continues WITHOUT the full re-prefill.
                           // A restored blob carries no logits row, so resume decodes a one-token
                           // BRIDGE: evict position n_past-1, re-decode the last saved token there
                           // -- the same single-token batch shape the original sequential decode
                           // used, which is what makes bit-exactness achievable (and it is the
                           // acceptance bar: greedy resume suffix == greedy re-prefill suffix).
                           const EngineCheckpoint* resume_from = nullptr,

                           // Execution-fork truncation (engine-debt: exact forks from an EARLIER
                           // point than the checkpoint's own n_past). -1 (default) => resume at
                           // resume_from->n_past, byte-identical to the original resume behavior
                           // above. >= 1 => bridge-decode at THIS position instead (must be in
                           // [1, resume_from->n_past]) -- the live-KV half of the execution-fork
                           // regime split. Requires resume_from; the caller (server_main.cpp) owns
                           // deciding whether live-KV truncation is even valid at this point (it is
                           // NOT, for a point at or before the checkpoint's prompt boundary -- see
                           // EngineCheckpoint::prompt_tokens and the /v1/execution-fork route,
                           // which re-prefills instead in that regime rather than passing this).
                           int resume_truncate_to = -1,
                           // Force the FIRST generated token of THIS call (the execution-fork
                           // "force_token" intervention) instead of sampling it; generation then
                           // continues normally (greedy or sampled, per `sample`) from the second
                           // token on. nullptr (default) = no force, unchanged behavior. Independent
                           // of resume_from: also applies on an ordinary fresh prefill, which is how
                           // the execution-fork reprefill regime forces its first token.
                           const int* force_first_token = nullptr,
                           // Already-measured worker phases for this request (template/tokenization)
                           // plus optional process-startup context. Their clocks are the same worker
                           // steady clock, but offsets may be absent when their local origin differs.
                           const std::vector<PerformancePhase>* request_phases = nullptr);

// Batched multi-sequence branching: prefill a shared prompt once, then decode N independent
// continuations in parallel using a single llama_decode per step. Each branch gets its own
// KV sequence (via branch_kv) and its own RNG (base_sample.seed + branch_index). Greedy
// branches from the same prompt produce identical output (the correctness bar). Returns one
// BranchResult per branch. Cleans up branch sequences before returning.
std::vector<BranchResult> generate_ar_branched(
    GgmlAdapter& adapter,
    const std::vector<int>& prompt_ids,
    int n_branches,
    int max_tokens,
    const SampleConfig& base_sample = {});

}  // namespace clozn
