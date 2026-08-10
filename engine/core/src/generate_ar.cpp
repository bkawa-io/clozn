#include "clozn/generate_ar.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "clozn/sample.hpp"
#include "clozn/whitebox.hpp"  // features_from / lens_from (shared with the diffusion loop)

namespace clozn {

namespace {

// Honest softmax readout for a FORCED token (execution-fork's "force_token" intervention): the
// forced commit bypasses sample_committed_candidate entirely (no penalty, no temperature, no
// draw), so its reported "confidence" is the RAW model probability under the untouched
// distribution -- the same stable float64 softmax sample.cpp's greedy path uses, evaluated at one
// token instead of an argmax. Never confabulated (never a bare 1.0); a receipt, not a narration.
double raw_token_probability(const ForwardResult& fwd, int token_id) {
    const int vocab = fwd.vocab;
    const float* row = fwd.row(0);  // ar_forward always returns exactly one row (see model_ggml.cpp)
    double mx = row[0];
    for (int t = 1; t < vocab; ++t) if (row[t] > mx) mx = row[t];
    double denom = 0.0;
    for (int t = 0; t < vocab; ++t) denom += std::exp(static_cast<double>(row[t]) - mx);
    return std::exp(static_cast<double>(row[token_id]) - mx) / denom;
}

std::string regex_escape(const std::string& value) {
    static const std::string metacharacters = R"(.^$|()*+?[]{}\)";
    std::string out;
    out.reserve(value.size() * 2);
    for (char ch : value) {
        if (metacharacters.find(ch) != std::string::npos) out.push_back('\\');
        out.push_back(ch);
    }
    return out;
}

std::string token_piece(const llama_vocab* vocab, llama_token token, bool special) {
    char local[128];
    int n = llama_token_to_piece(vocab, token, local, static_cast<int>(sizeof(local)), 0, special);
    if (n >= 0) return std::string(local, static_cast<size_t>(n));
    std::string out(static_cast<size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, token, out.data(), static_cast<int>(out.size()), 0, special);
    if (n < 0) throw std::runtime_error("failed to decode grammar token");
    out.resize(static_cast<size_t>(n));
    return out;
}

class LlamaGrammarConstraint final : public TokenConstraint {
public:
    LlamaGrammarConstraint(GgmlAdapter& adapter, const GrammarConfig& config)
        : vocab_(adapter.vocab()) {
        if (vocab_ == nullptr) throw std::invalid_argument("grammar requires a model vocabulary");
        if (config.grammar.empty()) throw std::invalid_argument("grammar must not be empty");

        for (const std::string& value : config.preserved_tokens) {
            const std::vector<int> ids = adapter.encode(value);
            if (ids.size() == 1) preserved_tokens_.insert(ids.front());
        }

        std::vector<std::string> patterns;
        std::vector<llama_token> tokens;
        for (const GrammarTrigger& trigger : config.grammar_triggers) {
            switch (trigger.type) {
                case GrammarTriggerType::Token:
                    if (trigger.token < 0 || trigger.token >= llama_vocab_n_tokens(vocab_)) {
                        throw std::invalid_argument("grammar trigger token is outside the model vocabulary");
                    }
                    tokens.push_back(static_cast<llama_token>(trigger.token));
                    break;
                case GrammarTriggerType::Word: {
                    const std::vector<int> ids = adapter.encode(trigger.value);
                    if (ids.size() == 1) {
                        if (preserved_tokens_.find(ids.front()) == preserved_tokens_.end()) {
                            throw std::invalid_argument("single-token grammar trigger must be preserved: " + trigger.value);
                        }
                        tokens.push_back(static_cast<llama_token>(ids.front()));
                    } else {
                        patterns.push_back(regex_escape(trigger.value));
                    }
                    break;
                }
                case GrammarTriggerType::Pattern:
                    patterns.push_back(trigger.value);
                    break;
                case GrammarTriggerType::PatternFull: {
                    std::string anchored = trigger.value.empty() ? "^$" : trigger.value;
                    if (!trigger.value.empty() && trigger.value.front() != '^') anchored.insert(anchored.begin(), '^');
                    if (!trigger.value.empty() && trigger.value.back() != '$') anchored.push_back('$');
                    patterns.push_back(std::move(anchored));
                    break;
                }
            }
        }

        if (config.grammar_lazy && patterns.empty() && tokens.empty()) {
            throw std::invalid_argument("lazy grammar requires at least one trigger");
        }

        if (config.grammar_lazy) {
            std::vector<const char*> pattern_ptrs;
            pattern_ptrs.reserve(patterns.size());
            for (const std::string& pattern : patterns) pattern_ptrs.push_back(pattern.c_str());
            sampler_.reset(llama_sampler_init_grammar_lazy_patterns(
                vocab_, config.grammar.c_str(), "root",
                pattern_ptrs.data(), pattern_ptrs.size(), tokens.data(), tokens.size()));
        } else {
            sampler_.reset(llama_sampler_init_grammar(vocab_, config.grammar.c_str(), "root"));
        }
        if (!sampler_) throw std::invalid_argument("invalid grammar emitted by chat template");

        if (config.grammar_lazy && !config.reasoning_start_tag.empty() &&
            !config.reasoning_end_tag.empty()) {
            reasoning_gate_ = std::make_unique<ReasoningBlockGate>(
                adapter.encode(config.reasoning_start_tag),
                adapter.encode(config.reasoning_end_tag));
        }

        // llama.cpp's common sampler advances output-format/tool-call grammars over
        // the assistant prefix already present in the prompt. Chat-template grammar
        // configs are exactly that category, so reproduce the prefill here.
        if (!config.grammar_lazy && !config.generation_prompt.empty()) {
            const std::vector<int> prefill = adapter.encode(config.generation_prompt);
            for (size_t i = 0; i < prefill.size(); ++i) {
                const std::string piece = token_piece(vocab_, static_cast<llama_token>(prefill[i]), true);
                if (i == 0 && !piece.empty() && !config.generation_prompt.empty() &&
                    std::isspace(static_cast<unsigned char>(piece.front())) &&
                    !std::isspace(static_cast<unsigned char>(config.generation_prompt.front()))) {
                    continue;
                }
                llama_sampler_accept(sampler_.get(), static_cast<llama_token>(prefill[i]));
            }
        }

        // Lazy grammars use the generation prefix only to establish whether generation begins
        // inside a reasoning block. The grammar itself remains unadvanced until its trigger fires.
        if (reasoning_gate_ && !config.generation_prompt.empty()) {
            for (int token : adapter.encode(config.generation_prompt)) reasoning_gate_->accept(token);
        }
    }

    void apply(std::vector<float>& logits) override {
        grammar_applied_ = !reasoning_gate_ || !reasoning_gate_->active();
        if (!grammar_applied_) return;
        candidates_.resize(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            candidates_[i] = llama_token_data{
                static_cast<llama_token>(i), logits[i], 0.0f};
        }
        llama_token_data_array array{
            candidates_.data(), candidates_.size(), -1, false};
        llama_sampler_apply(sampler_.get(), &array);
        for (const llama_token_data& candidate : candidates_) {
            if (candidate.id < 0 || static_cast<size_t>(candidate.id) >= logits.size()) {
                throw std::runtime_error("grammar sampler returned an invalid token id");
            }
            logits[static_cast<size_t>(candidate.id)] = candidate.logit;
        }
    }

    void accept(int token) override {
        const bool accept_grammar = grammar_applied_;
        if (reasoning_gate_) reasoning_gate_->accept(token);
        if (accept_grammar) {
            llama_sampler_accept(sampler_.get(), static_cast<llama_token>(token));
        }
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string out;
        for (int token : tokens) {
            out += token_piece(vocab_, static_cast<llama_token>(token),
                               preserved_tokens_.find(token) != preserved_tokens_.end());
        }
        return out;
    }

private:
    struct SamplerDeleter {
        void operator()(llama_sampler* sampler) const { llama_sampler_free(sampler); }
    };

    const llama_vocab* vocab_ = nullptr;
    std::set<int> preserved_tokens_;
    std::unique_ptr<llama_sampler, SamplerDeleter> sampler_;
    std::vector<llama_token_data> candidates_;
    std::unique_ptr<ReasoningBlockGate> reasoning_gate_;
    bool grammar_applied_ = true;
};

}  // namespace

GenerateResult generate_ar(GgmlAdapter& adapter,
                           const std::vector<int>& prompt_ids,
                           const GenerateConfig& config,
                           const std::function<void(const Event&)>& on_event,
                           const SampleConfig& sample,
                           const ConceptProbes* read_probes,
                           const std::vector<float>* prefix_embd,
                           int prefix_rows,
                           const std::vector<int>* reference,
                           const GrammarConfig* grammar,
                           const EngineCheckpoint* resume_from,
                           int resume_truncate_to,
                           const int* force_first_token,
                           const std::vector<PerformancePhase>* request_phases) {
    if (prompt_ids.empty()) throw std::invalid_argument("prompt_ids must be non-empty");
    if (config.max_new < 1) throw std::invalid_argument("max_new must be >= 1");
    if (resume_from != nullptr) {
        if (resume_from->tokens != prompt_ids)
            throw std::invalid_argument("resume_from: prompt_ids must equal the checkpoint's own "
                                        "token sequence (resume continues THAT state, never a "
                                        "different prompt)");
        if (resume_from->n_past < 1 ||
            resume_from->n_past > static_cast<int>(resume_from->tokens.size()))
            throw std::invalid_argument("resume_from: n_past out of range for the saved tokens");
        if (resume_truncate_to >= 0 &&
            (resume_truncate_to < 1 || resume_truncate_to > resume_from->n_past))
            throw std::invalid_argument("resume_truncate_to out of range [1, resume_from->n_past]");
        if (prefix_embd != nullptr && prefix_rows > 0)
            throw std::invalid_argument("resume_from cannot be combined with a soft prefix (the "
                                        "prefix is already inside the saved KV, if it was there "
                                        "at checkpoint time)");
    } else if (resume_truncate_to >= 0) {
        throw std::invalid_argument("resume_truncate_to requires resume_from");
    }
    if (force_first_token != nullptr && *force_first_token < 0)
        throw std::invalid_argument("force_first_token must be a valid (non-negative) token id");

    // Early-stop-on-divergence is armed only with a non-empty reference. This is a pure termination
    // condition -- nothing about sampling, batching, or the KV path changes -- so the generated prefix
    // is bit-identical to what a full (reference-less) generation would produce up to the stop point.
    const bool ref_active = (reference != nullptr && !reference->empty());
    const int ref_len = ref_active ? static_cast<int>(reference->size()) : 0;
    bool diverged = false;
    int diverged_at = -1;

    const ModelConfig& mcfg = adapter.config();
    const int eos = mcfg.eos_token_id;

    using clock = std::chrono::steady_clock;
    std::int64_t event_delivery_ns = 0;
    std::vector<Event> events;
    auto emit = [&](Event e) {
        if (on_event) {
            const clock::time_point delivery_start = clock::now();
            on_event(e);
            event_delivery_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                clock::now() - delivery_start).count();
        }
        events.push_back(std::move(e));
    };
    auto ms_since = [](clock::time_point a) {
        return std::chrono::duration<double, std::milli>(clock::now() - a).count();
    };
    const clock::time_point t_start = clock::now();

    // Effective prompt length for THIS run: normally the full prompt_ids, but a resume with an
    // explicit truncation point (execution-fork) uses only prompt_ids[0, resume_truncate_to) -- the
    // tail the checkpoint carried PAST that point must never enter the repetition-penalty board or
    // the reported board (it's being regenerated, not replayed). resume_truncate_to == -1 (every
    // caller before execution-fork) reduces to resume_from->n_past, which already equals
    // prompt_ids.size() at both existing checkpoint-creation sites -- byte-identical to the prior
    // `p = prompt_ids.size()` for every existing call site.
    const int p = (resume_from != nullptr)
                      ? ((resume_truncate_to >= 0) ? resume_truncate_to : resume_from->n_past)
                      : static_cast<int>(prompt_ids.size());
    emit(GenStarted{0, p, 0, config.max_new});  // block_len 0: AR has no blocks

    // Causal attention + a fresh KV cache -- EXCEPT on a KV-blob resume, where load_checkpoint
    // itself sets the mode and installs the saved cache (set_causal here would clear the very
    // state being restored). A steering control vector set by the caller persists across decodes
    // (it's on the context, independent of the attention mode).
    if (resume_from == nullptr) adapter.set_causal(true);

    // Optional: splice a PyTorch-trained soft prefix in ahead of the prompt (fills KV [0, prefix_rows))
    // so a memory learned on the HF model shapes this ggml generation. The prompt then decodes at n_past=base.
    int base = 0;
    if (prefix_embd != nullptr && prefix_rows > 0) {
        adapter.ar_forward_embd(*prefix_embd, prefix_rows, 0);
        base = prefix_rows;
    }

    std::mt19937_64 rng(sample.seed);
    if (sample.rng_discard > 0) rng.discard(sample.rng_discard);  // sampled-resume fast-forward
    // Running sequence for the repetition penalty: prompt_ids truncated to `p` (identical to the
    // full prompt_ids whenever p == prompt_ids.size(), i.e. every call site before execution-fork).
    std::vector<int> seq(prompt_ids.begin(), prompt_ids.begin() + p);
    SampleOpts sopts;
    sopts.temperature = sample.temperature;
    sopts.rep_penalty = sample.rep_penalty;
    sopts.top_k = sample.top_k;
    sopts.top_p = sample.top_p;
    sopts.mask_token = mcfg.mask_token_id;  // -1 on an AR model; harmless
    if (sample.rep_penalty != 1.0) sopts.board = &seq;
    if (sample.temperature > 0.0) sopts.rng = &rng;

    std::unique_ptr<LlamaGrammarConstraint> grammar_constraint;
    if (grammar != nullptr && !grammar->grammar.empty()) {
        grammar_constraint = std::make_unique<LlamaGrammarConstraint>(adapter, *grammar);
        sopts.constraint = grammar_constraint.get();
    }

    std::vector<int> generated;
    generated.reserve(config.max_new);
    struct PendingCommit { int t; CommitItem item; };
    std::vector<PendingCommit> pending_stops;
    auto flush_pending_stops = [&]() {
        for (const PendingCommit& pending : pending_stops)
            emit(TokensCommitted{pending.t, 0, {pending.item}});
        pending_stops.clear();
    };
    std::string reason = "length";
    std::string stopped_text;

    // Hard context window: absolute positions [0, n_ctx) fit the KV cache. The prompt (+ any soft prefix)
    // must fit it; a prompt that exceeds it is a client error surfaced cleanly (the handler turns it into a
    // 400), never a silent truncation and never an uncaught 500.
    const int n_ctx = adapter.n_ctx();
    if (base + p > n_ctx)
        throw std::invalid_argument("prompt exceeds context window (n_ctx): reduce the prompt or raise --ctx");

    // Prefill: the last prompt row's logits are the distribution for the first generated token.
    // KV-blob resume replaces the full prefill with load_checkpoint + a ONE-token bridge decode:
    // the blob restores KV [0, n_past) but no logits row, so we evict the last saved position and
    // re-decode its token there -- the identical single-token batch shape the original decode
    // used at that position, so the resulting row is the same distribution the interrupted
    // generation would have sampled from (the bit-exactness acceptance the route verifies).
    const clock::time_point prefill_start = clock::now();
    ForwardResult fwd;
    int n_past;
    if (resume_from != nullptr) {
        adapter.load_checkpoint(*resume_from);
        const int np = p;  // == resume_truncate_to when set, else resume_from->n_past (unchanged)
        adapter.evict_from(np - 1);
        fwd = adapter.ar_forward({resume_from->tokens[static_cast<size_t>(np - 1)]}, np - 1);
        n_past = np;
    } else {
        fwd = adapter.ar_forward(prompt_ids, base);
        n_past = base + p;
    }
    const clock::time_point prefill_end = clock::now();
    const std::int64_t prefill_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(prefill_end - prefill_start).count();
    const clock::time_point decode_start = prefill_end;
    event_delivery_ns = 0;  // GenStarted delivery preceded decode and must not be charged to it.
    int t = 0;

    for (int k = 0; k < config.max_new; ++k) {
        // Stop GRACEFULLY at the context limit: a token committed at position n_past must be decoded back
        // into the KV (ar_forward at n_past), which needs n_past < n_ctx. When there's no room left, this is
        // a "length" stop (finish_reason "length", a clean 200) -- NOT an ar_forward "exceeds n_ctx" throw,
        // which the non-streaming handler would surface as an empty 500. THE root fix for the long-generation
        // 500: a generation that reaches the context window stops, it never overflows.
        if (n_past >= n_ctx) { reason = "length"; break; }
        const std::vector<int> want = {n_past};  // the position about to be generated

        // Execution-fork "force_token": override ONLY the first slot of this call with a caller-
        // supplied token instead of sampling it -- no draw consumed (the rng stays exactly where
        // sample.rng_discard left it, ready for k==1's real sample), no penalty/temperature/grammar
        // applied. Reported confidence is still an honest measurement (raw_token_probability), not
        // a confabulated 1.0.
        int tok;
        double conf;
        if (k == 0 && force_first_token != nullptr) {
            tok = *force_first_token;
            if (tok >= adapter.config().vocab_size)
                throw std::invalid_argument("force_first_token: token id outside the model vocabulary");
            conf = raw_token_probability(fwd, tok);
            if (sopts.constraint != nullptr) sopts.constraint->accept(tok);  // keep grammar state honest
        } else {
            const Candidate cand = sample_committed_candidate(fwd, n_past, sopts);
            tok = cand.token_id;
            conf = cand.confidence;
        }

        generated.push_back(tok);
        seq.push_back(tok);
        const bool is_eos = (eos >= 0 && tok == eos);

        // llama.cpp chat templates and the public OpenAI request may provide extra assistant
        // terminators in addition to the model EOS token. Match them against the same
        // special-token-aware byte stream returned to the parser. The matched terminator is not
        // emitted, fed back into the KV, or exposed in result.text.
        bool hit_additional_stop = false;
        if (grammar != nullptr && !grammar->additional_stops.empty()) {
            const std::string decoded = grammar_constraint
                                            ? grammar_constraint->decode(generated)
                                            : adapter.decode(generated);
            for (const std::string& stop : grammar->additional_stops) {
                if (!stop.empty() && decoded.size() >= stop.size() &&
                    decoded.compare(decoded.size() - stop.size(), stop.size(), stop) == 0) {
                    stopped_text = decoded.substr(0, decoded.size() - stop.size());
                    hit_additional_stop = true;
                    break;
                }
            }
        }

        bool may_be_stop_prefix = false;
        if (!hit_additional_stop && grammar != nullptr && !grammar->additional_stops.empty()) {
            const std::string decoded = grammar_constraint
                                            ? grammar_constraint->decode(generated)
                                            : adapter.decode(generated);
            for (const std::string& stop : grammar->additional_stops) {
                if (stop.empty()) continue;
                const size_t max_suffix = std::min(decoded.size(), stop.size() - 1);
                for (size_t n = max_suffix; n > 0; --n) {
                    if (decoded.compare(decoded.size() - n, n, stop, 0, n) == 0) {
                        may_be_stop_prefix = true;
                        break;
                    }
                }
                if (may_be_stop_prefix) break;
            }
        }
        const CommitItem committed{n_past, tok, conf};
        if (may_be_stop_prefix) pending_stops.push_back(PendingCommit{t, committed});
        else {
            flush_pending_stops();
            emit(TokensCommitted{t, 0, {committed}});
        }
        if (auto sl = lens_from(fwd, want, 1, t, 0, 5)) emit(*sl);

        // Early-stop on divergence from the reference (prove-all ablated arms): the just-committed token
        // is already in `generated`, so the partial reply is a bit-exact PREFIX of the full reply. We stop
        // BEFORE feeding this token back in -- the remaining ~max_new-k decodes are exactly the work we're
        // here to save. `tok != reference[k]` (or running past the reference's length) is the divergence.
        // `is_eos` is folded in: an early EOS that the reference didn't have is itself a divergence at k.
        if (ref_active && (k >= ref_len || tok != (*reference)[k])) {
            diverged = true;
            diverged_at = k;
            ++t;  // count this committed step (matches the non-diverged ++t below)
            break;
        }

        if (hit_additional_stop) {
            reason = "stop";
            ++t;
            break;
        }

        // Feed the token back in: this decode (a) advances the KV and (b) yields the hidden state at
        // the token we just generated — its concept read. So StepFeatures is labeled with the
        // GENERATED token's own position (act_rows = {n_past}), the intuitive "this token is X".
        fwd = adapter.ar_forward({tok}, n_past);
        if (auto sf = features_from(fwd, t, 0, read_probes)) emit(*sf);
        if (auto sa = activations_from(fwd, t, 0)) emit(*sa);  // raw state (heavy; on-demand)

        ++n_past;
        ++t;
        if (is_eos) { reason = "eos"; break; }
    }

    if (reason != "stop") flush_pending_stops();

    const clock::time_point decode_end = clock::now();
    const std::int64_t decode_loop_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end - decode_start).count();
    const std::int64_t decode_ns = std::max<std::int64_t>(0, decode_loop_ns - event_delivery_ns);

    GenerateResult result;
    std::vector<int> kept = generated;
    if (reason == "stop") {
        for (size_t n = 0; n <= generated.size(); ++n) {
            std::vector<int> prefix(generated.begin(), generated.begin() + static_cast<std::ptrdiff_t>(n));
            if (adapter.decode(prefix) == stopped_text) {
                kept = std::move(prefix);
                break;
            }
        }
    }
    if (reason == "eos" && !kept.empty()) kept.pop_back();  // drop the trailing EOS for the text
    result.new_tokens = static_cast<int>(kept.size());
    result.generated = kept;
    result.text = reason == "stop"
                      ? stopped_text
                      : (grammar_constraint ? grammar_constraint->decode(kept) : adapter.decode(kept));
    result.steps_total = t;
    result.reason = reason;
    result.ref_active = ref_active;    // divergence checking was armed (a reference was supplied)
    result.diverged = diverged;        // true => stopped early at diverged_at; false + ref_active => matched fully
    result.diverged_at = diverged_at;  // generation index of the first divergent token (-1 if none)
    // Truncated to `p` for the same reason `seq` above is: a resumed execution-fork must never
    // report the checkpoint's PAST-truncate_to tail as part of this run's board (that tail is being
    // regenerated, not replayed). Identical to `prompt_ids` whenever p == prompt_ids.size().
    result.board.assign(prompt_ids.begin(), prompt_ids.begin() + p);
    result.board.insert(result.board.end(), kept.begin(), kept.end());

    const double wall_ms = ms_since(t_start);
    const double decode_s = static_cast<double>(decode_ns) / 1e9;
    const double prefill_s = static_cast<double>(prefill_ns) / 1e9;
    const double tok_per_s = decode_s > 0 ? result.new_tokens / decode_s : 0.0;
    const double prompt_tok_per_s =
        (resume_from == nullptr && prefill_s > 0) ? p / prefill_s : -1.0;
    std::vector<PerformancePhase> timing_phases =
        request_phases != nullptr ? *request_phases : std::vector<PerformancePhase>{};
    PerformancePhase prefill_phase{"prefill", "clozn_worker", prefill_ns,
        std::chrono::duration_cast<std::chrono::nanoseconds>(prefill_start - t_start).count()};
    if (resume_from != nullptr) prefill_phase.includes = {"kv_restore", "bridge_decode"};
    if (prefix_embd != nullptr && prefix_rows > 0) prefill_phase.includes.push_back("soft_prefix");
    timing_phases.push_back(std::move(prefill_phase));
    timing_phases.push_back(PerformancePhase{
        "decode", "clozn_worker", decode_ns,
        std::chrono::duration_cast<std::chrono::nanoseconds>(decode_start - t_start).count()});
    if (event_delivery_ns > 0) {
        timing_phases.push_back(PerformancePhase{
            "stream_flush", "clozn_worker", event_delivery_ns, -1, "exclusive"});
    }
    emit(GenFinished{t > 0 ? t - 1 : 0, result.reason, result.new_tokens, wall_ms, t, tok_per_s,
                     std::move(timing_phases), p, prompt_tok_per_s, tok_per_s});
    result.events = std::move(events);
    return result;
}

GenerateResult generate_ar_appended(GgmlAdapter& adapter,
                                    const EngineCheckpoint& checkpoint,
                                    const std::vector<int>& append_token_ids,
                                    const GenerateConfig& config,
                                    const SampleConfig& sample,
                                    const std::function<void()>& check_cancelled) {
    if (checkpoint.n_past < 1 || checkpoint.n_past > static_cast<int>(checkpoint.tokens.size()))
        throw std::invalid_argument("append continuation: checkpoint n_past is outside its token history");
    if (!checkpoint.causal)
        throw std::invalid_argument("append continuation requires a causal checkpoint");
    if (append_token_ids.empty())
        throw std::invalid_argument("append continuation requires non-empty append_token_ids");
    if (config.max_new < 1)
        throw std::invalid_argument("append continuation requires max_new >= 1");
    if (check_cancelled) check_cancelled();

    const ModelConfig& mcfg = adapter.config();
    for (const int token_id : append_token_ids) {
        if (token_id < 0 || token_id >= mcfg.vocab_size)
            throw std::invalid_argument("append continuation token id is outside the model vocabulary");
    }

    // `load_checkpoint` restores the historical KV byte-for-byte.  Do not bridge-decode or evict its
    // final historical position: the first call below starts at exactly checkpoint.n_past and consumes
    // a gateway-validated NEW append token.  This is the operation's defining no-reprefill invariant.
    adapter.load_checkpoint(checkpoint);
    int n_past = checkpoint.n_past;
    const int n_ctx = adapter.n_ctx();
    std::vector<int> board(checkpoint.tokens.begin(), checkpoint.tokens.begin() + checkpoint.n_past);
    board.reserve(board.size() + append_token_ids.size() + static_cast<size_t>(config.max_new));

    ForwardResult fwd;
    for (const int token_id : append_token_ids) {
        if (check_cancelled) check_cancelled();
        if (n_past >= n_ctx)
            throw std::invalid_argument(
                "append continuation exceeds context window before all append tokens were decoded");
        // Exactly one supplied token at exactly one position: no text tokenization, template rendering,
        // prefix prefill, or batch-shaped append is possible through this primitive.
        fwd = adapter.ar_forward({token_id}, n_past);
        board.push_back(token_id);
        ++n_past;
    }

    std::mt19937_64 rng(sample.seed);
    if (sample.rng_discard > 0) rng.discard(sample.rng_discard);
    SampleOpts sopts;
    sopts.temperature = sample.temperature;
    sopts.rep_penalty = sample.rep_penalty;
    sopts.top_k = sample.top_k;
    sopts.top_p = sample.top_p;
    sopts.mask_token = mcfg.mask_token_id;
    if (sample.rep_penalty != 1.0) sopts.board = &board;
    if (sample.temperature > 0.0) sopts.rng = &rng;

    std::vector<int> generated;
    generated.reserve(config.max_new);
    std::string reason = "length";
    int steps_total = 0;
    const int eos = mcfg.eos_token_id;
    for (int k = 0; k < config.max_new; ++k) {
        if (check_cancelled) check_cancelled();
        if (n_past >= n_ctx) { reason = "length"; break; }
        const Candidate candidate = sample_committed_candidate(fwd, n_past, sopts);
        const int token_id = candidate.token_id;
        generated.push_back(token_id);
        board.push_back(token_id);
        ++steps_total;
        // One generated token per forward is the existing AR decode regime.  It is deliberately
        // distinct from the supplied append loop only in who chose the token, never in batch shape.
        // This includes EOS: as in generate_ar, a committed EOS belongs in the live KV and advancing
        // n_past before stopping keeps `board`, saved checkpoints, and restored state consistent.
        fwd = adapter.ar_forward({token_id}, n_past);
        ++n_past;
        if (eos >= 0 && token_id == eos) { reason = "eos"; break; }
    }

    GenerateResult result;
    std::vector<int> visible = generated;
    if (reason == "eos" && !visible.empty()) visible.pop_back();
    result.board = std::move(board);  // includes EOS when one was committed, like generate_ar.
    result.generated = std::move(visible);
    result.text = adapter.decode(result.generated);
    result.reason = reason;
    result.new_tokens = static_cast<int>(result.generated.size());
    result.steps_total = steps_total;
    return result;
}

std::vector<BranchResult> generate_ar_branched(
    GgmlAdapter& adapter,
    const std::vector<int>& prompt_ids,
    int n_branches,
    int max_tokens,
    const SampleConfig& base_sample) {

    if (prompt_ids.empty()) throw std::invalid_argument("prompt_ids must be non-empty");
    if (n_branches < 1) throw std::invalid_argument("n_branches must be >= 1");
    if (max_tokens < 1) throw std::invalid_argument("max_tokens must be >= 1");

    const ModelConfig& mcfg = adapter.config();
    const int eos = mcfg.eos_token_id;
    const int n_ctx = adapter.n_ctx();
    const int p = static_cast<int>(prompt_ids.size());

    adapter.set_causal(true);

    if (p > n_ctx)
        throw std::invalid_argument("prompt exceeds context window");

    ForwardResult shared_fwd = adapter.ar_forward(prompt_ids, 0);
    int n_past = p;

    if (n_past >= n_ctx) {
        std::vector<BranchResult> results(n_branches);
        for (auto& r : results) r.reason = "length";
        return results;
    }

    adapter.branch_kv(n_branches);

    std::vector<std::vector<int>> seqs(n_branches, prompt_ids);
    std::vector<std::vector<int>> gen(n_branches);
    std::vector<std::string> reasons(n_branches, "length");
    std::vector<bool> active(n_branches, true);
    std::vector<std::mt19937_64> rngs;
    for (int i = 0; i < n_branches; ++i)
        rngs.emplace_back(base_sample.seed + static_cast<uint64_t>(i));

    std::vector<ForwardResult> batch_fwd;

    for (int k = 0; k < max_tokens; ++k) {
        if (n_past >= n_ctx) break;

        bool any_active = false;
        for (int i = 0; i < n_branches; ++i) if (active[i]) any_active = true;
        if (!any_active) break;

        std::vector<int> next_tokens(n_branches, 0);
        for (int i = 0; i < n_branches; ++i) {
            if (!active[i]) continue;
            SampleOpts sopts;
            sopts.temperature = base_sample.temperature;
            sopts.rep_penalty = base_sample.rep_penalty;
            sopts.top_k = base_sample.top_k;
            sopts.top_p = base_sample.top_p;
            sopts.mask_token = mcfg.mask_token_id;
            if (base_sample.rep_penalty != 1.0) sopts.board = &seqs[i];
            if (base_sample.temperature > 0.0) sopts.rng = &rngs[i];

            const ForwardResult& cur = (k == 0) ? shared_fwd : batch_fwd[i];
            auto cand = sample_candidates(cur, {n_past}, sopts);
            if (cand.empty()) { active[i] = false; continue; }
            int tok = cand[0].token_id;
            next_tokens[i] = tok;
            gen[i].push_back(tok);
            seqs[i].push_back(tok);
            if (eos >= 0 && tok == eos) {
                reasons[i] = "eos";
                active[i] = false;
            }
        }

        any_active = false;
        for (int i = 0; i < n_branches; ++i) if (active[i]) any_active = true;
        if (!any_active || k + 1 >= max_tokens) break;

        batch_fwd = adapter.ar_forward_batch(next_tokens, n_past, active);
        ++n_past;
    }

    adapter.cleanup_seqs(n_branches);

    std::vector<BranchResult> results(n_branches);
    for (int i = 0; i < n_branches; ++i) {
        auto& r = results[i];
        std::vector<int> kept = gen[i];
        if (reasons[i] == "eos" && !kept.empty()) kept.pop_back();
        r.generated = gen[i];
        r.text = adapter.decode(kept);
        r.reason = reasons[i];
        r.new_tokens = static_cast<int>(kept.size());
    }
    return results;
}

}  // namespace clozn
