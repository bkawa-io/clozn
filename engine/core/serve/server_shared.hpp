// Shared helpers and context pooling for the private v1 worker.
#pragma once

#include "nlohmann/json.hpp"

#include "clozn/events.hpp"
#include "clozn/generate_ar.hpp"
#include "clozn/model_ggml.hpp"
#include "clozn/prompt_prefix_cache.hpp"

#include "ggml.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clozn {

using json = nlohmann::json;

// Pinned identically in clozn/protocol.py and protocol/fixtures/handshake.json.
inline constexpr const char* PROTOCOL_VERSION = "1.1";

inline std::atomic<std::uint64_t> g_request_counter{0};

inline std::string make_id(const char* prefix) {
    return std::string(prefix) + std::to_string(g_request_counter.fetch_add(1));
}

inline const char* finish_reason(const std::string& reason) {
    return (reason == "eos" || reason == "stop") ? "stop" : "length";
}

inline void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

inline SampleConfig sample_from(const json& body) {
    SampleConfig sample;
    sample.temperature = body.value("temperature", 0.0);
    sample.rep_penalty = body.value("rep_penalty", 1.0);
    sample.top_k = body.value("top_k", 0);
    sample.top_p = body.value("top_p", 1.0);
    sample.seed = body.value("seed", static_cast<std::uint64_t>(0));
    return sample;
}

// Token pieces may split a UTF-8 code point. Replacement serialization keeps one malformed piece
// from aborting a stream before its final frame arrives.
inline std::string dump_json(const json& value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

struct GenerationCancelled : std::runtime_error {
    GenerationCancelled() : std::runtime_error("cancelled") {}
};

class CancelRegistry {
public:
    std::shared_ptr<std::atomic<bool>> register_request(const std::string& request_id) {
        auto flag = std::make_shared<std::atomic<bool>>(false);
        std::lock_guard<std::mutex> lock(mutex_);
        flags_[request_id] = flag;
        return flag;
    }

    void unregister_request(const std::string& request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        flags_.erase(request_id);
    }

    bool cancel(const std::string& request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = flags_.find(request_id);
        if (found == flags_.end()) return false;
        found->second->store(true, std::memory_order_relaxed);
        return true;
    }

    struct Guard {
        CancelRegistry& registry;
        std::string request_id;
        ~Guard() { registry.unregister_request(request_id); }
    };

private:
    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<std::atomic<bool>>> flags_;
};

struct StreamEnvelope {
    std::string request_id;
    std::function<bool(const std::string&)> write;
    std::uint64_t sequence = 0;
    std::shared_ptr<std::atomic<bool>> cancel;

    void write_raw(const std::string& value) {
        if (!write(value) && cancel) cancel->store(true, std::memory_order_relaxed);
    }

    void frame(json value) {
        value["req"] = request_id;
        value["seq"] = sequence++;
        write_raw("data: " + dump_json(value) + "\n\n");
    }

    void frame_str(const std::string& serialized) {
        json value = json::parse(serialized, nullptr, /*allow_exceptions=*/false);
        if (value.is_discarded() || !value.is_object()) {
            write_raw("data: " + serialized + "\n\n");
            return;
        }
        frame(std::move(value));
    }

    void done() { write_raw("data: [DONE]\n\n"); }

    void check_cancelled() const {
        if (cancel && cancel->load(std::memory_order_relaxed)) throw GenerationCancelled();
    }
};

inline std::string sse_data(const Event& event, const GgmlModel& model,
                            const std::vector<int>& prompt_ids,
                            const std::vector<int>& suffix_ids) {
    if (const auto* started = std::get_if<GenStarted>(&event)) {
        json prompt_pieces = json::array();
        for (int id : prompt_ids) prompt_pieces.push_back(model.decode({id}));
        json suffix_pieces = json::array();
        for (int id : suffix_ids) suffix_pieces.push_back(model.decode({id}));
        return dump_json({
            {"t", started->t},
            {"type", "gen_started"},
            {"prompt_tokens", started->prompt_tokens},
            {"block_len", started->block_len},
            {"max_new", started->max_new},
            {"prompt_pieces", prompt_pieces},
            {"suffix_pieces", suffix_pieces},
        });
    }
    if (const auto* committed = std::get_if<TokensCommitted>(&event)) {
        json items = json::array();
        for (const auto& item : committed->items) {
            items.push_back({
                {"pos", item.pos},
                {"id", item.id},
                {"conf", item.conf},
                {"piece", model.decode({item.id})},
            });
        }
        return dump_json({
            {"t", committed->t},
            {"type", "tokens_committed"},
            {"block", committed->block},
            {"items", items},
        });
    }
    if (const auto* lens = std::get_if<StepLens>(&event)) {
        json pieces = json::array();
        for (int id : lens->ids) pieces.push_back(model.decode({id}));
        return dump_json({
            {"t", lens->t},
            {"type", "step_lens"},
            {"block", lens->block},
            {"k", lens->k},
            {"positions", lens->positions},
            {"ids", lens->ids},
            {"probs", lens->probs},
            {"pieces", pieces},
        });
    }
    return to_jsonl_line(event);
}

// A bounded pool of contexts over one shared model. Persistent exact-reference sessions use one
// additional dedicated context so ordinary scalar confirmation remains available.
class ContextPool {
    struct Slot {
        std::unique_ptr<GgmlAdapter> adapter;
        std::vector<int> resident_generation_tokens;
        std::vector<int> resident_score_tokens;
        bool generation_cache_valid = false;
        bool score_cache_valid = false;
    };

public:
    ContextPool(std::shared_ptr<GgmlModel> model, int workers, int n_ctx,
                bool flash_attention = true, int n_batch = 0, int n_ubatch = 0)
        : model_(std::move(model)), flash_attention_(flash_attention) {
        for (int i = 0; i < workers; ++i) {
            auto slot = std::make_unique<Slot>();
            slot->adapter = std::make_unique<GgmlAdapter>(
                model_, n_ctx, flash_attention_, n_batch, n_ubatch);
            free_.push_back(slot.get());
            slots_.push_back(std::move(slot));
        }
    }

    class Lease {
    public:
        Lease(ContextPool& pool, Slot* slot, int cached_prompt_prefix = 0,
              bool generation = false, bool score = false)
            : pool_(pool), slot_(slot), cached_prompt_prefix_(cached_prompt_prefix),
              generation_(generation), score_(score) {}
        ~Lease() {
            if (slot_ != nullptr) {
                if (generation_ && !generation_committed_) {
                    slot_->resident_generation_tokens.clear();
                    slot_->generation_cache_valid = false;
                }
                if (score_ && !score_committed_) {
                    slot_->resident_score_tokens.clear();
                    slot_->score_cache_valid = false;
                }
                pool_.release(slot_);
            }
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
            : pool_(other.pool_), slot_(other.slot_),
              cached_prompt_prefix_(other.cached_prompt_prefix_),
              generation_(other.generation_),
              generation_committed_(other.generation_committed_),
              score_(other.score_), score_committed_(other.score_committed_) {
            other.slot_ = nullptr;
        }
        Lease& operator=(Lease&&) = delete;
        GgmlAdapter& operator*() const { return *slot_->adapter; }
        int cached_prompt_prefix() const { return cached_prompt_prefix_; }
        int cached_score_prefix() const { return cached_prompt_prefix_; }

        void commit_generation_cache(std::vector<int> resident_tokens) {
            if (!generation_)
                throw std::logic_error("generation cache commit requires a generation lease");
            if (resident_tokens.empty() ||
                slot_->adapter->ar_seq_size(0) != static_cast<int>(resident_tokens.size())) {
                slot_->resident_generation_tokens.clear();
                slot_->generation_cache_valid = false;
            } else {
                slot_->resident_generation_tokens = std::move(resident_tokens);
                slot_->generation_cache_valid = true;
            }
            generation_committed_ = true;
        }

        void commit_score_cache(std::vector<int> resident_tokens) {
            if (!score_)
                throw std::logic_error("score cache commit requires a score lease");
            if (resident_tokens.empty() ||
                slot_->adapter->ar_seq_size(0) != static_cast<int>(resident_tokens.size())) {
                slot_->resident_score_tokens.clear();
                slot_->score_cache_valid = false;
            } else {
                slot_->resident_score_tokens = std::move(resident_tokens);
                slot_->score_cache_valid = true;
            }
            score_committed_ = true;
        }

    private:
        ContextPool& pool_;
        Slot* slot_;
        int cached_prompt_prefix_ = 0;
        bool generation_ = false;
        bool generation_committed_ = false;
        bool score_ = false;
        bool score_committed_ = false;
    };

    class PersistentLease {
    public:
        PersistentLease(ContextPool& pool, std::unique_ptr<GgmlAdapter> adapter)
            : pool_(pool), adapter_(std::move(adapter)) {}
        ~PersistentLease() {
            if (adapter_ == nullptr) return;
            adapter_.reset();
            pool_.release_persistent();
        }
        PersistentLease(const PersistentLease&) = delete;
        PersistentLease& operator=(const PersistentLease&) = delete;
        PersistentLease(PersistentLease&& other) noexcept
            : pool_(other.pool_), adapter_(std::move(other.adapter_)) {}
        PersistentLease& operator=(PersistentLease&&) = delete;
        GgmlAdapter& operator*() const { return *adapter_; }

    private:
        ContextPool& pool_;
        std::unique_ptr<GgmlAdapter> adapter_;
    };

    Lease acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [&] { return !free_.empty(); });
        Slot* slot = free_.back();
        free_.pop_back();
        slot->resident_generation_tokens.clear();
        slot->resident_score_tokens.clear();
        slot->generation_cache_valid = false;
        slot->score_cache_valid = false;
        return Lease(*this, slot);
    }

    Lease acquire_generation(const std::vector<int>& prompt_tokens) {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [&] { return !free_.empty(); });
        auto best = free_.begin();
        int best_prefix = -1;
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            Slot* candidate = *it;
            int prefix = 0;
            if (candidate->generation_cache_valid && candidate->adapter->causal() &&
                candidate->adapter->ar_seq_size(0) ==
                    static_cast<int>(candidate->resident_generation_tokens.size())) {
                prefix = exact_token_prefix_length(
                    candidate->resident_generation_tokens, prompt_tokens);
            }
            if (prefix > best_prefix) {
                best = it;
                best_prefix = prefix;
            }
        }
        Slot* slot = *best;
        free_.erase(best);
        slot->resident_score_tokens.clear();
        slot->score_cache_valid = false;
        if (best_prefix <= 0) {
            slot->resident_generation_tokens.clear();
            slot->generation_cache_valid = false;
            best_prefix = 0;
        }
        return Lease(*this, slot, best_prefix, true);
    }

    Lease acquire_score(const std::vector<int>& tokens) {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [&] { return !free_.empty(); });
        auto best = free_.begin();
        int best_prefix = -1;
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            Slot* candidate = *it;
            int prefix = 0;
            if (candidate->score_cache_valid && candidate->adapter->causal() &&
                candidate->adapter->ar_seq_size(0) ==
                    static_cast<int>(candidate->resident_score_tokens.size())) {
                prefix = exact_token_prefix_length(candidate->resident_score_tokens, tokens);
            }
            if (prefix > best_prefix) {
                best = it;
                best_prefix = prefix;
            }
        }
        Slot* slot = *best;
        free_.erase(best);
        slot->resident_generation_tokens.clear();
        slot->generation_cache_valid = false;
        if (best_prefix <= 0) {
            slot->resident_score_tokens.clear();
            slot->score_cache_valid = false;
            best_prefix = 0;
        }
        return Lease(*this, slot, best_prefix, false, true);
    }

    PersistentLease acquire_persistent() {
        std::lock_guard<std::mutex> lock(persistent_mutex_);
        if (persistent_active_) {
            throw std::runtime_error("persistent reference-match context is busy");
        }
        persistent_active_ = true;
        try {
            return PersistentLease(*this, std::make_unique<GgmlAdapter>(
                model_, n_ctx(), flash_attention_, n_batch(), n_ubatch()));
        } catch (...) {
            persistent_active_ = false;
            throw;
        }
    }

    int size() const { return static_cast<int>(slots_.size()); }
    int n_ctx() const { return slots_.empty() ? 0 : slots_.front()->adapter->n_ctx(); }
    int n_batch() const { return slots_.empty() ? 0 : slots_.front()->adapter->n_batch(); }
    int n_ubatch() const { return slots_.empty() ? 0 : slots_.front()->adapter->n_ubatch(); }

    std::map<std::string, std::size_t> memory_breakdown() {
        if (slots_.empty()) return {};
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [&] { return !free_.empty(); });
        Slot* slot = free_.back();
        free_.pop_back();
        lock.unlock();
        try {
            const auto result = slot->adapter->memory_breakdown();
            release(slot);
            return result;
        } catch (...) {
            release(slot);
            throw;
        }
    }

private:
    void release(Slot* slot) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            free_.push_back(slot);
        }
        available_.notify_one();
    }

    void release_persistent() {
        std::lock_guard<std::mutex> lock(persistent_mutex_);
        persistent_active_ = false;
    }

    std::shared_ptr<GgmlModel> model_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<Slot*> free_;
    std::mutex mutex_;
    std::condition_variable available_;
    bool persistent_active_ = false;
    std::mutex persistent_mutex_;
    bool flash_attention_ = true;
};

}  // namespace clozn
