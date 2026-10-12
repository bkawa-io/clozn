// Model-free wire contract for scoring one recorded continuation under related prompts.
//
// This header intentionally has no JSON, HTTP, llama.cpp, or model dependency.  A route parser
// maps its closed JSON shape into these structs, preserving whether each token-id value was an
// actual JSON integer.  Validation then fails closed before a context lease or model work begins.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "clozn/score_variants_contract.generated.hpp"

namespace clozn {

inline constexpr const char* SCORE_VARIANTS_PROTOCOL_VERSION =
    score_variants_contract::SCHEMA_VERSION.data();
inline constexpr std::size_t SCORE_VARIANTS_MAX_VARIANTS =
    score_variants_contract::MAX_VARIANTS;
inline constexpr std::size_t SCORE_VARIANTS_MAX_VARIANT_ID_BYTES =
    score_variants_contract::MAX_VARIANT_ID_BYTES;
inline constexpr long long SCORE_VARIANTS_MAX_PROMPT_ROWS_PER_CONTEXT =
    score_variants_contract::MAX_PROMPT_ROWS_PER_CONTEXT;

template <std::size_t N>
inline bool score_variants_contract_contains(
    const std::array<std::string_view, N>& values,
    const std::string& value) {
    for (const auto candidate : values) {
        if (candidate == value) return true;
    }
    return false;
}

// The route parser must distinguish a JSON integer from every other JSON value.  `value` is only
// meaningful when kind == Integer; NonInteger represents a float (including 1.0), bool, null,
// string, object, or array that appeared where a token id was required.
enum class ScoreVariantsParsedIntegerKind {
    Integer,
    NonInteger,
};

struct ScoreVariantsParsedInteger {
    ScoreVariantsParsedIntegerKind kind = ScoreVariantsParsedIntegerKind::NonInteger;
    std::int64_t value = 0;

    static ScoreVariantsParsedInteger integer(std::int64_t value) {
        return {ScoreVariantsParsedIntegerKind::Integer, value};
    }

    static ScoreVariantsParsedInteger non_integer() {
        return {ScoreVariantsParsedIntegerKind::NonInteger, 0};
    }
};

struct ScoreVariantRequestRow {
    std::string id;
    std::vector<ScoreVariantsParsedInteger> prompt_token_ids;
};

struct ScoreVariantsGenerationStateRequest {
    std::string checkpoint_id;
    std::string worker_generation_id;
    std::string baseline_variant_id;
};

struct ScoreVariantsRequest {
    std::vector<ScoreVariantRequestRow> variants;
    std::vector<ScoreVariantsParsedInteger> continuation_token_ids;
    // Optional acceleration hint. It names a bounded checkpoint owned by this exact worker and the
    // variant whose prompt must equal that checkpoint's declared original prompt. Failure to reuse
    // never changes scoring semantics; the runtime reports why it performed an ordinary prefill.
    bool has_generation_state = false;
    ScoreVariantsGenerationStateRequest generation_state;
};

// These helpers let a JSON route enforce the v1 closed shape before it constructs the parsed
// structs.  Unknown fields are rejected, never silently ignored as a future execution override.
inline bool score_variants_request_key_allowed(const std::string& key) {
    return score_variants_contract_contains(score_variants_contract::REQUEST_KEYS, key);
}

inline bool score_variant_request_row_key_allowed(const std::string& key) {
    return score_variants_contract_contains(score_variants_contract::VARIANT_KEYS, key);
}

inline bool score_variants_generation_state_key_allowed(const std::string& key) {
    return score_variants_contract_contains(
        score_variants_contract::GENERATION_STATE_KEYS, key);
}

enum class ScoreVariantsProtocolCode {
    Valid,
    RequestInvalid,
    VariantCountInvalid,
    VariantIdInvalid,
    DuplicateVariantId,
    PromptTokenIdsInvalid,
    ContinuationTokenIdsInvalid,
    RequestLimitExceeded,
    ResponseInvalid,
    ResultOrderInvalid,
    ResultStatusInvalid,
    ResultScoresInvalid,
    ExecutionMetadataInvalid,
};

inline const char* score_variants_protocol_code(ScoreVariantsProtocolCode code) {
    switch (code) {
        case ScoreVariantsProtocolCode::Valid: return "valid";
        case ScoreVariantsProtocolCode::RequestInvalid: return "request_invalid";
        case ScoreVariantsProtocolCode::VariantCountInvalid: return "variant_count_invalid";
        case ScoreVariantsProtocolCode::VariantIdInvalid: return "variant_id_invalid";
        case ScoreVariantsProtocolCode::DuplicateVariantId: return "duplicate_variant_id";
        case ScoreVariantsProtocolCode::PromptTokenIdsInvalid: return "prompt_token_ids_invalid";
        case ScoreVariantsProtocolCode::ContinuationTokenIdsInvalid:
            return "continuation_token_ids_invalid";
        case ScoreVariantsProtocolCode::RequestLimitExceeded: return "request_limit_exceeded";
        case ScoreVariantsProtocolCode::ResponseInvalid: return "response_invalid";
        case ScoreVariantsProtocolCode::ResultOrderInvalid: return "result_order_invalid";
        case ScoreVariantsProtocolCode::ResultStatusInvalid: return "result_status_invalid";
        case ScoreVariantsProtocolCode::ResultScoresInvalid: return "result_scores_invalid";
        case ScoreVariantsProtocolCode::ExecutionMetadataInvalid:
            return "execution_metadata_invalid";
    }
    return "response_invalid";
}

struct ScoreVariantsRequestValidation {
    ScoreVariantsProtocolCode code = ScoreVariantsProtocolCode::RequestInvalid;
    std::string message;
    std::vector<std::string> variant_ids;
    std::vector<std::vector<int>> prompt_token_ids;
    std::vector<int> continuation_token_ids;

    bool ok() const noexcept { return code == ScoreVariantsProtocolCode::Valid; }
};

struct ScoreVariantsValidation {
    ScoreVariantsProtocolCode code = ScoreVariantsProtocolCode::ResponseInvalid;
    std::string message;

    bool ok() const noexcept { return code == ScoreVariantsProtocolCode::Valid; }
};

inline bool score_variant_id_is_ascii_safe(const std::string& id) {
    if (id.empty() || id.size() > SCORE_VARIANTS_MAX_VARIANT_ID_BYTES) return false;
    for (const unsigned char c : id) {
        const bool alpha_numeric = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                   (c >= '0' && c <= '9');
        if (!alpha_numeric && c != '_' && c != '-' && c != '.' && c != ':') return false;
    }
    return true;
}

inline bool score_variants_parse_token_ids(
    const std::vector<ScoreVariantsParsedInteger>& parsed,
    int vocab_size,
    std::vector<int>& out) {
    out.clear();
    out.reserve(parsed.size());
    for (const auto& token : parsed) {
        if (token.kind != ScoreVariantsParsedIntegerKind::Integer || token.value < 0 ||
            token.value >= static_cast<std::int64_t>(vocab_size) ||
            token.value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            out.clear();
            return false;
        }
        out.push_back(static_cast<int>(token.value));
    }
    return true;
}

inline ScoreVariantsRequestValidation validate_score_variants_request(
    const ScoreVariantsRequest& request,
    int vocab_size,
    int n_ctx) {
    ScoreVariantsRequestValidation result;
    if (vocab_size < 1 || n_ctx < 1) {
        result.code = ScoreVariantsProtocolCode::RequestInvalid;
        result.message = "worker vocabulary and context limits must be positive";
        return result;
    }
    if (request.variants.empty() || request.variants.size() > SCORE_VARIANTS_MAX_VARIANTS) {
        result.code = ScoreVariantsProtocolCode::VariantCountInvalid;
        result.message = "variants must contain between 1 and 64 rows";
        return result;
    }
    if (request.continuation_token_ids.empty()) {
        result.code = ScoreVariantsProtocolCode::ContinuationTokenIdsInvalid;
        result.message = "continuation_ids must be non-empty";
        return result;
    }
    if (!score_variants_parse_token_ids(
            request.continuation_token_ids, vocab_size, result.continuation_token_ids)) {
        result.code = ScoreVariantsProtocolCode::ContinuationTokenIdsInvalid;
        result.message = "continuation_ids must be in-vocabulary JSON integers";
        return result;
    }
    if (request.has_generation_state) {
        const auto& state = request.generation_state;
        if (state.checkpoint_id.empty() ||
            state.checkpoint_id.size() > score_variants_contract::MAX_CHECKPOINT_ID_BYTES ||
            state.worker_generation_id.empty() ||
            state.worker_generation_id.size() >
                score_variants_contract::MAX_WORKER_GENERATION_ID_BYTES ||
            !score_variant_id_is_ascii_safe(state.baseline_variant_id)) {
            result.code = ScoreVariantsProtocolCode::RequestInvalid;
            result.message =
                "generation_state requires bounded non-empty checkpoint_id, "
                "worker_generation_id, and a valid baseline_variant_id";
            return result;
        }
    }

    const long long max_total_prompt_ids =
        SCORE_VARIANTS_MAX_PROMPT_ROWS_PER_CONTEXT * static_cast<long long>(n_ctx);
    long long total_prompt_ids = 0;
    std::unordered_set<std::string> seen_ids;
    result.variant_ids.reserve(request.variants.size());
    result.prompt_token_ids.reserve(request.variants.size());
    for (const auto& variant : request.variants) {
        if (!score_variant_id_is_ascii_safe(variant.id)) {
            result.code = ScoreVariantsProtocolCode::VariantIdInvalid;
            result.message = "variant ids must be 1..128 ASCII-safe characters";
            return result;
        }
        if (!seen_ids.insert(variant.id).second) {
            result.code = ScoreVariantsProtocolCode::DuplicateVariantId;
            result.message = "variant ids must be unique";
            return result;
        }
        if (variant.prompt_token_ids.empty()) {
            result.code = ScoreVariantsProtocolCode::PromptTokenIdsInvalid;
            result.message = "every variant prompt_ids array must be non-empty";
            return result;
        }
        std::vector<int> prompt;
        if (!score_variants_parse_token_ids(variant.prompt_token_ids, vocab_size, prompt)) {
            result.code = ScoreVariantsProtocolCode::PromptTokenIdsInvalid;
            result.message = "prompt_ids must be in-vocabulary JSON integers";
            return result;
        }

        const long long prompt_size = static_cast<long long>(prompt.size());
        const long long continuation_size =
            static_cast<long long>(result.continuation_token_ids.size());
        if (prompt_size + continuation_size > static_cast<long long>(n_ctx)) {
            result.code = ScoreVariantsProtocolCode::RequestLimitExceeded;
            result.message = "each prompt plus the full continuation must fit within n_ctx";
            return result;
        }
        if (total_prompt_ids > max_total_prompt_ids - prompt_size) {
            result.code = ScoreVariantsProtocolCode::RequestLimitExceeded;
            result.message = "total prompt token ids exceed 16 * n_ctx";
            return result;
        }
        total_prompt_ids += prompt_size;
        result.variant_ids.push_back(variant.id);
        result.prompt_token_ids.push_back(std::move(prompt));
    }
    if (request.has_generation_state &&
        seen_ids.find(request.generation_state.baseline_variant_id) == seen_ids.end()) {
        result.code = ScoreVariantsProtocolCode::RequestInvalid;
        result.message = "generation_state baseline_variant_id must name a request variant";
        return result;
    }
    result.code = ScoreVariantsProtocolCode::Valid;
    result.message = "validated";
    return result;
}

inline constexpr const char* SCORE_VARIANT_STATUS_COMPLETED =
    score_variants_contract::ROW_STATUSES[0].data();
inline constexpr const char* SCORE_VARIANT_STATUS_FAILED =
    score_variants_contract::ROW_STATUSES[1].data();
inline constexpr const char* SCORE_VARIANT_STATUS_CANCELLED =
    score_variants_contract::ROW_STATUSES[2].data();
inline constexpr const char* SCORE_VARIANT_STATUS_NOT_STARTED =
    score_variants_contract::ROW_STATUSES[3].data();
inline constexpr const char* SCORE_VARIANT_STATUS_UNSUPPORTED =
    score_variants_contract::ROW_STATUSES[4].data();

inline bool score_variant_result_status_allowed(const std::string& status) {
    return score_variants_contract_contains(score_variants_contract::ROW_STATUSES, status);
}

struct ScoreVariantResultRow {
    std::string id;
    std::string status;
    int prompt_token_count = 0;
    std::vector<double> target_logprobs;
    std::string reason_code;
};

struct ScoreVariantsExecutionMetadata {
    std::string strategy;
    std::string numerical_regime;
    std::string evidence_use;
    bool qualified = false;
    std::string execution_identity;
    struct Wave {
        int wave_index = -1;
        std::vector<std::string> variant_ids;
        int unique_prompt_count = 0;
        long long logical_prompt_rows = 0;
        long long physical_prompt_rows = 0;
        long long continuation_kv_rows = 0;
        long long required_kv_rows = 0;
    };
    std::vector<Wave> waves;
};

struct ScoreVariantsResponse {
    std::string status;
    std::vector<int> continuation_token_ids;
    std::vector<ScoreVariantResultRow> results;
    ScoreVariantsExecutionMetadata execution;
};

inline ScoreVariantsValidation validate_score_variants_response(
    const ScoreVariantsResponse& response,
    const ScoreVariantsRequestValidation& request) {
    if (!request.ok()) {
        return {ScoreVariantsProtocolCode::RequestInvalid,
                "cannot validate a response against an invalid request"};
    }
    const auto& execution = response.execution;
    const std::string strategy_prefix(score_variants_contract::STRATEGY_PREFIX);
    if (execution.strategy.size() <= strategy_prefix.size() ||
        execution.strategy.rfind(strategy_prefix, 0) != 0 ||
        execution.numerical_regime != score_variants_contract::NUMERICAL_REGIME ||
        execution.evidence_use != score_variants_contract::EVIDENCE_USE ||
        execution.qualified != score_variants_contract::QUALIFIED ||
        execution.execution_identity.empty()) {
        return {ScoreVariantsProtocolCode::ExecutionMetadataInvalid,
                "v1 execution metadata must describe unqualified native screening"};
    }
    if (response.continuation_token_ids != request.continuation_token_ids ||
        response.results.size() != request.variant_ids.size()) {
        return {ScoreVariantsProtocolCode::ResponseInvalid,
                "response continuation or row count does not match the request"};
    }
    std::size_t completed_count = 0;
    for (std::size_t index = 0; index < response.results.size(); ++index) {
        const auto& row = response.results[index];
        if (row.id != request.variant_ids[index] ||
            row.prompt_token_count !=
                static_cast<int>(request.prompt_token_ids[index].size())) {
            return {ScoreVariantsProtocolCode::ResultOrderInvalid,
                    "result ids and prompt counts must remain in caller order"};
        }
        if (!score_variant_result_status_allowed(row.status)) {
            return {ScoreVariantsProtocolCode::ResultStatusInvalid,
                    "result row has an unsupported status"};
        }
        if (row.status == SCORE_VARIANT_STATUS_COMPLETED) {
            ++completed_count;
            if (!row.reason_code.empty() ||
                row.target_logprobs.size() != request.continuation_token_ids.size()) {
                return {ScoreVariantsProtocolCode::ResultScoresInvalid,
                        "completed rows need exact-length scores and no failure reason"};
            }
            for (const double value : row.target_logprobs) {
                if (!std::isfinite(value)) {
                    return {ScoreVariantsProtocolCode::ResultScoresInvalid,
                            "completed target logprobs must be finite"};
                }
            }
        } else if (!row.target_logprobs.empty() || row.reason_code.empty()) {
            return {ScoreVariantsProtocolCode::ResultScoresInvalid,
                    "incomplete rows need a reason and cannot carry partial scores"};
        }
    }
    const bool completed = response.status == score_variants_contract::TOP_STATUSES[0] &&
                           completed_count == response.results.size();
    const bool partial = response.status == score_variants_contract::TOP_STATUSES[1] &&
                         completed_count > 0 &&
                         completed_count < response.results.size();
    const bool failed_or_cancelled =
        (response.status == score_variants_contract::TOP_STATUSES[2] ||
         response.status == score_variants_contract::TOP_STATUSES[3]) &&
        completed_count == 0;
    if (!completed && !partial && !failed_or_cancelled) {
        return {ScoreVariantsProtocolCode::ResponseInvalid,
                "top-level status does not agree with result rows"};
    }

    if (execution.waves.empty()) {
        return {ScoreVariantsProtocolCode::ExecutionMetadataInvalid,
                "execution metadata must contain at least one wave"};
    }
    std::unordered_set<std::string> wave_members;
    for (std::size_t index = 0; index < execution.waves.size(); ++index) {
        const auto& wave = execution.waves[index];
        if (wave.wave_index != static_cast<int>(index) || wave.variant_ids.empty() ||
            wave.unique_prompt_count <= 0 ||
            wave.unique_prompt_count > static_cast<int>(wave.variant_ids.size()) ||
            wave.logical_prompt_rows < 0 || wave.physical_prompt_rows < 0 ||
            wave.continuation_kv_rows < 0 || wave.required_kv_rows < 0 ||
            wave.physical_prompt_rows > wave.logical_prompt_rows ||
            wave.required_kv_rows < wave.physical_prompt_rows + wave.continuation_kv_rows) {
            return {ScoreVariantsProtocolCode::ExecutionMetadataInvalid,
                    "execution wave metadata is malformed"};
        }
        for (const auto& variant_id : wave.variant_ids) {
            if (!wave_members.insert(variant_id).second) {
                return {ScoreVariantsProtocolCode::ExecutionMetadataInvalid,
                        "execution wave membership contains duplicates"};
            }
        }
    }
    if (wave_members.size() != request.variant_ids.size()) {
        return {ScoreVariantsProtocolCode::ExecutionMetadataInvalid,
                "execution wave membership is incomplete"};
    }
    for (const auto& variant_id : request.variant_ids) {
        if (wave_members.find(variant_id) == wave_members.end()) {
            return {ScoreVariantsProtocolCode::ExecutionMetadataInvalid,
                    "execution wave membership does not match the request"};
        }
    }
    return {ScoreVariantsProtocolCode::Valid, "validated"};
}

}  // namespace clozn
