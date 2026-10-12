// Generated from protocol/fixtures/score_variants.json. Do not edit by hand.
#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace clozn::score_variants_contract {

inline constexpr std::string_view ENDPOINT = "/v1/score-variants";
inline constexpr std::string_view CAPABILITY = "score_variants";
inline constexpr std::string_view SCHEMA_VERSION = "clozn.score-variants-worker.v1";
inline constexpr std::size_t MAX_VARIANTS = 64;
inline constexpr std::size_t MAX_VARIANT_ID_BYTES = 128;
inline constexpr long long MAX_PROMPT_ROWS_PER_CONTEXT = 16;
inline constexpr std::size_t MAX_CHECKPOINT_ID_BYTES = 512;
inline constexpr std::size_t MAX_WORKER_GENERATION_ID_BYTES = 128;
inline constexpr std::array<std::string_view, 3> REQUEST_KEYS = {"variants", "continuation_ids", "generation_state"};
inline constexpr std::array<std::string_view, 2> VARIANT_KEYS = {"id", "prompt_ids"};
inline constexpr std::array<std::string_view, 3> GENERATION_STATE_KEYS = {"checkpoint_id", "worker_generation_id", "baseline_variant_id"};
inline constexpr std::array<std::string_view, 4> TOP_STATUSES = {"completed", "partial", "failed", "cancelled"};
inline constexpr std::array<std::string_view, 5> ROW_STATUSES = {"completed", "failed", "cancelled", "not_started", "unsupported"};
inline constexpr std::string_view STRATEGY_PREFIX = "native_";
inline constexpr std::string_view NUMERICAL_REGIME = "screening_unqualified";
inline constexpr std::string_view EVIDENCE_USE = "screening";
inline constexpr bool QUALIFIED = false;

}  // namespace clozn::score_variants_contract
