#include "clozn/score_variants_protocol.hpp"

#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace {

using clozn::ScoreVariantRequestRow;
using clozn::ScoreVariantResultRow;
using clozn::ScoreVariantsParsedInteger;
using clozn::ScoreVariantsProtocolCode;
using clozn::ScoreVariantsRequest;
using clozn::ScoreVariantsResponse;
using json = nlohmann::json;

const json& conformance_fixture() {
    static const json fixture = [] {
        std::ifstream stream(CLOZN_SCORE_VARIANTS_FIXTURE);
        assert(stream.good());
        json document;
        stream >> document;
        return document.at("conformance");
    }();
    return fixture;
}

std::vector<ScoreVariantsParsedInteger> ids(std::initializer_list<std::int64_t> values) {
    std::vector<ScoreVariantsParsedInteger> result;
    for (const auto value : values) result.push_back(ScoreVariantsParsedInteger::integer(value));
    return result;
}

ScoreVariantsRequest valid_request(bool include_generation_state = false) {
    const auto& source = conformance_fixture().at("request");
    ScoreVariantsRequest request;
    for (const auto& variant : source.at("variants")) {
        ScoreVariantRequestRow row;
        row.id = variant.at("id").get<std::string>();
        for (const auto token : variant.at("prompt_ids"))
            row.prompt_token_ids.push_back(ScoreVariantsParsedInteger::integer(token.get<int>()));
        request.variants.push_back(std::move(row));
    }
    for (const auto token : source.at("continuation_ids"))
        request.continuation_token_ids.push_back(
            ScoreVariantsParsedInteger::integer(token.get<int>()));
    if (include_generation_state) {
        const auto& state = source.at("generation_state");
        request.has_generation_state = true;
        request.generation_state = {
            state.at("checkpoint_id").get<std::string>(),
            state.at("worker_generation_id").get<std::string>(),
            state.at("baseline_variant_id").get<std::string>(),
        };
    }
    return request;
}

ScoreVariantsResponse valid_response() {
    const auto& source = conformance_fixture().at("response");
    ScoreVariantsResponse response;
    response.status = source.at("status").get<std::string>();
    response.continuation_token_ids = source.at("continuation_ids").get<std::vector<int>>();
    for (const auto& item : source.at("results")) {
        response.results.push_back({
            item.at("id").get<std::string>(),
            item.at("status").get<std::string>(),
            item.at("prompt_token_count").get<int>(),
            item.at("target_logprobs").get<std::vector<double>>(),
            item.value("reason_code", ""),
        });
    }
    const auto& execution = source.at("execution");
    response.execution = {
        execution.at("strategy").get<std::string>(),
        execution.at("numerical_regime").get<std::string>(),
        execution.at("evidence_use").get<std::string>(),
        execution.at("qualified").get<bool>(),
        execution.at("execution_identity").at("algorithm_id").get<std::string>(),
        {},
    };
    for (const auto& item : execution.at("waves")) {
        response.execution.waves.push_back({
            item.at("wave_index").get<int>(),
            item.at("variant_ids").get<std::vector<std::string>>(),
            item.at("unique_prompt_count").get<int>(),
            item.at("logical_prompt_rows").get<long long>(),
            item.at("physical_prompt_rows").get<long long>(),
            item.at("continuation_kv_rows").get<long long>(),
            item.at("required_kv_rows").get<long long>(),
        });
    }
    return response;
}

void test_shared_conformance_fixture() {
    const auto request = clozn::validate_score_variants_request(valid_request(true), 128, 8);
    assert(request.ok());
    assert(clozn::validate_score_variants_response(valid_response(), request).ok());
}

void test_closed_request_shape_and_happy_path() {
    assert(clozn::score_variants_request_key_allowed("variants"));
    assert(clozn::score_variants_request_key_allowed("continuation_ids"));
    assert(clozn::score_variants_request_key_allowed("generation_state"));
    assert(!clozn::score_variants_request_key_allowed("continuation_token_ids"));
    assert(!clozn::score_variants_request_key_allowed("topk"));
    assert(!clozn::score_variants_request_key_allowed("execution_policy"));
    assert(clozn::score_variant_request_row_key_allowed("id"));
    assert(clozn::score_variant_request_row_key_allowed("prompt_ids"));
    assert(!clozn::score_variant_request_row_key_allowed("prompt_token_ids"));
    assert(!clozn::score_variant_request_row_key_allowed("prompt"));
    assert(clozn::score_variants_generation_state_key_allowed("checkpoint_id"));
    assert(clozn::score_variants_generation_state_key_allowed("worker_generation_id"));
    assert(clozn::score_variants_generation_state_key_allowed("baseline_variant_id"));
    assert(!clozn::score_variants_generation_state_key_allowed("keep_alive"));

    const auto request = clozn::validate_score_variants_request(valid_request(), 128, 8);
    assert(request.ok());
    assert((request.variant_ids == std::vector<std::string>{"baseline", "without-source:a"}));
    assert((request.prompt_token_ids[1] == std::vector<int>{1, 2, 4}));
    assert((request.continuation_token_ids == std::vector<int>{10, 11}));
    assert(clozn::validate_score_variants_response(valid_response(), request).ok());
}

void test_generation_state_reference_is_closed_and_names_a_variant() {
    auto request = valid_request();
    request.has_generation_state = true;
    request.generation_state = {
        "ckpt-worker-7", "worker", "baseline",
    };
    assert(clozn::validate_score_variants_request(request, 128, 8).ok());

    request.generation_state.baseline_variant_id = "absent";
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::RequestInvalid);

    request = valid_request();
    request.has_generation_state = true;
    request.generation_state = {"", "worker", "baseline"};
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::RequestInvalid);

    request.generation_state = {"ckpt-worker-7", "", "baseline"};
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::RequestInvalid);
}

void test_request_ids_are_nonempty_unique_bounded_and_ascii_safe() {
    auto request = valid_request();
    request.variants[0].id.clear();
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::VariantIdInvalid);

    request = valid_request();
    request.variants[1].id = request.variants[0].id;
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::DuplicateVariantId);

    request = valid_request();
    request.variants[0].id.assign(clozn::SCORE_VARIANTS_MAX_VARIANT_ID_BYTES + 1, 'a');
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::VariantIdInvalid);

    request = valid_request();
    request.variants[0].id = "contains space";
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::VariantIdInvalid);
}

void test_request_rejects_empty_and_non_integer_token_inputs() {
    auto request = valid_request();
    request.variants.clear();
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::VariantCountInvalid);

    request = valid_request();
    request.variants[0].prompt_token_ids.clear();
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::PromptTokenIdsInvalid);

    request = valid_request();
    request.continuation_token_ids.clear();
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::ContinuationTokenIdsInvalid);

    // Represents a JSON float/bool/null/etc. that the parser found where an integer was required.
    request = valid_request();
    request.variants[0].prompt_token_ids[0] = ScoreVariantsParsedInteger::non_integer();
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::PromptTokenIdsInvalid);

    request = valid_request();
    request.continuation_token_ids[0] = ScoreVariantsParsedInteger::non_integer();
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::ContinuationTokenIdsInvalid);

    request = valid_request();
    request.variants[0].prompt_token_ids[0] = ScoreVariantsParsedInteger::integer(-1);
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::PromptTokenIdsInvalid);

    request = valid_request();
    request.continuation_token_ids[0] = ScoreVariantsParsedInteger::integer(128);
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::ContinuationTokenIdsInvalid);
}

void test_request_hard_limits_are_enforced() {
    auto request = valid_request();
    request.variants.resize(clozn::SCORE_VARIANTS_MAX_VARIANTS + 1);
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::VariantCountInvalid);

    // Full prompt plus continuation, not merely N-1 resident continuation rows, is the v1
    // scalar-oracle-compatible context bound.
    request = valid_request();
    request.variants[0].prompt_token_ids = ids({1, 2, 3, 4, 5, 6, 7});
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::RequestLimitExceeded);

    request = valid_request();
    request.continuation_token_ids = ids({10, 11, 12, 13, 14, 15});
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::RequestLimitExceeded);

    // Sixty-four variants may still exceed the separate 16*n_ctx total-prompt input budget.
    request.variants.clear();
    request.continuation_token_ids = ids({10});
    for (int index = 0; index < 19; ++index) {
        request.variants.push_back(
            {"v" + std::to_string(index), ids({1, 2, 3, 4, 5, 6, 7})});
    }
    assert(clozn::validate_score_variants_request(request, 128, 8).code ==
           ScoreVariantsProtocolCode::RequestLimitExceeded);

    // The exact per-sequence and total-input boundaries remain accepted.
    request.variants.clear();
    for (int index = 0; index < 18; ++index) {
        request.variants.push_back(
            {"v" + std::to_string(index), ids({1, 2, 3, 4, 5, 6, 7})});
    }
    request.variants.push_back({"v18", ids({1, 2})});
    assert(clozn::validate_score_variants_request(request, 128, 8).ok());
}

void test_response_preserves_caller_order_and_closed_statuses() {
    const auto request = clozn::validate_score_variants_request(valid_request(), 128, 8);
    auto response = valid_response();

    response.continuation_token_ids[0] = 12;
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResponseInvalid);

    response = valid_response();
    response.results.pop_back();
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResponseInvalid);

    response = valid_response();
    std::swap(response.results[0], response.results[1]);
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultOrderInvalid);

    response = valid_response();
    response.results[0].status = "mystery";
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultStatusInvalid);

    for (const std::string status : {"failed", "cancelled", "not_started", "unsupported"}) {
        response = valid_response();
        response.status = "partial";
        response.results[1].status = status;
        response.results[1].target_logprobs.clear();
        response.results[1].reason_code = "fixture_reason";
        assert(clozn::validate_score_variants_response(response, request).ok());
    }
}

void test_completed_rows_require_exact_finite_scores() {
    const auto request = clozn::validate_score_variants_request(valid_request(), 128, 8);
    auto response = valid_response();
    response.results[0].target_logprobs.pop_back();
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultScoresInvalid);

    response = valid_response();
    response.results[0].target_logprobs[0] = std::numeric_limits<double>::infinity();
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultScoresInvalid);

    response = valid_response();
    response.results[0].target_logprobs[0] = std::numeric_limits<double>::quiet_NaN();
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultScoresInvalid);

    response = valid_response();
    response.results[0].reason_code = "not actually complete";
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultScoresInvalid);

    response = valid_response();
    response.results[0].status = "failed";
    response.results[0].reason_code = "decode_failed";
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultScoresInvalid);

    response = valid_response();
    response.results[0].status = "failed";
    response.results[0].target_logprobs.clear();
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResultScoresInvalid);
}

void test_response_requires_screening_only_native_execution_metadata() {
    const auto request = clozn::validate_score_variants_request(valid_request(), 128, 8);
    auto response = valid_response();

    response.execution.strategy = "scalar_reference";
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ExecutionMetadataInvalid);

    response = valid_response();
    response.execution.numerical_regime = "qualified";
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ExecutionMetadataInvalid);

    response = valid_response();
    response.execution.evidence_use = "trusted";
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ExecutionMetadataInvalid);

    response = valid_response();
    response.execution.qualified = true;
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ExecutionMetadataInvalid);

    response = valid_response();
    response.execution.execution_identity.clear();
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ExecutionMetadataInvalid);

    response = valid_response();
    response.status = "partial";
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ResponseInvalid);

    response = valid_response();
    response.execution.waves[0].variant_ids.pop_back();
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ExecutionMetadataInvalid);

    response = valid_response();
    response.execution.waves[0].required_kv_rows = 5;
    assert(clozn::validate_score_variants_response(response, request).code ==
           ScoreVariantsProtocolCode::ExecutionMetadataInvalid);
}

}  // namespace

int main() {
    test_shared_conformance_fixture();
    test_closed_request_shape_and_happy_path();
    test_generation_state_reference_is_closed_and_names_a_variant();
    test_request_ids_are_nonempty_unique_bounded_and_ascii_safe();
    test_request_rejects_empty_and_non_integer_token_inputs();
    test_request_hard_limits_are_enforced();
    test_response_preserves_caller_order_and_closed_statuses();
    test_completed_rows_require_exact_finite_scores();
    test_response_requires_screening_only_native_execution_metadata();
    return 0;
}
