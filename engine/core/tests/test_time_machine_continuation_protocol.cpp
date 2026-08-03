// Model-free contract test for ADR 010's private worker continuation preflight.
#include <cstdio>
#include <string>
#include <vector>

#include "time_machine_continuation.hpp"

using namespace clozn;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition); \
            return 1;                                                                     \
        }                                                                                 \
    } while (false)

namespace {

TimeMachineContinuationRequest valid_request(const TimeMachineContinuationCheckpointView& checkpoint) {
    TimeMachineContinuationRequest request;
    request.checkpoint_id = "ckpt-test-0";
    request.worker_generation_id = checkpoint.worker_generation_id;
    request.expected_n_past = checkpoint.n_past;
    request.expected_token_history_sha256 =
        time_machine_token_ids_sha256(checkpoint.historical_token_ids);
    request.expected_checkpoint_payload_sha256 = checkpoint.payload_sha256;
    request.append_token_ids = {31, 41, 59};
    request.append_token_ids_sha256 = time_machine_token_ids_sha256(request.append_token_ids);
    request.max_tokens = 16;
    request.request_id = "tmc-test";
    return request;
}

}  // namespace

int main() {
    const TimeMachineContinuationCheckpointView checkpoint{
        true, "generation-a", 3, {11, 22, 33},
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"};
    const TimeMachineContinuationRequest request = valid_request(checkpoint);

    // Hashes include their domain separator and length, so historical and append proof use the same
    // canonical primitive without allowing a length-extension/concatenation ambiguity.
    CHECK(time_machine_token_ids_sha256({11, 22, 33}) ==
          "e4f850e635cb02e677d8b46f8814e61f3c75fa287b1a3c17563f09b50d5fba0a");
    CHECK(time_machine_token_ids_sha256({11, 22, 33}) != time_machine_token_ids_sha256({11, 22, 33, 0}));
    CHECK(time_machine_is_sha256(request.append_token_ids_sha256));
    CHECK(!time_machine_is_sha256("abc"));
    CHECK(!time_machine_is_sha256("0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef"));
    CHECK(time_machine_continuation_request_key_allowed("checkpoint_id"));
    CHECK(time_machine_continuation_request_key_allowed("checkpoint_on_finish"));
    CHECK(!time_machine_continuation_request_key_allowed("temperature"));
    CHECK(!time_machine_continuation_request_key_allowed("adapter_state_sha256"));

    TimeMachineSamplerProvenance sampler;
    sampler.has_sampler = true;
    sampler.temperature = 0.7;
    sampler.rep_penalty = 1.1;
    sampler.top_k = 40;
    sampler.top_p = 0.9;
    sampler.seed = 99;
    sampler.rng_draws = 12;
    const std::string sampler_config = time_machine_sampler_config_sha256(sampler);
    const std::string sampler_state = time_machine_sampler_state_sha256(sampler);
    CHECK(time_machine_is_sha256(sampler_config));
    CHECK(time_machine_is_sha256(sampler_state));
    ++sampler.rng_draws;
    CHECK(time_machine_sampler_config_sha256(sampler) == sampler_config);
    CHECK(time_machine_sampler_state_sha256(sampler) != sampler_state);

    // Success means the route may acquire a lease.  Every rejection below occurs before model work.
    CHECK(validate_time_machine_continuation(request, checkpoint, 128, false).ok());

    TimeMachineContinuationCheckpointView absent = checkpoint;
    absent.exists = false;
    CHECK(validate_time_machine_continuation(request, absent, 128, false).code ==
          TimeMachineContinuationCode::CheckpointUnavailable);

    TimeMachineContinuationRequest absent_stale = request;
    absent_stale.worker_generation_id = "generation-old";
    CHECK(validate_time_machine_continuation(absent_stale, absent, 128, false).code ==
          TimeMachineContinuationCode::WorkerGenerationStale);

    TimeMachineContinuationRequest stale = request;
    stale.worker_generation_id = "generation-old";
    CHECK(validate_time_machine_continuation(stale, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::WorkerGenerationStale);

    TimeMachineContinuationRequest wrong_position = request;
    ++wrong_position.expected_n_past;
    CHECK(validate_time_machine_continuation(wrong_position, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::CheckpointIdentityMismatch);

    TimeMachineContinuationRequest wrong_history = request;
    wrong_history.expected_token_history_sha256.assign(64, 'a');
    CHECK(validate_time_machine_continuation(wrong_history, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::CheckpointIdentityMismatch);

    TimeMachineContinuationRequest wrong_payload = request;
    wrong_payload.expected_checkpoint_payload_sha256.assign(64, 'c');
    CHECK(validate_time_machine_continuation(wrong_payload, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::CheckpointIdentityMismatch);

    TimeMachineContinuationRequest empty_append = request;
    empty_append.append_token_ids.clear();
    empty_append.append_token_ids_sha256 = time_machine_token_ids_sha256({});
    CHECK(validate_time_machine_continuation(empty_append, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::AppendEmpty);

    TimeMachineContinuationRequest malformed_append = request;
    malformed_append.append_token_ids = {31, -1};
    malformed_append.append_token_ids_sha256.assign(64, 'b');
    CHECK(validate_time_machine_continuation(malformed_append, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::AppendTokensInvalid);

    TimeMachineContinuationRequest append_digest_mismatch = request;
    append_digest_mismatch.append_token_ids_sha256.assign(64, 'd');
    CHECK(validate_time_machine_continuation(append_digest_mismatch, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::AppendTokensInvalid);

    TimeMachineContinuationRequest out_of_vocab_append = request;
    out_of_vocab_append.append_token_ids = {128};
    out_of_vocab_append.append_token_ids_sha256 =
        time_machine_token_ids_sha256(out_of_vocab_append.append_token_ids);
    CHECK(validate_time_machine_continuation(out_of_vocab_append, checkpoint, 128, false).code ==
          TimeMachineContinuationCode::AppendTokensInvalid);

    CHECK(validate_time_machine_continuation(request, checkpoint, 128, true).code ==
          TimeMachineContinuationCode::RequestCancelled);
    CHECK(std::string(time_machine_continuation_code(
              TimeMachineContinuationCode::WorkerProtocolError)) == "worker_protocol_error");

    std::puts("time-machine continuation protocol tests passed");
    return 0;
}
