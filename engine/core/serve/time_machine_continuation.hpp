// Private worker protocol helpers for ADR 010 exact appended-turn continuation.
//
// These functions intentionally know nothing about a model or HTTP.  That keeps the
// fail-closed wire preconditions testable on a CPU-only build, while server_main.cpp
// remains the only place that can actually acquire a context and touch a checkpoint.
#pragma once

#include "sha256.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace clozn {

inline constexpr const char* TIME_MACHINE_CONTINUATION_PROTOCOL_VERSION =
    "clozn.time-machine-continuation-worker.v1";

// These codes are deliberately the public subset in clozn.time-machine-continuation.v1's
// ReasonCode enum.  The gateway translates the worker response into the closed receipt; it never
// has to infer a reason from an English exception string.
enum class TimeMachineContinuationCode {
    ContinuationCompleted,
    RequestInvalid,
    CheckpointUnavailable,
    CheckpointIdentityMismatch,
    WorkerGenerationStale,
    AppendTokensInvalid,
    AppendEmpty,
    RequestCancelled,
    WorkerProtocolError,
};

inline const char* time_machine_continuation_code(TimeMachineContinuationCode code) {
    switch (code) {
        case TimeMachineContinuationCode::ContinuationCompleted: return "continuation_completed";
        case TimeMachineContinuationCode::RequestInvalid: return "request_invalid";
        case TimeMachineContinuationCode::CheckpointUnavailable: return "checkpoint_unavailable";
        case TimeMachineContinuationCode::CheckpointIdentityMismatch:
            return "checkpoint_identity_mismatch";
        case TimeMachineContinuationCode::WorkerGenerationStale: return "worker_generation_stale";
        case TimeMachineContinuationCode::AppendTokensInvalid: return "append_tokens_invalid";
        case TimeMachineContinuationCode::AppendEmpty: return "append_empty";
        case TimeMachineContinuationCode::RequestCancelled: return "request_cancelled";
        case TimeMachineContinuationCode::WorkerProtocolError: return "worker_protocol_error";
    }
    return "worker_protocol_error";
}

struct TimeMachineContinuationRequest {
    std::string checkpoint_id;
    std::string worker_generation_id;
    int expected_n_past = 0;
    std::string expected_token_history_sha256;
    std::string expected_checkpoint_payload_sha256;
    std::vector<int> append_token_ids;
    std::string append_token_ids_sha256;
    int max_tokens = 0;
    std::string request_id;
};

struct TimeMachineContinuationCheckpointView {
    bool exists = false;
    std::string worker_generation_id;
    int n_past = 0;
    std::vector<int> historical_token_ids;
    std::string payload_sha256;
};

struct TimeMachineContinuationValidation {
    TimeMachineContinuationCode code = TimeMachineContinuationCode::RequestInvalid;
    std::string message;

    bool ok() const noexcept { return code == TimeMachineContinuationCode::ContinuationCompleted; }
};

inline bool time_machine_is_sha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (const char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    }
    return true;
}

// The private worker request is closed just like the public ADR 010 request.  In particular, a
// caller cannot smuggle a future sampler/template/adapter override into a v1 continuation and have
// an older worker silently ignore it while returning an apparently exact result.
inline bool time_machine_continuation_request_key_allowed(const std::string& key) {
    return key == "checkpoint_id" ||
           key == "worker_generation_id" ||
           key == "expected_n_past" ||
           key == "expected_token_history_sha256" ||
           key == "expected_checkpoint_payload_sha256" ||
           key == "append_token_ids" ||
           key == "append_token_ids_sha256" ||
           key == "max_tokens" ||
           key == "request_id" ||
           key == "checkpoint_on_finish";
}

inline void time_machine_sha256_update_u32_le(sha256_t& ctx, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(value & 0xffU),
        static_cast<unsigned char>((value >> 8U) & 0xffU),
        static_cast<unsigned char>((value >> 16U) & 0xffU),
        static_cast<unsigned char>((value >> 24U) & 0xffU),
    };
    sha256_update(&ctx, bytes, sizeof(bytes));
}

inline void time_machine_sha256_update_u64_le(sha256_t& ctx, std::uint64_t value) {
    const unsigned char bytes[8] = {
        static_cast<unsigned char>(value & 0xffULL),
        static_cast<unsigned char>((value >> 8U) & 0xffULL),
        static_cast<unsigned char>((value >> 16U) & 0xffULL),
        static_cast<unsigned char>((value >> 24U) & 0xffULL),
        static_cast<unsigned char>((value >> 32U) & 0xffULL),
        static_cast<unsigned char>((value >> 40U) & 0xffULL),
        static_cast<unsigned char>((value >> 48U) & 0xffULL),
        static_cast<unsigned char>((value >> 56U) & 0xffULL),
    };
    sha256_update(&ctx, bytes, sizeof(bytes));
}

inline void time_machine_sha256_update_f64_le(sha256_t& ctx, double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    time_machine_sha256_update_u64_le(ctx, bits);
}

inline std::string time_machine_sha256_finish(sha256_t& ctx) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const unsigned char byte : digest)
        out << std::setw(2) << static_cast<unsigned int>(byte);
    return out.str();
}

// This is the complete sampler provenance that an EngineCheckpoint can carry.  A checkpoint with
// has_sampler=false is an explicit greedy/default provenance, not a request-time invitation to pick
// new sampling settings.
struct TimeMachineSamplerProvenance {
    bool has_sampler = false;
    double temperature = 0.0;
    double rep_penalty = 1.0;
    int top_k = 0;
    double top_p = 1.0;
    std::uint64_t seed = 0;
    std::uint64_t rng_draws = 0;
};

inline std::string time_machine_sampler_config_sha256(const TimeMachineSamplerProvenance& sampler) {
    static constexpr char kDomain[] = "clozn.time-machine.sampler-config.v1";
    sha256_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<const unsigned char*>(kDomain), sizeof(kDomain));
    const unsigned char present = sampler.has_sampler ? 1 : 0;
    sha256_update(&ctx, &present, 1);
    time_machine_sha256_update_f64_le(ctx, sampler.temperature);
    time_machine_sha256_update_f64_le(ctx, sampler.rep_penalty);
    time_machine_sha256_update_u32_le(ctx, static_cast<std::uint32_t>(sampler.top_k));
    time_machine_sha256_update_f64_le(ctx, sampler.top_p);
    return time_machine_sha256_finish(ctx);
}

inline std::string time_machine_sampler_state_sha256(const TimeMachineSamplerProvenance& sampler) {
    static constexpr char kDomain[] = "clozn.time-machine.sampler-state.v1";
    sha256_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<const unsigned char*>(kDomain), sizeof(kDomain));
    time_machine_sha256_update_u64_le(ctx, sampler.seed);
    time_machine_sha256_update_u64_le(ctx, sampler.rng_draws);
    return time_machine_sha256_finish(ctx);
}

// Cross-language canonical token-id digest.  Python must hash the ASCII domain separator, then
// a uint32-le token count and each non-negative token id as uint32-le.  It is purpose-specific so
// a hash over the append cannot accidentally be relabelled as a historical-prefix hash in a receipt.
inline std::string time_machine_token_ids_sha256(const std::vector<int>& token_ids) {
    static constexpr char kDomain[] = "clozn.time-machine.token-ids.v1";
    sha256_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, reinterpret_cast<const unsigned char*>(kDomain), sizeof(kDomain));
    time_machine_sha256_update_u32_le(ctx, static_cast<std::uint32_t>(token_ids.size()));
    for (const int token_id : token_ids) {
        if (token_id < 0)
            throw std::invalid_argument("token ids must be non-negative");
        time_machine_sha256_update_u32_le(ctx, static_cast<std::uint32_t>(token_id));
    }
    return time_machine_sha256_finish(ctx);
}

inline TimeMachineContinuationValidation validate_time_machine_continuation(
    const TimeMachineContinuationRequest& request,
    const TimeMachineContinuationCheckpointView& checkpoint,
    int vocab_size,
    bool cancelled) {
    if (cancelled)
        return {TimeMachineContinuationCode::RequestCancelled, "request was cancelled"};
    if (request.checkpoint_id.empty() || request.worker_generation_id.empty() ||
        request.request_id.empty() || request.expected_n_past < 1 || request.max_tokens < 1 ||
        !time_machine_is_sha256(request.expected_token_history_sha256) ||
        !time_machine_is_sha256(request.expected_checkpoint_payload_sha256)) {
        return {TimeMachineContinuationCode::RequestInvalid,
                "checkpoint, request, position, limit, and SHA-256 proofs are required"};
    }
    if (request.worker_generation_id != checkpoint.worker_generation_id)
        return {TimeMachineContinuationCode::WorkerGenerationStale,
                "checkpoint belongs to a different worker generation"};
    if (!checkpoint.exists)
        return {TimeMachineContinuationCode::CheckpointUnavailable, "checkpoint is not live on this worker"};
    if (request.expected_n_past != checkpoint.n_past ||
        request.expected_token_history_sha256 !=
            time_machine_token_ids_sha256(checkpoint.historical_token_ids) ||
        request.expected_checkpoint_payload_sha256 != checkpoint.payload_sha256) {
        return {TimeMachineContinuationCode::CheckpointIdentityMismatch,
                "checkpoint position, history, or payload proof does not match"};
    }
    if (request.append_token_ids.empty())
        return {TimeMachineContinuationCode::AppendEmpty, "append_token_ids must be non-empty"};
    if (vocab_size < 1)
        return {TimeMachineContinuationCode::RequestInvalid, "worker reported an invalid vocabulary size"};
    // Do not hash malformed ids: the digest helper quite correctly rejects negative values, but a
    // malformed wire request must still receive the stable append_tokens_invalid response rather
    // than an exception that the HTTP framework might relabel as a generic server error.
    for (const int token_id : request.append_token_ids) {
        if (token_id < 0 || token_id >= vocab_size)
            return {TimeMachineContinuationCode::AppendTokensInvalid,
                    "append token id is outside the model vocabulary"};
    }
    if (!time_machine_is_sha256(request.append_token_ids_sha256) ||
        request.append_token_ids_sha256 != time_machine_token_ids_sha256(request.append_token_ids)) {
        return {TimeMachineContinuationCode::AppendTokensInvalid,
                "append_token_ids digest does not match"};
    }
    return {TimeMachineContinuationCode::ContinuationCompleted, "validated"};
}

}  // namespace clozn
