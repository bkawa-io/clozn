// serve/checkpoint_codec.hpp -- FORK-PIN-01: the wire envelope for durable checkpoint export/
// import. A checkpoint is worker-memory state (checkpoint_store.hpp); this header is what lets a
// caller take it OUT of that memory as self-describing, hashed bytes it can persist (Python's
// content-addressed blob store) and later hand back to /v1/checkpoint/import -- possibly to a
// DIFFERENT worker process, on a DIFFERENT day, after the ORIGINAL worker_generation_id is long
// gone. That is exactly the durability gap checkpoint_store.hpp's own docstring names.
//
// Two different hashes exist here on purpose, for two different jobs:
//   - payload_sha256 (hash_checkpoint_payload, below): an INNER hash the ENGINE itself computes at
//     export and re-verifies at import, over the checkpoint's own fields (kv bytes + tokens +
//     n_past + prompt_tokens + sampler + steer, in a fixed order) -- independent of how the bytes
//     were transported or stored. This is what "verify the blob hash before import" means at the
//     engine layer: it catches a bit flipped ANYWHERE in the exported state, not just in the KV
//     blob, before any of it is trusted enough to reconstruct an EngineCheckpoint.
//   - the Python blob store's own outer content-address (computed independently, over whatever
//     bytes it actually writes to disk) is a SEPARATE integrity layer (disk corruption / tampering
//     between write and read) -- this header has no opinion about it and never computes it.
//
// Every accepted import is additionally PROVE-LOADED: server_main.cpp's /v1/checkpoint/import route
// acquires a real context lease and calls load_checkpoint() on the reconstructed state before
// trusting it, so llama.cpp's OWN internal KV-format/version check (llama_state_seq_set_data) gets
// one more chance to refuse a blob this header's own checks didn't anticipate -- belt and braces,
// never a silent "close enough."
#pragma once

#include "nlohmann/json.hpp"

#include "clozn/model_ggml.hpp"
#include "sha256.h"  // vendored (third_party/sha256); self-guards extern "C" for a C++ caller

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace clozn {

using json = nlohmann::json;

// The envelope format version this build produces AND accepts. A future breaking change to the
// envelope shape ships as a NEW string (".v2"); this build refuses anything else outright rather
// than guess at a shape it was never tested against (see the import route's envelope_version check).
inline constexpr const char* CHECKPOINT_EXPORT_ENVELOPE_VERSION = "clozn.checkpoint-export.v1";

// ---- sha256 (vendored, third_party/sha256 -- see that directory's own provenance comments) ----

inline std::string sha256_bytes_to_hex(const unsigned char* digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_SIZE; ++i)
        out << std::setw(2) << static_cast<unsigned int>(digest[i]);
    return out.str();
}

// A running hash so the payload can be fed in pieces (kv bytes, then packed scalar fields, ...)
// without ever materializing one giant concatenated buffer just to hash it.
class RunningSha256 {
public:
    RunningSha256() { sha256_init(&ctx_); }
    void update(const void* data, size_t len) {
        sha256_update(&ctx_, static_cast<const unsigned char*>(data), len);
    }
    template <typename T>
    void update_pod(const T& value) {
        static_assert(std::is_trivially_copyable<T>::value, "update_pod needs a trivially copyable type");
        update(&value, sizeof(T));
    }
    std::string hex_digest() {
        unsigned char digest[SHA256_DIGEST_SIZE];
        sha256_final(&ctx_, digest);
        return sha256_bytes_to_hex(digest);
    }

private:
    sha256_t ctx_;
};

// Deterministic payload hash binding kv bytes + tokens + n_past + prompt_tokens + sampler + steer
// together, in this exact fixed order, at BOTH export and import time -- so a bit changed anywhere
// in the exported state (not just the KV blob itself) is caught before import ever trusts it. Uses
// fixed-width casts throughout so the hash is stable across builds of this same codebase (it is
// never compared cross-process against an INDEPENDENTLY computed hash -- only against the value
// this exact function produced at export time, carried alongside the envelope).
inline std::string hash_checkpoint_payload(const EngineCheckpoint& ckpt) {
    RunningSha256 h;
    h.update(ckpt.kv_data.data(), ckpt.kv_data.size());
    for (int t : ckpt.tokens) h.update_pod(static_cast<int32_t>(t));
    h.update_pod(static_cast<int32_t>(ckpt.n_past));
    h.update_pod(static_cast<int32_t>(ckpt.prompt_tokens));
    const unsigned char causal_byte = ckpt.causal ? 1 : 0;
    h.update(&causal_byte, 1);
    const unsigned char has_sampler_byte = ckpt.has_sampler ? 1 : 0;
    h.update(&has_sampler_byte, 1);
    if (ckpt.has_sampler) {
        h.update_pod(ckpt.temperature);
        h.update_pod(ckpt.rep_penalty);
        h.update_pod(static_cast<int32_t>(ckpt.top_k));
        h.update_pod(ckpt.top_p);
        h.update_pod(static_cast<uint64_t>(ckpt.seed));
        h.update_pod(static_cast<uint64_t>(ckpt.rng_draws));
    }
    const unsigned char has_steer_byte = ckpt.has_steer ? 1 : 0;
    h.update(&has_steer_byte, 1);
    if (ckpt.has_steer) {
        for (float f : ckpt.steer_cvec) h.update_pod(f);
        h.update_pod(static_cast<int32_t>(ckpt.steer_lo));
        h.update_pod(static_cast<int32_t>(ckpt.steer_hi));
    }
    return h.hex_digest();
}

// ---- base64 (encode mirrors server_shared.hpp's tensor_json_f32 convention; decode is new) ----

inline std::string checkpoint_base64_encode(const uint8_t* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out.push_back(tbl[(n >> 18) & 63]); out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(tbl[(n >> 6) & 63]);  out.push_back(tbl[n & 63]);
    }
    if (i < len) {
        uint32_t n = uint32_t(data[i]) << 16;
        const bool two = (i + 1 < len);
        if (two) n |= uint32_t(data[i + 1]) << 8;
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(two ? tbl[(n >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

inline std::vector<uint8_t> checkpoint_base64_decode(const std::string& in) {
    // Function-local static with an immediately-invoked initializer: C++11 guarantees this
    // initialization happens exactly once, safely, under concurrent first calls (clozn-server
    // handles requests on multiple threads) -- unlike a hand-rolled "static bool init" flag, which
    // would race two threads both observing !init before either finishes writing the table.
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) t[static_cast<unsigned char>(tbl[i])] = static_cast<int8_t>(i);
        return t;
    }();
    std::vector<uint8_t> out;
    out.reserve(in.size() / 4 * 3);
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        const int8_t digit = table[c];
        if (digit == -1) throw std::invalid_argument("checkpoint_base64_decode: invalid character");
        val = (val << 6) + digit;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

// ---- EngineCheckpoint <-> JSON state (the envelope's "state" object) ----
//
// checkpoint_state_from_json is where "token count and n_past" and "sampler-state shape" (the
// task's fail-closed axes) are structurally enforced: every field the encoder declared redundantly
// (n_tokens, kv_bytes, steer_dims) is checked against the actually-decoded array length, so a
// truncated/tampered transfer is refused here, before any identity or hash comparison even runs.

inline json checkpoint_state_to_json(const EngineCheckpoint& ckpt) {
    json j{
        {"n_tokens", static_cast<int>(ckpt.tokens.size())},
        {"tokens", ckpt.tokens},
        {"n_past", ckpt.n_past},
        {"prompt_tokens", ckpt.prompt_tokens},
        {"causal", ckpt.causal},
        {"has_sampler", ckpt.has_sampler},
        {"has_steer", ckpt.has_steer},
        {"kv_bytes", ckpt.kv_data.size()},
        {"kv_data_b64", checkpoint_base64_encode(ckpt.kv_data.data(), ckpt.kv_data.size())},
    };
    if (ckpt.has_sampler) {
        j["temperature"] = ckpt.temperature;
        j["rep_penalty"] = ckpt.rep_penalty;
        j["top_k"] = ckpt.top_k;
        j["top_p"] = ckpt.top_p;
        j["seed"] = ckpt.seed;
        j["rng_draws"] = ckpt.rng_draws;
    }
    if (ckpt.has_steer) {
        j["steer_lo"] = ckpt.steer_lo;
        j["steer_hi"] = ckpt.steer_hi;
        j["steer_dims"] = static_cast<int>(ckpt.steer_cvec.size());
        const auto* bytes = reinterpret_cast<const uint8_t*>(ckpt.steer_cvec.data());
        j["steer_cvec_b64"] = checkpoint_base64_encode(bytes, ckpt.steer_cvec.size() * sizeof(float));
    }
    return j;
}

inline EngineCheckpoint checkpoint_state_from_json(const json& j) {
    if (!j.is_object())
        throw std::invalid_argument("checkpoint state must be a JSON object");

    EngineCheckpoint ckpt;
    if (!j.contains("tokens") || !j["tokens"].is_array() || j["tokens"].empty())
        throw std::invalid_argument("checkpoint state missing a non-empty 'tokens' array");
    ckpt.tokens = j["tokens"].get<std::vector<int>>();
    if (!j.contains("n_tokens") || !j["n_tokens"].is_number_integer())
        throw std::invalid_argument("checkpoint state missing integer 'n_tokens'");
    if (j["n_tokens"].get<int>() != static_cast<int>(ckpt.tokens.size()))
        throw std::invalid_argument("declared n_tokens does not match the tokens array length");

    if (!j.contains("n_past") || !j["n_past"].is_number_integer())
        throw std::invalid_argument("checkpoint state missing integer 'n_past'");
    ckpt.n_past = j["n_past"].get<int>();
    if (ckpt.n_past < 1 || ckpt.n_past > static_cast<int>(ckpt.tokens.size()))
        throw std::invalid_argument("n_past out of range for the declared tokens");

    if (!j.contains("prompt_tokens") || !j["prompt_tokens"].is_number_integer())
        throw std::invalid_argument("checkpoint state missing integer 'prompt_tokens'");
    ckpt.prompt_tokens = j["prompt_tokens"].get<int>();
    if (ckpt.prompt_tokens < 0 || ckpt.prompt_tokens > ckpt.n_past)
        throw std::invalid_argument("prompt_tokens out of range for n_past");

    ckpt.causal = j.value("causal", true);
    ckpt.has_sampler = j.value("has_sampler", false);
    if (ckpt.has_sampler) {
        if (!j.contains("temperature") || !j.contains("rep_penalty") || !j.contains("top_k") ||
            !j.contains("top_p") || !j.contains("seed") || !j.contains("rng_draws"))
            throw std::invalid_argument("has_sampler=true but the sampler fields are incomplete");
        ckpt.temperature = j["temperature"].get<double>();
        ckpt.rep_penalty = j["rep_penalty"].get<double>();
        ckpt.top_k = j["top_k"].get<int>();
        ckpt.top_p = j["top_p"].get<double>();
        ckpt.seed = j["seed"].get<uint64_t>();
        ckpt.rng_draws = j["rng_draws"].get<uint64_t>();
    }

    ckpt.has_steer = j.value("has_steer", false);
    if (ckpt.has_steer) {
        if (!j.contains("steer_cvec_b64") || !j["steer_cvec_b64"].is_string() ||
            !j.contains("steer_lo") || !j.contains("steer_hi") || !j.contains("steer_dims"))
            throw std::invalid_argument("has_steer=true but the steer fields are incomplete");
        const std::vector<uint8_t> raw = checkpoint_base64_decode(j["steer_cvec_b64"].get<std::string>());
        if (raw.size() % sizeof(float) != 0)
            throw std::invalid_argument("steer_cvec_b64 is not a whole number of float32 elements");
        const int declared_dims = j["steer_dims"].get<int>();
        if (declared_dims != static_cast<int>(raw.size() / sizeof(float)))
            throw std::invalid_argument("declared steer_dims does not match steer_cvec_b64 length");
        ckpt.steer_cvec.resize(raw.size() / sizeof(float));
        std::memcpy(ckpt.steer_cvec.data(), raw.data(), raw.size());
        ckpt.steer_lo = j["steer_lo"].get<int>();
        ckpt.steer_hi = j["steer_hi"].get<int>();
    }

    if (!j.contains("kv_data_b64") || !j["kv_data_b64"].is_string())
        throw std::invalid_argument("checkpoint state missing 'kv_data_b64'");
    ckpt.kv_data = checkpoint_base64_decode(j["kv_data_b64"].get<std::string>());
    if (ckpt.kv_data.empty())
        throw std::invalid_argument("kv_data_b64 decoded to zero bytes");
    if (!j.contains("kv_bytes") || !j["kv_bytes"].is_number_integer())
        throw std::invalid_argument("checkpoint state missing integer 'kv_bytes'");
    if (j["kv_bytes"].get<int64_t>() < 0 ||
        static_cast<uint64_t>(j["kv_bytes"].get<int64_t>()) != static_cast<uint64_t>(ckpt.kv_data.size()))
        throw std::invalid_argument("declared kv_bytes does not match the decoded kv_data length");

    return ckpt;
}

// The full export envelope: identity (caller-supplied, e.g. server_main.cpp's checkpoint_identity_json())
// + state + the inner payload hash. `identity` is opaque here -- this header has no opinion about
// WHICH fields matter for compatibility; that fail-closed comparison belongs to the import route,
// which knows what "this worker" actually is.
inline json checkpoint_export_envelope(const EngineCheckpoint& ckpt, const json& identity) {
    json env{
        {"envelope_version", CHECKPOINT_EXPORT_ENVELOPE_VERSION},
        {"identity", identity},
        {"state", checkpoint_state_to_json(ckpt)},
    };
    env["payload_sha256"] = hash_checkpoint_payload(ckpt);
    return env;
}

}  // namespace clozn
