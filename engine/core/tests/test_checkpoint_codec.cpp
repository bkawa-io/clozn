// FORK-PIN-01: contract test for the checkpoint export/import envelope codec
// (serve/checkpoint_codec.hpp). Every EngineCheckpoint here is hand-built -- no GGUF, no context,
// no llama_state_seq_*_data call -- so this proves the CODEC's own fail-closed behavior (structural
// declared-vs-actual checks, payload-hash determinism and sensitivity, base64 round-trip) in
// isolation from live-model correctness, which scripts/smoke covers separately against a real GGUF.
#include <cstdio>
#include <string>

#include "checkpoint_codec.hpp"

using namespace clozn;

#define CHECK(condition)                                                                     \
    do {                                                                                     \
        if (!(condition)) {                                                                  \
            std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition);    \
            return 1;                                                                        \
        }                                                                                     \
    } while (false)

#define EXPECT_THROW(expr)                                                                    \
    do {                                                                                       \
        bool threw = false;                                                                    \
        try { (void)(expr); } catch (const std::exception&) { threw = true; }                  \
        if (!threw) {                                                                          \
            std::fprintf(stderr, "EXPECT_THROW failed at line %d: %s\n", __LINE__, #expr);     \
            return 1;                                                                          \
        }                                                                                       \
    } while (false)

namespace {

EngineCheckpoint make_plain_checkpoint() {
    EngineCheckpoint ckpt;
    ckpt.kv_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    ckpt.tokens = {100, 200, 300, 400, 500};
    ckpt.n_past = 5;
    ckpt.prompt_tokens = 3;
    ckpt.causal = true;
    return ckpt;
}

EngineCheckpoint make_full_checkpoint() {
    EngineCheckpoint ckpt = make_plain_checkpoint();
    ckpt.has_sampler = true;
    ckpt.temperature = 0.7;
    ckpt.rep_penalty = 1.1;
    ckpt.top_k = 40;
    ckpt.top_p = 0.9;
    ckpt.seed = 12345ULL;
    ckpt.rng_draws = 2ULL;
    ckpt.has_steer = true;
    ckpt.steer_cvec = {0.1f, 0.2f, 0.3f, 0.4f};  // pretend n_embd=2, n_layer=2
    ckpt.steer_lo = 1;
    ckpt.steer_hi = 1;
    return ckpt;
}

}  // namespace

int main() {
    // --- base64 round-trip -------------------------------------------------------------------
    {
        const std::vector<uint8_t> raw = {0, 1, 2, 253, 254, 255, 42};
        const std::string enc = checkpoint_base64_encode(raw.data(), raw.size());
        const std::vector<uint8_t> dec = checkpoint_base64_decode(enc);
        CHECK(dec == raw);

        const std::vector<uint8_t> empty;
        CHECK(checkpoint_base64_encode(empty.data(), 0).empty());
        CHECK(checkpoint_base64_decode("").empty());
    }

    // --- state JSON round-trip, plain checkpoint (no sampler, no steer) ----------------------
    {
        const EngineCheckpoint ckpt = make_plain_checkpoint();
        const json state = checkpoint_state_to_json(ckpt);
        CHECK(state["n_tokens"].get<int>() == 5);
        CHECK(state["has_sampler"].get<bool>() == false);
        CHECK(state["has_steer"].get<bool>() == false);
        CHECK(!state.contains("temperature"));
        CHECK(!state.contains("steer_cvec_b64"));

        const EngineCheckpoint back = checkpoint_state_from_json(state);
        CHECK(back.tokens == ckpt.tokens);
        CHECK(back.n_past == ckpt.n_past);
        CHECK(back.prompt_tokens == ckpt.prompt_tokens);
        CHECK(back.causal == ckpt.causal);
        CHECK(back.kv_data == ckpt.kv_data);
        CHECK(back.has_sampler == false);
        CHECK(back.has_steer == false);
    }

    // --- state JSON round-trip, full checkpoint (sampler + steer) ----------------------------
    {
        const EngineCheckpoint ckpt = make_full_checkpoint();
        const json state = checkpoint_state_to_json(ckpt);
        const EngineCheckpoint back = checkpoint_state_from_json(state);
        CHECK(back.has_sampler == true);
        CHECK(back.temperature == ckpt.temperature);
        CHECK(back.rep_penalty == ckpt.rep_penalty);
        CHECK(back.top_k == ckpt.top_k);
        CHECK(back.top_p == ckpt.top_p);
        CHECK(back.seed == ckpt.seed);
        CHECK(back.rng_draws == ckpt.rng_draws);
        CHECK(back.has_steer == true);
        CHECK(back.steer_cvec == ckpt.steer_cvec);
        CHECK(back.steer_lo == ckpt.steer_lo);
        CHECK(back.steer_hi == ckpt.steer_hi);
    }

    // --- fail closed: declared-vs-actual structural mismatches ------------------------------
    {
        const EngineCheckpoint ckpt = make_full_checkpoint();
        json state = checkpoint_state_to_json(ckpt);

        json bad_n_tokens = state; bad_n_tokens["n_tokens"] = 999;
        EXPECT_THROW(checkpoint_state_from_json(bad_n_tokens));

        json bad_kv_bytes = state; bad_kv_bytes["kv_bytes"] = 999;
        EXPECT_THROW(checkpoint_state_from_json(bad_kv_bytes));

        json bad_steer_dims = state; bad_steer_dims["steer_dims"] = 999;
        EXPECT_THROW(checkpoint_state_from_json(bad_steer_dims));

        json missing_kv = state; missing_kv.erase("kv_data_b64");
        EXPECT_THROW(checkpoint_state_from_json(missing_kv));

        json n_past_oob = state; n_past_oob["n_past"] = 999;
        EXPECT_THROW(checkpoint_state_from_json(n_past_oob));

        json prompt_tokens_oob = state; prompt_tokens_oob["prompt_tokens"] = 999;
        EXPECT_THROW(checkpoint_state_from_json(prompt_tokens_oob));

        json incomplete_sampler = state; incomplete_sampler.erase("seed");
        EXPECT_THROW(checkpoint_state_from_json(incomplete_sampler));

        json incomplete_steer = state; incomplete_steer.erase("steer_lo");
        EXPECT_THROW(checkpoint_state_from_json(incomplete_steer));

        json not_object = json::array({1, 2, 3});
        EXPECT_THROW(checkpoint_state_from_json(not_object));

        json empty_tokens = state; empty_tokens["tokens"] = json::array();
        EXPECT_THROW(checkpoint_state_from_json(empty_tokens));
    }

    // --- payload hash: deterministic, and sensitive to every field that matters ---------------
    {
        const EngineCheckpoint base = make_full_checkpoint();
        const std::string h0 = hash_checkpoint_payload(base);
        const std::string h1 = hash_checkpoint_payload(base);
        CHECK(h0 == h1);                 // deterministic
        CHECK(h0.size() == 64);          // hex sha256

        EngineCheckpoint kv_changed = base; kv_changed.kv_data.back() ^= 0xFF;
        CHECK(hash_checkpoint_payload(kv_changed) != h0);

        EngineCheckpoint tok_changed = base; tok_changed.tokens.back() += 1;
        CHECK(hash_checkpoint_payload(tok_changed) != h0);

        EngineCheckpoint npast_changed = base; npast_changed.n_past -= 1;
        CHECK(hash_checkpoint_payload(npast_changed) != h0);

        EngineCheckpoint prompt_changed = base; prompt_changed.prompt_tokens -= 1;
        CHECK(hash_checkpoint_payload(prompt_changed) != h0);

        EngineCheckpoint sampler_changed = base; sampler_changed.seed += 1;
        CHECK(hash_checkpoint_payload(sampler_changed) != h0);

        EngineCheckpoint steer_changed = base; steer_changed.steer_cvec[0] += 0.001f;
        CHECK(hash_checkpoint_payload(steer_changed) != h0);

        EngineCheckpoint no_sampler = base; no_sampler.has_sampler = false;
        CHECK(hash_checkpoint_payload(no_sampler) != h0);

        EngineCheckpoint no_steer = base; no_steer.has_steer = false;
        CHECK(hash_checkpoint_payload(no_steer) != h0);

        // A struct-identical copy must hash identically (no hidden nondeterminism, e.g. padding).
        const EngineCheckpoint copy = base;
        CHECK(hash_checkpoint_payload(copy) == h0);
    }

    // --- full envelope: shape + hash agrees with a direct call, envelope_version is stamped --
    {
        const EngineCheckpoint ckpt = make_full_checkpoint();
        const json identity{{"model_sha256", "deadbeef"}, {"n_embd", 2}, {"n_layer", 2}};
        const json env = checkpoint_export_envelope(ckpt, identity);
        CHECK(env["envelope_version"].get<std::string>() == CHECKPOINT_EXPORT_ENVELOPE_VERSION);
        CHECK(env["identity"] == identity);
        CHECK(env["payload_sha256"].get<std::string>() == hash_checkpoint_payload(ckpt));

        const EngineCheckpoint back = checkpoint_state_from_json(env["state"]);
        CHECK(hash_checkpoint_payload(back) == env["payload_sha256"].get<std::string>());
    }

    std::puts("checkpoint codec tests passed");
    return 0;
}
