// clozn/events.hpp — typed generation events (DESIGN §5.1), the event-sourced spine, C++ port
// of lab/clozn_lab/scheduler/events.py. The scheduler emits these; the CLI, benchmarks, logs, and
// the future server are consumers only (DESIGN invariant 2). Field names are the §5.1 wire keys
// verbatim (t/type/pos/id/conf/old/span/...), so JSONL logs are these structs serialized with no
// mapping layer — replayable across the lab and the C++ core.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace clozn {

// One inked token, as it appears in tokens_committed items.
struct CommitItem {
    int pos;
    int id;
    double conf;
};

// One re-masked token (remask_lowconf, §5.2); unused until revisions exist in the C++ loop.
struct ReviseItem {
    int pos;
    int old;
    int id;
    double conf;
};

struct WorkspaceReadoutItem {
    std::string label;
    double score;
};

struct GenStarted {
    int t;
    int prompt_tokens;
    int block_len;  // 0 = whole-sequence (§5.4)
    int max_new;
};

struct BlockStarted {
    int t;
    int block;
    std::pair<int, int> span;  // [start, end) board positions
};

struct TokensCommitted {
    int t;
    int block;
    std::vector<CommitItem> items;
};

struct TokensRevised {
    int t;
    int block;
    std::vector<ReviseItem> items;
};

struct StepStats {
    int t;
    int block;
    int step;
    int committed;
    int remaining;
    double ms;
    double cache_hit;
};

struct BlockFinalized {
    int t;
    int block;
    std::string text;
    int steps_used;
};

// One protocol-native timing phase. start_ns is local to this worker request's steady-clock origin;
// -1 means the duration is measured but its offset is not available. aggregation keeps overlapping
// transport/context spans visible without allowing consumers to double-count them.
struct PerformancePhase {
    std::string name;
    std::string owner = "clozn_worker";
    std::int64_t duration_ns = 0;
    std::int64_t start_ns = -1;
    std::string aggregation = "exclusive";
    std::string scope;
    std::vector<std::string> includes;
};

struct GenFinished {
    int t;
    std::string reason;  // "eos" | "length" | "steps_exhausted"
    int new_tokens;
    double wall_ms;
    int steps_total;
    double tok_per_s;
    std::vector<PerformancePhase> timing_phases;
    int prompt_tokens = 0;
    double prompt_tok_per_s = -1.0;
    double decode_tok_per_s = -1.0;
};

// White-box feature tap (Tier 2): per-pass concept-feature activations on the active block. `features`
// are the K concept names; `scores` is [positions.size() * K] position-major (scores[i*K + k] =
// feature k on positions[i]). Emitted only when the adapter's activation tap is on — a pure observer
// like every other event, so it never touches the board (invariant 2; goldens untouched).
struct StepFeatures {
    int t;
    int block;
    std::vector<int> positions;
    std::vector<std::string> features;
    std::vector<float> scores;  // [positions.size() * features.size()], position-major
};

// White-box logit-lens (Tier 1): the top-k token CANDIDATES per still-masked slot this pass —
// "what is this blank considering, and how confidently". `ids`/`probs` are [positions.size()*k]
// position-major. The server decodes ids -> pieces for the viz (the events stay tokenizer-free).
struct StepLens {
    int t;
    int block;
    std::vector<int> positions;
    int k = 0;
    std::vector<int> ids;        // [positions.size() * k]
    std::vector<float> probs;    // [positions.size() * k]
};

// White-box RAW activation tap (Tier 2, the heavy state): the per-position hidden state itself,
// the substrate's "memory" slice this pass (the state-stream protocol's `StateStep.state`). Unlike
// StepFeatures (which PROJECTS the activations onto concept probes => K scalars/slot), this carries
// the unprojected [positions.size() * n_embd] tensor. Emitted on the SAME condition as the lens
// (only when the adapter's activation tap is on => zero cost on the default path; the 8 scheduler
// goldens are activation-free and untouched). Heavy: a consumer streams it only on demand
// (state="full"); the server omits it from the light frame. Pure observer (invariant 2).
struct StepActivations {
    int t;
    int block;
    std::vector<int> positions;          // board positions for each row (== the active block / act_rows)
    int n_embd = 0;
    std::vector<float> values;           // [positions.size() * n_embd], position-major row r = positions[r]
};

// Latent workspace readout for one token/layer position. `provider` is the concrete adapter id;
// `provider_type` and `readout_kind` are the stable taxonomy fields consumers should branch on.
struct WorkspaceReadout {
    int t;
    std::string run_id;
    int token_index;
    std::string token_text;
    int layer;
    int position;
    std::vector<WorkspaceReadoutItem> top_readouts;
    double entropy;
    std::string provider;
    std::string provider_type = "mock";
    std::string readout_kind = "risk";
};

using Event = std::variant<GenStarted, BlockStarted, TokensCommitted, TokensRevised, StepStats,
                           BlockFinalized, GenFinished, StepFeatures, StepLens, StepActivations,
                           WorkspaceReadout>;

// §5.1 wire form: one JSON object per event, {"t": ..., "type": "...", **payload} — byte-compatible
// with the lab's event_to_dict / to_jsonl_line so logs replay across both runtimes.
std::string to_jsonl_line(const Event& event);

// Versioned optional timing object embedded in GenFinished wire forms.
std::string worker_timing_json(const GenFinished& event);

// Flight-recorder log: one event per line, replayable. Returns false if the file can't be opened.
bool write_jsonl(const std::vector<Event>& events, const std::string& path);

}  // namespace clozn
