#include "clozn/events.hpp"

#include <cstdio>

namespace clozn {

namespace {

// JSON string escape, ensure_ascii=False (raw UTF-8 passes through, like the lab) — escape only
// the structural/control characters.
std::string jstr(const std::string& s) {
    std::string out = "\"";
    char buf[8];
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) { std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
                else          { out += static_cast<char>(c); }
        }
    }
    out += '"';
    return out;
}

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);  // shortest exact-ish round-trip
    return buf;
}

std::string items_array(const std::vector<CommitItem>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
        char head[64];
        std::snprintf(head, sizeof(head), "%s{\"pos\": %d, \"id\": %d, \"conf\": ",
                      i ? ", " : "", items[i].pos, items[i].id);
        out += head;
        out += num(items[i].conf);
        out += "}";
    }
    out += "]";
    return out;
}

std::string items_array(const std::vector<ReviseItem>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
        char head[80];
        std::snprintf(head, sizeof(head), "%s{\"pos\": %d, \"old\": %d, \"id\": %d, \"conf\": ",
                      i ? ", " : "", items[i].pos, items[i].old, items[i].id);
        out += head;
        out += num(items[i].conf);
        out += "}";
    }
    out += "]";
    return out;
}

// One visitor producing the §5.1 wire form {"t", "type", **payload}. Same schema/keys as the lab's
// event_to_dict (float text may differ; ints/keys/structure match), so consumers parse either.
std::string readouts_array(const std::vector<WorkspaceReadoutItem>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += ", ";
        out += "{\"label\": " + jstr(items[i].label) + ", \"score\": " + num(items[i].score) + "}";
    }
    out += "]";
    return out;
}

struct ToJsonl {
    std::string operator()(const GenStarted& e) const {
        char b[160];
        std::snprintf(b, sizeof(b),
            "{\"t\": %d, \"type\": \"gen_started\", \"prompt_tokens\": %d, \"block_len\": %d, \"max_new\": %d}",
            e.t, e.prompt_tokens, e.block_len, e.max_new);
        return b;
    }
    std::string operator()(const BlockStarted& e) const {
        char b[160];
        std::snprintf(b, sizeof(b),
            "{\"t\": %d, \"type\": \"block_started\", \"block\": %d, \"span\": [%d, %d]}",
            e.t, e.block, e.span.first, e.span.second);
        return b;
    }
    std::string operator()(const TokensCommitted& e) const {
        char b[96];
        std::snprintf(b, sizeof(b), "{\"t\": %d, \"type\": \"tokens_committed\", \"block\": %d, \"items\": ",
                      e.t, e.block);
        return std::string(b) + items_array(e.items) + "}";
    }
    std::string operator()(const TokensRevised& e) const {
        char b[96];
        std::snprintf(b, sizeof(b), "{\"t\": %d, \"type\": \"tokens_revised\", \"block\": %d, \"items\": ",
                      e.t, e.block);
        return std::string(b) + items_array(e.items) + "}";
    }
    std::string operator()(const StepStats& e) const {
        char b[160];
        std::snprintf(b, sizeof(b),
            "{\"t\": %d, \"type\": \"step_stats\", \"block\": %d, \"step\": %d, \"committed\": %d, \"remaining\": %d, \"ms\": ",
            e.t, e.block, e.step, e.committed, e.remaining);
        return std::string(b) + num(e.ms) + ", \"cache_hit\": " + num(e.cache_hit) + "}";
    }
    std::string operator()(const BlockFinalized& e) const {
        char b[96];
        std::snprintf(b, sizeof(b), "{\"t\": %d, \"type\": \"block_finalized\", \"block\": %d, \"text\": ",
                      e.t, e.block);
        char tail[48];
        std::snprintf(tail, sizeof(tail), ", \"steps_used\": %d}", e.steps_used);
        return std::string(b) + jstr(e.text) + tail;
    }
    std::string operator()(const GenFinished& e) const {
        char b[96];
        std::snprintf(b, sizeof(b), "{\"t\": %d, \"type\": \"gen_finished\", \"reason\": ", e.t);
        char mid[64];
        std::snprintf(mid, sizeof(mid), ", \"new_tokens\": %d, \"wall_ms\": ", e.new_tokens);
        char end[64];
        std::snprintf(end, sizeof(end), ", \"steps_total\": %d, \"tok_per_s\": ", e.steps_total);
        return std::string(b) + jstr(e.reason) + mid + num(e.wall_ms) + end + num(e.tok_per_s) + "}";
    }
    std::string operator()(const StepFeatures& e) const {
        std::string out = "{\"t\": " + std::to_string(e.t) + ", \"type\": \"step_features\", \"block\": "
                        + std::to_string(e.block) + ", \"positions\": [";
        for (size_t i = 0; i < e.positions.size(); ++i) { if (i) out += ", "; out += std::to_string(e.positions[i]); }
        out += "], \"features\": [";
        for (size_t i = 0; i < e.features.size(); ++i) { if (i) out += ", "; out += jstr(e.features[i]); }
        out += "], \"scores\": [";
        for (size_t i = 0; i < e.scores.size(); ++i) { if (i) out += ", "; out += num(e.scores[i]); }
        out += "]}";
        return out;
    }
    std::string operator()(const StepLens& e) const {
        std::string out = "{\"t\": " + std::to_string(e.t) + ", \"type\": \"step_lens\", \"block\": "
                        + std::to_string(e.block) + ", \"k\": " + std::to_string(e.k) + ", \"positions\": [";
        for (size_t i = 0; i < e.positions.size(); ++i) { if (i) out += ", "; out += std::to_string(e.positions[i]); }
        out += "], \"ids\": [";
        for (size_t i = 0; i < e.ids.size(); ++i) { if (i) out += ", "; out += std::to_string(e.ids[i]); }
        out += "], \"probs\": [";
        for (size_t i = 0; i < e.probs.size(); ++i) { if (i) out += ", "; out += num(e.probs[i]); }
        out += "]}";
        return out;
    }
    std::string operator()(const StepActivations& e) const {
        // Flight-recorder JSONL form (raw floats). The SSE protocol layer re-encodes this as a
        // base64 {dtype,shape,data} tensor for the wire; here we keep the human-readable array so a
        // replayed log stays self-describing. Heavy — only present when the activation tap is on.
        std::string out = "{\"t\": " + std::to_string(e.t) + ", \"type\": \"step_activations\", \"block\": "
                        + std::to_string(e.block) + ", \"n_embd\": " + std::to_string(e.n_embd) + ", \"positions\": [";
        for (size_t i = 0; i < e.positions.size(); ++i) { if (i) out += ", "; out += std::to_string(e.positions[i]); }
        out += "], \"values\": [";
        for (size_t i = 0; i < e.values.size(); ++i) { if (i) out += ", "; out += num(e.values[i]); }
        out += "]}";
        return out;
    }
    std::string operator()(const WorkspaceReadout& e) const {
        std::string out = "{\"t\": " + std::to_string(e.t) + ", \"type\": \"workspace_readout\", \"run_id\": "
                        + jstr(e.run_id) + ", \"token_index\": " + std::to_string(e.token_index)
                        + ", \"token_text\": " + jstr(e.token_text)
                        + ", \"layer\": " + std::to_string(e.layer)
                        + ", \"position\": " + std::to_string(e.position)
                        + ", \"top_readouts\": " + readouts_array(e.top_readouts)
                        + ", \"entropy\": " + num(e.entropy)
                        + ", \"provider\": " + jstr(e.provider)
                        + ", \"provider_type\": " + jstr(e.provider_type)
                        + ", \"readout_kind\": " + jstr(e.readout_kind) + "}";
        return out;
    }
};

}  // namespace

std::string to_jsonl_line(const Event& event) {
    return std::visit(ToJsonl{}, event);
}

bool write_jsonl(const std::vector<Event>& events, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    for (const Event& e : events) {
        const std::string line = to_jsonl_line(e);
        std::fwrite(line.data(), 1, line.size(), f);
        std::fputc('\n', f);
    }
    std::fclose(f);
    return true;
}

}  // namespace clozn
