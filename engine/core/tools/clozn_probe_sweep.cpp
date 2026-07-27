// clozn_probe_sweep — sweep all layers to find the best tap for concept probes.
// For each layer, calibrates diff-in-means probes and scores them by mean inter-class
// separation (higher = concepts are more distinguishable at that layer).
//
// usage: clozn-probe-sweep <model.gguf> [--gpu-layers N] [--mask-token ID]
#include "clozn/model_ggml.hpp"
#include "clozn/probe.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace clozn;

static const char* token_category(const std::string& piece) {
    std::string s, low;
    for (unsigned char c : piece)
        if (!std::isspace(c)) { s.push_back(static_cast<char>(c)); low.push_back(static_cast<char>(std::tolower(c))); }
    if (s.empty()) return nullptr;
    bool any_digit = false, any_alnum = false, all_alpha = true;
    for (unsigned char c : s) {
        if (std::isdigit(c)) any_digit = true;
        if (std::isalnum(c)) any_alnum = true;
        if (!std::isalpha(c)) all_alpha = false;
    }
    if (any_digit) return "number";
    if (!any_alnum) return "punct";
    if (all_alpha) {
        static const std::set<std::string> kFunction = {
            "the","a","an","of","and","to","in","is","it","that","this","for","on","with","as","at",
            "by","or","but","not","are","was","were","be","been","being","i","you","he","she","they",
            "we","his","her","its","their","our","my","your","from","up","out","if","then","than","so",
            "no","do","does","did","have","has","had","will","would","can","could","should","may","me"
        };
        return kFunction.count(low) ? "function" : "content";
    }
    return nullptr;
}

static const std::vector<std::string> corpus = {
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "In 2024 the company sold 15 boats, 320 bikes, and 7 cars to 4 buyers.",
    "She said, \"Hello!\" Then he asked: why now? Nobody really knew.",
    "Pi is about 3.14159 and the speed of light is 299792458 meters per second.",
    "Prices fell 12 percent in March, rose 8 percent in April, and held in May.",
    "Wait, stop -- listen carefully; the answer matters more than the question.",
};
static const std::vector<std::string> cats = {"punct", "number", "function", "content"};

struct LayerScore {
    int layer;
    double separation;
    int n_concepts;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: clozn-probe-sweep <model.gguf> [--gpu-layers N] [--mask-token ID]\n");
        return 1;
    }
    const char* model_path = argv[1];
    int gpu_layers = 0;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) gpu_layers = std::atoi(argv[++i]);
    }

    ggml_log_set([](ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
    }, nullptr);

    auto model = std::make_shared<GgmlModel>(model_path, -1, -1, gpu_layers);
    GgmlAdapter ad(model, 512);
    ad.set_emit_activations(true);

    const int n_layer = ad.n_layer();
    std::printf("Model: %s, n_layer=%d, n_embd=%d\n", model_path, n_layer, llama_model_n_embd(model->handle()));
    std::printf("Sweeping layers 1..%d (0 = final-layer fallback)\n\n", n_layer - 1);

    std::vector<LayerScore> results;

    auto sweep_layer = [&](int il) {
        ad.set_tap_layer(il);

        int n_embd = 0;
        long long N = 0;
        std::vector<double> sum, sumsq;
        std::vector<std::vector<double>> cat_sum(cats.size());
        std::vector<long long> cat_cnt(cats.size(), 0);

        for (const std::string& text : corpus) {
            const std::vector<int> toks = model->encode(text);
            if (toks.size() < 2) continue;
            const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
            const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
            if (fwd.activations.empty() || fwd.n_embd <= 0) continue;
            if (n_embd == 0) {
                n_embd = fwd.n_embd;
                sum.assign(n_embd, 0.0);
                sumsq.assign(n_embd, 0.0);
                for (auto& cs : cat_sum) cs.assign(n_embd, 0.0);
            }
            for (size_t r = 0; r < fwd.act_rows.size(); ++r) {
                const float* h = fwd.activations.data() + r * static_cast<size_t>(n_embd);
                for (int i = 0; i < n_embd; ++i) { sum[i] += h[i]; sumsq[i] += static_cast<double>(h[i]) * h[i]; }
                ++N;
                const char* c = token_category(model->decode({toks[fwd.act_rows[r]]}));
                if (!c) continue;
                for (size_t ci = 0; ci < cats.size(); ++ci)
                    if (cats[ci] == c) {
                        for (int i = 0; i < n_embd; ++i) cat_sum[ci][i] += h[i];
                        ++cat_cnt[ci];
                        break;
                    }
            }
        }

        if (n_embd == 0 || N == 0) return;

        // Build standardization
        std::vector<float> mean(n_embd), inv_std(n_embd);
        for (int i = 0; i < n_embd; ++i) {
            double mu = sum[i] / static_cast<double>(N);
            double var = sumsq[i] / static_cast<double>(N) - mu * mu;
            mean[i] = static_cast<float>(mu);
            inv_std[i] = static_cast<float>(1.0 / std::sqrt(var > 1e-12 ? var : 1e-12));
        }

        // Compute separation: for each category, the L2 norm of its diff-in-means direction
        // (before unit normalization) measures how far apart it is from the rest.
        double total_sep = 0.0;
        int n_valid = 0;
        for (size_t ci = 0; ci < cats.size(); ++ci) {
            if (cat_cnt[ci] < 3) continue;
            long long neg_cnt = 0;
            std::vector<double> neg_sum(n_embd, 0.0);
            for (size_t cj = 0; cj < cats.size(); ++cj)
                if (cj != ci && cat_cnt[cj] > 0) {
                    neg_cnt += cat_cnt[cj];
                    for (int i = 0; i < n_embd; ++i) neg_sum[i] += cat_sum[cj][i];
                }
            if (neg_cnt == 0) continue;
            double norm2 = 0.0;
            for (int i = 0; i < n_embd; ++i) {
                double pos_std = (cat_sum[ci][i] / static_cast<double>(cat_cnt[ci]) - mean[i]) * inv_std[i];
                double neg_std = (neg_sum[i] / static_cast<double>(neg_cnt) - mean[i]) * inv_std[i];
                double d = pos_std - neg_std;
                norm2 += d * d;
            }
            total_sep += std::sqrt(norm2);
            ++n_valid;
        }

        double avg_sep = n_valid > 0 ? total_sep / n_valid : 0.0;
        results.push_back({il, avg_sep, n_valid});
        std::printf("  layer %2d: separation = %7.2f  (%d concepts)\n", il, avg_sep, n_valid);
    };

    // Layer 0 = final (via embeddings fallback)
    sweep_layer(0);
    // Layers 1..n_layer-1
    for (int il = 1; il < n_layer; ++il) sweep_layer(il);

    // Find best
    if (!results.empty()) {
        auto best = std::max_element(results.begin(), results.end(),
            [](const LayerScore& a, const LayerScore& b) { return a.separation < b.separation; });
        std::printf("\nBest layer: %d (separation = %.2f)\n", best->layer, best->separation);
        std::printf("Current default: %d (2/3 depth)\n", n_layer * 2 / 3);
        auto def = std::find_if(results.begin(), results.end(),
            [&](const LayerScore& s) { return s.layer == n_layer * 2 / 3; });
        if (def != results.end())
            std::printf("Default separation: %.2f (%.1f%% of best)\n",
                        def->separation, 100.0 * def->separation / best->separation);
    }

    return 0;
}
