// clozn_ar — autoregressive white-box CLI. Decode a prompt left-to-right through any llama.cpp AR
// GGUF (Llama/Qwen/Mistral/...) and print, per generated token, the logit-lens (top-k next-token
// candidates) + the activation read. Proves the AR read harness — GgmlAdapter::ar_forward +
// generate_ar + the shared white-box helpers — end-to-end, the AR counterpart of `clozn`.
//
//   clozn-ar <model.gguf> [--gpu-layers N] [--new N] [--prompt "..."] [--temp T] [--no-features]
#include "clozn/generate_ar.hpp"
#include "clozn/model_ggml.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <variant>
#include <vector>

using namespace clozn;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--gpu-layers N] [--new N] [--prompt \"...\"] "
                             "[--temp T] [--no-features]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    int gpu_layers = 0, n_new = 32;
    double temp = 0.0;
    bool features = true;
    std::string prompt = "The capital of France is";
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--gpu-layers") gpu_layers = std::atoi(nx());
        else if (a == "--new") n_new = std::atoi(nx());
        else if (a == "--prompt") prompt = nx();
        else if (a == "--temp") temp = std::atof(nx());
        else if (a == "--no-features") features = false;
    }

    ggml_log_set([](ggml_log_level lvl, const char* t, void*) {
        if (lvl == GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    auto model = std::make_shared<GgmlModel>(path, -1, -1, gpu_layers);  // AR model: no mask token
    GgmlAdapter ad(model, 2048);
    ad.set_emit_activations(features);

    const std::vector<int> prompt_ids = model->encode(prompt);
    std::printf("model: %s  (n_layer=%d, read tap=%d)\n", path.c_str(), ad.n_layer(), ad.tap_layer());
    std::printf("prompt: %s\n", prompt.c_str());
    std::printf("---- per generated token: 'piece'  lens: cand(prob)...  | read ----\n");

    auto on_event = [&](const Event& e) {
        if (const auto* tc = std::get_if<TokensCommitted>(&e)) {
            for (const auto& it : tc->items)
                std::printf("[%4d] %-16s", it.pos, ("'" + model->decode({it.id}) + "'").c_str());
        } else if (const auto* sl = std::get_if<StepLens>(&e)) {
            std::printf(" lens:");
            for (size_t r = 0; r < sl->positions.size(); ++r)
                for (int j = 0; j < sl->k; ++j) {
                    const int id = sl->ids[r * sl->k + j];
                    const float pr = sl->probs[r * sl->k + j];
                    std::printf(" %s(%.2f)", model->decode({id}).c_str(), pr);
                }
        } else if (const auto* sf = std::get_if<StepFeatures>(&e)) {
            std::printf("  | ");
            const int K = static_cast<int>(sf->features.size());
            for (int k = 0; k < K; ++k)
                std::printf("%s=%.1f ", sf->features[k].c_str(), sf->scores.empty() ? 0.f : sf->scores[k]);
            std::printf("\n");
        }
    };

    GenerateConfig cfg;
    cfg.max_new = n_new;
    cfg.steps = 1;
    SampleConfig sample;
    sample.temperature = temp;
    const GenerateResult r = generate_ar(ad, prompt_ids, cfg, on_event, sample, nullptr);

    std::printf("----\n");
    std::printf("text: %s\n", r.text.c_str());
    std::printf("reason: %s, tokens: %d\n", r.reason.c_str(), r.new_tokens);
    return 0;
}
