// clozn_bench.cpp — the §9 quant-calibration sweep: the honest quality column for the
// quantization axis (DESIGN invariant 5). Quantizes an f16 GGUF to several levels in-process
// (llama_model_quantize), runs a fixed prompt set through each under greedy/deterministic
// decoding, and reports per quant: file size, token-AGREEMENT vs the f16 reference, and tok/s.
// Greedy makes the picks deterministic, so divergence is exact token disagreement — no speed
// number without its quality column.
//
//   clozn-bench <model-f16.gguf> [--gpu-layers N] [--max-new N] [--steps T] [--ctx N]
//
// Runs the autoregressive generator (generate_ar); the diffusion generate() this tool used to call
// was removed with the diffusion program (THE_CUT) -- quant calibration now measures the same AR
// models the product actually serves.
#include "clozn/generate_ar.hpp"
#include "clozn/model_ggml.hpp"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace clozn;
namespace fs = std::filesystem;

namespace {

struct QuantSpec { const char* name; llama_ftype ftype; };

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

double mb(const std::string& path) {
    std::error_code ec;
    auto sz = fs::file_size(path, ec);
    return ec ? 0.0 : static_cast<double>(sz) / 1e6;
}

// Run every prompt through one model; return the generated token ids per prompt + tok/s.
struct RunResult {
    std::vector<std::vector<int>> tokens;
    double tok_per_s = 0.0;
};

RunResult run_model(const std::string& path, int mask, int eos, int n_ctx, int gpu_layers,
                    const std::vector<std::string>& prompts, const GenerateConfig& cfg) {
    GgmlAdapter adapter(path, mask, eos, n_ctx, gpu_layers);
    RunResult out;
    int total_tokens = 0;
    double total_ms = 0.0;
    for (const std::string& p : prompts) {
        const std::vector<int> ids = adapter.encode(p);
        const auto t0 = std::chrono::steady_clock::now();
        GenerateResult r = generate_ar(adapter, ids, cfg);
        const auto t1 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_tokens += r.new_tokens;
        out.tokens.push_back(r.generated);
    }
    out.tok_per_s = total_ms > 0 ? total_tokens * 1000.0 / total_ms : 0.0;
    return out;
}

// Token agreement vs the reference: fraction of reference positions whose token matches.
double agreement(const std::vector<std::vector<int>>& ref, const std::vector<std::vector<int>>& q) {
    int matched = 0, total = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        total += static_cast<int>(ref[i].size());
        const size_t n = std::min(ref[i].size(), q[i].size());
        for (size_t j = 0; j < n; ++j)
            if (ref[i][j] == q[i][j]) ++matched;
    }
    return total > 0 ? static_cast<double>(matched) / total : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model-f16.gguf> [--gpu-layers N] [--max-new N] "
                             "[--steps T] [--ctx N]\n", argv[0]);
        return 1;
    }
    const std::string ref_path = argv[1];
    int gpu_layers = 0, n_ctx = 256, mask = 151665, eos = 151643;
    GenerateConfig cfg;
    cfg.max_new = 24; cfg.steps = 12; cfg.block_len = 0; cfg.topk = -1;  // greedy quota
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::atoi(argv[++i]) : 0; };
        if (a == "--gpu-layers") gpu_layers = next();
        else if (a == "--max-new") cfg.max_new = next();
        else if (a == "--steps") cfg.steps = next();
        else if (a == "--ctx") n_ctx = next();
    }

    const std::vector<std::string> prompts = {
        "def add(a, b):", "def is_prime(n):", "def fib(n):",
        "def reverse(s):", "class Stack:", "import numpy as",
    };
    const QuantSpec quants[] = {
        {"Q8_0", LLAMA_FTYPE_MOSTLY_Q8_0},
        {"Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M},
        {"Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M},
        {"Q4_0", LLAMA_FTYPE_MOSTLY_Q4_0},
    };

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    std::fprintf(stderr, "[clozn-bench] %s, %zu prompts, max_new=%d steps=%d (greedy)\n",
                 gpu_layers > 0 ? "GPU" : "CPU", prompts.size(), cfg.max_new, cfg.steps);

    // Reference: the f16 model.
    RunResult ref = run_model(ref_path, mask, eos, n_ctx, gpu_layers, prompts, cfg);

    std::printf("\n  %-8s %9s %14s %10s\n", "quant", "size(MB)", "agree-vs-f16", "tok/s");
    std::printf("  %-8s %9.0f %13s %10.1f\n", "F16", mb(ref_path), "(reference)", ref.tok_per_s);

    const fs::path dir = fs::path(ref_path).parent_path();
    for (const QuantSpec& q : quants) {
        const std::string out_path = (dir / ("clozn-bench-" + std::string(q.name) + ".gguf")).string();
        if (!fs::exists(out_path)) {
            llama_model_quantize_params qp = llama_model_quantize_default_params();
            qp.ftype = q.ftype;
            std::fprintf(stderr, "  quantizing -> %s ...\n", q.name);
            if (llama_model_quantize(ref_path.c_str(), out_path.c_str(), &qp) != 0) {
                std::fprintf(stderr, "  quantize %s FAILED\n", q.name);
                continue;
            }
        }
        RunResult r = run_model(out_path, mask, eos, n_ctx, gpu_layers, prompts, cfg);
        std::printf("  %-8s %9.0f %13.1f%% %10.1f\n", q.name, mb(out_path),
                    agreement(ref.tokens, r.tokens) * 100.0, r.tok_per_s);
    }
    std::printf("\n  agreement = greedy token picks identical to f16, per reference position.\n");
    return 0;
}
