// bench.cu — measure the host-handoff step the confidence-select kernel replaces.
//
// The naive diffusion loop copies the full [n_masked, vocab] logits GPU->host every
// step and samples on the CPU. The kernel keeps the logits on-device, samples +
// selects there, and copies back only 2*n_masked values. This times both, at real
// model scales, on the local GPU. Build: part of CMakeLists (cs_bench target).
//
// HONEST SCOPE: this measures the per-step HOST-HANDOFF in isolation (the transfer +
// the select), NOT end-to-end tok/s — in a real loop the model forward also costs
// time, so the end-to-end share depends on the forward. The data-movement reduction
// (n_masked*vocab -> 2*n_masked floats/ints) is structural and exact regardless.
#include "confidence_select.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace clozn;

int main() {
    struct Dim { int n; int vocab; const char* label; };
    const Dim dims[] = {
        {8, 256, "toy"},
        {16, 32000, "llama-32k"},
        {32, 126464, "LLaDA 8B"},
        {32, 152064, "Dream 7B"},
    };
    const int ITERS = 300, WARMUP = 30;

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    printf("%-10s %8s %8s | %12s %12s | %10s %9s | %8s\n",
           "model", "n_masked", "vocab", "A: D2H full", "B: kernel", "A moves", "B moves", "speedup");
    printf("%s\n", "-----------------------------------------------------------------------------------------------");

    for (const Dim& d : dims) {
        const size_t nlog = (size_t)d.n * d.vocab;
        float* d_logits = nullptr;
        cudaMalloc(&d_logits, nlog * sizeof(float));
        cudaMemset(d_logits, 0, nlog * sizeof(float));  // content irrelevant for timing
        std::vector<float> h_logits(nlog);

        int32_t* d_tok; float* d_conf; int32_t* d_sel; int32_t* d_nsel;
        cudaMalloc(&d_tok, d.n * sizeof(int32_t));
        cudaMalloc(&d_conf, d.n * sizeof(float));
        cudaMalloc(&d_sel, d.n * sizeof(int32_t));
        cudaMalloc(&d_nsel, sizeof(int32_t));
        std::vector<int32_t> h_tok(d.n); std::vector<float> h_conf(d.n);

        ConfidenceSelectParams p{};
        p.n_masked = d.n; p.vocab = d.vocab; p.temperature = 0.0f; p.top_p = 1.0f;
        p.confidence = ConfidenceKind::MaxProb; p.mode = SelectMode::TopK;
        p.k_commit = d.n; p.tau = 0.0f; p.min_commit = 1; p.rng_seed = 0;
        ConfidenceSelectOutputs o{d_tok, d_conf, d_sel, d_nsel};

        // Path A: copy the full logits buffer to host (what the naive loop transfers).
        for (int i = 0; i < WARMUP; ++i)
            cudaMemcpy(h_logits.data(), d_logits, nlog * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; ++i)
            cudaMemcpy(h_logits.data(), d_logits, nlog * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float a_ms = 0.0f; cudaEventElapsedTime(&a_ms, s, e); a_ms /= ITERS;

        // Path B: run the kernel on-device, copy back only the 2*n_masked results.
        for (int i = 0; i < WARMUP; ++i) {
            confidence_select(d_logits, p, o, nullptr);
            cudaMemcpy(h_tok.data(), d_tok, d.n * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_conf.data(), d_conf, d.n * sizeof(float), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        for (int i = 0; i < ITERS; ++i) {
            confidence_select(d_logits, p, o, nullptr);
            cudaMemcpy(h_tok.data(), d_tok, d.n * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_conf.data(), d_conf, d.n * sizeof(float), cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(e); cudaEventSynchronize(e);
        float b_ms = 0.0f; cudaEventElapsedTime(&b_ms, s, e); b_ms /= ITERS;

        const double a_mb = nlog * sizeof(float) / 1048576.0;
        const long b_bytes = (long)d.n * (sizeof(int32_t) + sizeof(float));
        printf("%-10s %8d %8d | %9.3f ms %9.3f ms | %7.1f MB %7ld B | %6.1fx\n",
               d.label, d.n, d.vocab, a_ms, b_ms, a_mb, b_bytes, a_ms / b_ms);

        cudaFree(d_logits); cudaFree(d_tok); cudaFree(d_conf); cudaFree(d_sel); cudaFree(d_nsel);
    }
    return 0;
}
