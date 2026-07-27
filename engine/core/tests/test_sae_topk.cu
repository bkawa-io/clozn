// test_sae_topk.cu — parity gate for the SAE-sparsify top-k kernel (ROADMAP 3.3): the CUDA
// sae_topk must produce the SAME per-row sparse code as a straightforward in-process CPU
// reference on identical pre-activations. Selected feature indices must match EXACTLY (per
// row, ascending; tie -> lower index); values within float32 epsilon. Runs on the GPU; build
// with -DCLOZN_BUILD_CUDA=ON. (kernels/sae_topk/validate.py independently checks the kernel
// against the numpy oracle; this proves it drops into the engine build with identical picks.)
#include "sae_topk.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

using namespace clozn;

namespace {

int failures = 0;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("CUDA error (%s): %s\n", what, cudaGetErrorString(e)); ++failures; }
}

// Deterministic pre-activations: a structured value per (row, feature) so the top-k order is
// unambiguous (and reproducible host vs device). Mix of positive and negative to exercise ReLU.
std::vector<float> make_pre(int rows, int n_features) {
    std::vector<float> a(static_cast<size_t>(rows) * n_features);
    for (int r = 0; r < rows; ++r)
        for (int f = 0; f < n_features; ++f) {
            // A smooth, row-shifted ramp with a sign flip; distinct values avoid accidental ties
            // except where we want them, so the "exact index" check is meaningful.
            float v = std::sin(0.001f * ((r * 131 + f * 7) % 9973)) * 4.0f - 0.3f;
            a[static_cast<size_t>(r) * n_features + f] = v;
        }
    return a;
}

// CPU reference: per row, top-k over max(v,0) when relu (else raw v), tie -> LOWER index,
// indices emitted ASCENDING, values aligned (gated to 0 when relu and the pick is <= 0).
// Mirrors kernels/sae_topk/reference.py exactly.
void cpu_sae_topk(const std::vector<float>& pre, int rows, int n_features, int k, bool relu,
                  std::vector<int>& out_idx, std::vector<float>& out_val) {
    const int k_eff = std::min(k, n_features);
    out_idx.assign(static_cast<size_t>(rows) * k, 0);
    out_val.assign(static_cast<size_t>(rows) * k, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float* row = pre.data() + static_cast<size_t>(r) * n_features;
        // Stable sort feature indices by descending ranked value; ties keep lower index.
        std::vector<int> order(n_features);
        for (int f = 0; f < n_features; ++f) order[f] = f;
        auto ranked = [&](int f) { float v = row[f]; return (relu && v < 0.0f) ? 0.0f : v; };
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return ranked(a) > ranked(b); });
        std::vector<int> top(order.begin(), order.begin() + k_eff);
        std::sort(top.begin(), top.end());  // ascending
        for (int c = 0; c < k_eff; ++c) {
            out_idx[static_cast<size_t>(r) * k + c] = top[c];
            out_val[static_cast<size_t>(r) * k + c] = ranked(top[c]);
        }
        for (int c = k_eff; c < k; ++c) {
            out_idx[static_cast<size_t>(r) * k + c] = k_eff > 0 ? top[k_eff - 1] : 0;
            out_val[static_cast<size_t>(r) * k + c] = 0.0f;
        }
    }
}

void check_case(const char* name, int rows, int n_features, int k, bool relu) {
    std::vector<float> pre = make_pre(rows, n_features);

    std::vector<int> cpu_idx; std::vector<float> cpu_val;
    cpu_sae_topk(pre, rows, n_features, k, relu, cpu_idx, cpu_val);

    float* d_pre = nullptr; int32_t* d_idx = nullptr; float* d_val = nullptr;
    check(cudaMalloc(&d_pre, sizeof(float) * pre.size()), "malloc pre");
    check(cudaMalloc(&d_idx, sizeof(int32_t) * (size_t)rows * k), "malloc idx");
    check(cudaMalloc(&d_val, sizeof(float) * (size_t)rows * k), "malloc val");
    check(cudaMemcpy(d_pre, pre.data(), sizeof(float) * pre.size(), cudaMemcpyHostToDevice), "H2D");

    SaeTopKParams p{}; p.rows = rows; p.n_features = n_features; p.k = k; p.relu = relu;
    SaeTopKOutputs o{d_idx, d_val};
    sae_topk(d_pre, p, o, /*stream=*/nullptr);
    check(cudaDeviceSynchronize(), "sync");
    check(cudaGetLastError(), "kernel");

    std::vector<int32_t> gpu_idx((size_t)rows * k); std::vector<float> gpu_val((size_t)rows * k);
    check(cudaMemcpy(gpu_idx.data(), d_idx, sizeof(int32_t) * gpu_idx.size(), cudaMemcpyDeviceToHost), "D2H idx");
    check(cudaMemcpy(gpu_val.data(), d_val, sizeof(float) * gpu_val.size(), cudaMemcpyDeviceToHost), "D2H val");

    const int k_eff = std::min(k, n_features);
    bool ok = true;
    float max_dval = 0.0f;
    for (int r = 0; r < rows && ok; ++r)
        for (int c = 0; c < k_eff; ++c) {
            const size_t i = static_cast<size_t>(r) * k + c;
            if (cpu_idx[i] != gpu_idx[i]) { ok = false; break; }
            max_dval = std::max(max_dval, std::fabs(cpu_val[i] - gpu_val[i]));
        }
    if (max_dval > 1e-4f) ok = false;

    std::printf("  %-22s rows=%d nF=%d k=%d relu=%d -> idx %s  val|d|=%.1e : %s\n",
                name, rows, n_features, k, (int)relu, ok ? "ok" : "X", max_dval,
                ok ? "PASS" : "MISMATCH");
    if (!ok) {
        ++failures;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < k_eff; ++c) {
                const size_t i = static_cast<size_t>(r) * k + c;
                if (cpu_idx[i] != gpu_idx[i])
                    std::printf("    row %d col %d: cpu idx=%d  gpu idx=%d\n", r, c,
                                cpu_idx[i], gpu_idx[i]);
            }
        }
    }

    cudaFree(d_pre); cudaFree(d_idx); cudaFree(d_val);
}

}  // namespace

int main() {
    std::printf("test_sae_topk: CPU reference vs CUDA sae_topk parity (per-row top-k over features)\n");
    check_case("relu k=8  nF=512",   6,   512, 8,  true);
    check_case("relu k=16 nF=4096",  8,  4096, 16, true);
    check_case("relu k=32 nF=16384", 4, 16384, 32, true);
    check_case("relu k=1  nF=2000",  6,  2000, 1,  true);
    check_case("signed k=8 nF=512",  6,   512, 8,  false);
    check_case("k>nF clamp+pad",     4,     8, 20, true);

    if (failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d case(s) FAILED\n", failures);
    return 1;
}
