// validate.cu — run the CUDA sae_topk kernel on pre-activations from a file and dump its
// outputs, so validate.py can diff them against the numpy reference on identical inputs.
// Mirrors ../confidence_select/validate.cu. Build via this dir's CMakeLists (sae_validate).
//
// File formats (little-endian):
//   in : int32 ROWS, int32 NFEAT, float32 pre_acts[ROWS*NFEAT]   (row-major)
//   out: int32 indices[ROWS*K], float32 values[ROWS*K]           (row-major)
// args: validate <in> <out> <k> <relu 0|1>
#include "sae_topk.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace clozn;

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e)); exit(2); }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: validate <in> <out> <k> <relu 0|1>\n");
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    const int k = atoi(argv[3]);
    const bool relu = atoi(argv[4]) != 0;

    FILE* f = fopen(in_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
    int32_t ROWS = 0, NFEAT = 0;
    fread(&ROWS, sizeof(int32_t), 1, f);
    fread(&NFEAT, sizeof(int32_t), 1, f);
    std::vector<float> pre((size_t)ROWS * NFEAT);
    fread(pre.data(), sizeof(float), (size_t)ROWS * NFEAT, f);
    fclose(f);

    float* d_pre = nullptr;
    int32_t* d_idx = nullptr; float* d_val = nullptr;
    check(cudaMalloc(&d_pre, sizeof(float) * (size_t)ROWS * NFEAT), "malloc pre");
    check(cudaMalloc(&d_idx, sizeof(int32_t) * (size_t)ROWS * k), "malloc idx");
    check(cudaMalloc(&d_val, sizeof(float) * (size_t)ROWS * k), "malloc val");
    check(cudaMemcpy(d_pre, pre.data(), sizeof(float) * (size_t)ROWS * NFEAT, cudaMemcpyHostToDevice), "H2D pre");

    SaeTopKParams p{};
    p.rows = ROWS; p.n_features = NFEAT; p.k = k; p.relu = relu;
    SaeTopKOutputs o{d_idx, d_val};

    sae_topk(d_pre, p, o, /*stream=*/nullptr);
    check(cudaDeviceSynchronize(), "sync");
    check(cudaGetLastError(), "kernel");

    std::vector<int32_t> idx((size_t)ROWS * k);
    std::vector<float> val((size_t)ROWS * k);
    check(cudaMemcpy(idx.data(), d_idx, sizeof(int32_t) * (size_t)ROWS * k, cudaMemcpyDeviceToHost), "D2H idx");
    check(cudaMemcpy(val.data(), d_val, sizeof(float) * (size_t)ROWS * k, cudaMemcpyDeviceToHost), "D2H val");

    FILE* g = fopen(out_path, "wb");
    fwrite(idx.data(), sizeof(int32_t), (size_t)ROWS * k, g);
    fwrite(val.data(), sizeof(float), (size_t)ROWS * k, g);
    fclose(g);

    printf("ran sae_topk: rows=%d n_features=%d k=%d relu=%d\n", ROWS, NFEAT, k, relu);
    return 0;
}
