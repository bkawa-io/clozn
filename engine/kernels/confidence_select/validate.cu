// validate.cu — run the CUDA confidence_select kernel on logits from a file and dump
// its outputs, so validate.py can diff them against the numpy reference on identical
// inputs. Greedy (temperature=0) + MaxProb + TopK — the deterministic, bit-validatable
// path. Build: nvcc -arch=sm_120 validate.cu confidence_select.cu -o validate.exe
//
// File formats (little-endian):
//   in : int32 N, int32 V, float32 logits[N*V]   (row-major)
//   out: int32 token_ids[N], float32 conf[N], int32 n_selected, int32 selected[n_selected]
// args: validate <in> <out> <conf 0=maxprob|1=margin|2=negentropy> <mode 0=topk|1=threshold>
//       <k_commit> <tau> <min_commit>
#include "confidence_select.cuh"

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
    if (argc < 8) {
        fprintf(stderr, "usage: validate <in> <out> <conf> <mode> <k_commit> <tau> <min_commit>\n");
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    const int conf_kind = atoi(argv[3]);   // 0 maxprob | 1 margin | 2 negentropy
    const int sel_mode = atoi(argv[4]);    // 0 topk | 1 threshold
    const int k_commit = atoi(argv[5]);
    const float tau = (float)atof(argv[6]);
    const int min_commit = atoi(argv[7]);

    FILE* f = fopen(in_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", in_path); return 1; }
    int32_t N = 0, V = 0;
    fread(&N, sizeof(int32_t), 1, f);
    fread(&V, sizeof(int32_t), 1, f);
    std::vector<float> logits((size_t)N * V);
    fread(logits.data(), sizeof(float), (size_t)N * V, f);
    fclose(f);

    float* d_logits = nullptr;
    int32_t* d_tok = nullptr; float* d_conf = nullptr; int32_t* d_sel = nullptr; int32_t* d_nsel = nullptr;
    check(cudaMalloc(&d_logits, sizeof(float) * (size_t)N * V), "malloc logits");
    check(cudaMalloc(&d_tok, sizeof(int32_t) * N), "malloc tok");
    check(cudaMalloc(&d_conf, sizeof(float) * N), "malloc conf");
    check(cudaMalloc(&d_sel, sizeof(int32_t) * N), "malloc sel");
    check(cudaMalloc(&d_nsel, sizeof(int32_t)), "malloc nsel");
    check(cudaMemcpy(d_logits, logits.data(), sizeof(float) * (size_t)N * V, cudaMemcpyHostToDevice), "H2D logits");

    ConfidenceSelectParams p{};
    p.n_masked = N; p.vocab = V; p.temperature = 0.0f; p.top_p = 1.0f;
    p.confidence = static_cast<ConfidenceKind>(conf_kind);
    p.mode = static_cast<SelectMode>(sel_mode);
    p.k_commit = k_commit; p.tau = tau; p.min_commit = min_commit; p.rng_seed = 0;
    ConfidenceSelectOutputs o{d_tok, d_conf, d_sel, d_nsel};

    confidence_select(d_logits, p, o, /*stream=*/nullptr);
    check(cudaDeviceSynchronize(), "sync");
    check(cudaGetLastError(), "kernel");

    std::vector<int32_t> tok(N); std::vector<float> conf(N); std::vector<int32_t> sel(N); int32_t nsel = 0;
    check(cudaMemcpy(tok.data(), d_tok, sizeof(int32_t) * N, cudaMemcpyDeviceToHost), "D2H tok");
    check(cudaMemcpy(conf.data(), d_conf, sizeof(float) * N, cudaMemcpyDeviceToHost), "D2H conf");
    check(cudaMemcpy(&nsel, d_nsel, sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H nsel");
    check(cudaMemcpy(sel.data(), d_sel, sizeof(int32_t) * N, cudaMemcpyDeviceToHost), "D2H sel");

    FILE* g = fopen(out_path, "wb");
    fwrite(tok.data(), sizeof(int32_t), N, g);
    fwrite(conf.data(), sizeof(float), N, g);
    fwrite(&nsel, sizeof(int32_t), 1, g);
    fwrite(sel.data(), sizeof(int32_t), nsel, g);
    fclose(g);

    printf("ran kernel: N=%d V=%d k_commit=%d n_selected=%d\n", N, V, k_commit, nsel);
    return 0;
}
