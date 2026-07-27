// sae_encoder.cu — clozn/sae.hpp implementation: the JumpReLU SAE encoder + top-k readout on-device.
// =============================================================================
// The GEMV + epilogue in front of the validated kernels/sae_topk kernel (see sae.hpp for the
// pipeline diagram and the numerics contract). Everything here is deliberately readout-shaped, not
// decode-hot-loop-shaped: rows is tiny (1 for AR, an active block for diffusion), n_features is
// huge (131072), and the whole call is bounded by streaming W_enc (~0.94 GB fp16) through the SMs
// once per <=8-row tile — ~1-3 ms on an RTX 5080, per readout, on the encoder's own stream.
//
// Numerics: research/sae7b.py GpuSAE.encode is the oracle. torch computes
//     hp = fp16( fp16(x) - b_dec ) @ W_enc  (fp16 GEMM, fp32 accumulate, fp16 result)
//     hp = fp16( hp + b_enc );  gated = relu(hp) * (float(hp) > threshold)
// so this file rounds to fp16 at the SAME two places (GEMV result, bias add) around an fp32
// accumulator, keeping engine-vs-torch differences to accumulation ORDER only — sub-ulp almost
// everywhere once rounded (tests/test_sae_encoder.cpp measures the actual gap on real vectors).

#include "clozn/sae.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "sae_topk.cuh"  // kernels/sae_topk — the validated per-row top-k over the feature dim

namespace clozn {

namespace {

constexpr int kWarpsPerBlock = 4;   // encode: one warp per feature, 4 features per 128-thread block
constexpr int kRowsTile = 8;        // rows per W_enc sweep (fp32 accumulators held in registers)

// xh = fp16(x) - b_dec, elementwise over [rows * d_in]. The fp32->fp16 cast happens BEFORE the
// subtract, exactly like torch's `x.half() - b_dec` (both operands and the arithmetic are fp16).
__global__ void prep_input_kernel(const float* __restrict__ x, const __half* __restrict__ b_dec,
                                  __half* __restrict__ xh, int rows, int d_in) {
    const size_t n = static_cast<size_t>(rows) * d_in;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        xh[i] = __hsub(__float2half(x[i]), b_dec[i % d_in]);
}

// Vector width for the W_enc / xh streaming loads: 8 contiguous __half lanes (16 bytes) per
// thread per step, moved as one float4-shaped transaction and unpacked via 4x __half2 ->
// __half22float2. This is the bandwidth fix: the scalar version below issued one 2-byte load
// per lane per element (measured ~100 GB/s effective on this card, far under its ~900 GB/s
// peak); a 16-byte-per-lane load lets the memory system coalesce a whole warp's request into
// full cache-line-sized transactions instead of 32 scattered halfword fetches. d_in is always a
// clean multiple of 8 for every SAE this loads (128/256/... dictionaries over 3584/4096/...-wide
// residuals), but the loop still tails off scalar for safety if it ever isn't.
constexpr int kVecHalfs = 8;  // 8 * sizeof(__half) == sizeof(float4) == 16 bytes

// One warp per FEATURE: lane-strided fp32 dot of the feature's contiguous W_enc_t row against up
// to kRowsTile activation rows at once (register accumulators; W_enc streams through once per
// tile), warp-shuffle tree reduce, then the JumpReLU epilogue with torch's fp16 roundings:
//   hp = fp16(acc) + b_enc[f] (fp16 add) -> float;  gated = hp > threshold[f] ? max(hp, 0) : 0.
// Accumulation is unordered vs the scalar reference (SIMD-width lanes summed in a different
// sequence) but stays fp32 throughout with the SAME two fp16 rounding points as before -- the
// parity contract only ever claimed accumulation-ORDER independence (sae.hpp's numerics note),
// so this is exactly the noise budget the existing tolerance already covers.
__global__ void encode_jumprelu_kernel(
    const __half* __restrict__ w_enc_t,    // [d_sae, d_in] fp16, feature-major
    const __half* __restrict__ b_enc,      // [d_sae] fp16
    const float* __restrict__ threshold,   // [d_sae] fp32
    const __half* __restrict__ xh,         // [rows, d_in] fp16 (already x - b_dec)
    float* __restrict__ gated,             // [rows, d_sae] fp32 out
    int rows, int d_in, int d_sae) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int f = blockIdx.x * kWarpsPerBlock + warp;
    if (f >= d_sae) return;
    const __half* w = w_enc_t + static_cast<size_t>(f) * d_in;

    const int d_in_vec = d_in / kVecHalfs;         // whole 8-wide chunks
    const int vec_tail = d_in_vec * kVecHalfs;     // first index NOT covered by the vector loop
    const float4* w4 = reinterpret_cast<const float4*>(w);

    for (int r0 = 0; r0 < rows; r0 += kRowsTile) {
        const int rn = rows - r0 < kRowsTile ? rows - r0 : kRowsTile;
        float acc[kRowsTile];
        for (int j = 0; j < kRowsTile; ++j) acc[j] = 0.0f;

        // Vector body: each lane pulls one float4 (8 halfs) of W and, per row, one float4 of xh,
        // then unpacks both as 4x half2 -> float2 and accumulates all 8 products.
        const float4* xh4_row[kRowsTile];
        for (int j = 0; j < rn; ++j)
            xh4_row[j] = reinterpret_cast<const float4*>(xh + static_cast<size_t>(r0 + j) * d_in);
        for (int iv = lane; iv < d_in_vec; iv += 32) {
            const float4 wraw = w4[iv];
            const __half2 w01 = *reinterpret_cast<const __half2*>(&wraw.x);
            const __half2 w23 = *reinterpret_cast<const __half2*>(&wraw.y);
            const __half2 w45 = *reinterpret_cast<const __half2*>(&wraw.z);
            const __half2 w67 = *reinterpret_cast<const __half2*>(&wraw.w);
            const float2 wf01 = __half22float2(w01), wf23 = __half22float2(w23);
            const float2 wf45 = __half22float2(w45), wf67 = __half22float2(w67);
            for (int j = 0; j < rn; ++j) {
                const float4 xraw = xh4_row[j][iv];
                const __half2 x01 = *reinterpret_cast<const __half2*>(&xraw.x);
                const __half2 x23 = *reinterpret_cast<const __half2*>(&xraw.y);
                const __half2 x45 = *reinterpret_cast<const __half2*>(&xraw.z);
                const __half2 x67 = *reinterpret_cast<const __half2*>(&xraw.w);
                const float2 xf01 = __half22float2(x01), xf23 = __half22float2(x23);
                const float2 xf45 = __half22float2(x45), xf67 = __half22float2(x67);
                acc[j] += wf01.x * xf01.x + wf01.y * xf01.y + wf23.x * xf23.x + wf23.y * xf23.y +
                          wf45.x * xf45.x + wf45.y * xf45.y + wf67.x * xf67.x + wf67.y * xf67.y;
            }
        }
        // Scalar tail (only when d_in isn't a multiple of 8 -- never true for a real SAE export,
        // kept for correctness rather than as a perf-relevant path).
        for (int i = vec_tail + lane; i < d_in; i += 32) {
            const float wv = __half2float(w[i]);
            for (int j = 0; j < rn; ++j)
                acc[j] += wv * __half2float(xh[static_cast<size_t>(r0 + j) * d_in + i]);
        }

        for (int off = 16; off > 0; off >>= 1)
            for (int j = 0; j < rn; ++j)
                acc[j] += __shfl_down_sync(0xffffffffu, acc[j], off);
        if (lane == 0) {
            const __half be = b_enc[f];
            const float thr = threshold[f];
            for (int j = 0; j < rn; ++j) {
                const float hp = __half2float(__hadd(__float2half(acc[j]), be));
                gated[static_cast<size_t>(r0 + j) * d_sae + f] =
                    (hp > thr) ? (hp > 0.0f ? hp : 0.0f) : 0.0f;
            }
        }
    }
}

// Read one whole binary blob; empty vector on failure or size mismatch.
std::vector<uint8_t> read_blob(const std::string& path, size_t expect_bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize sz = f.tellg();
    if (sz < 0 || static_cast<size_t>(sz) != expect_bytes) return {};
    std::vector<uint8_t> buf(expect_bytes);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(expect_bytes));
    return f ? buf : std::vector<uint8_t>{};
}

// Host-side L2 norm of a blob (fp16 or fp32 elements), double accumulate — checked against the
// exporter's meta.txt receipt so a truncated or bit-rotted file never loads.
double blob_l2(const std::vector<uint8_t>& blob, bool f16) {
    double s = 0.0;
    if (f16) {
        const __half* p = reinterpret_cast<const __half*>(blob.data());
        const size_t n = blob.size() / 2;
        for (size_t i = 0; i < n; ++i) { const double v = __half2float(p[i]); s += v * v; }
    } else {
        const float* p = reinterpret_cast<const float*>(blob.data());
        const size_t n = blob.size() / 4;
        for (size_t i = 0; i < n; ++i) { const double v = p[i]; s += v * v; }
    }
    return std::sqrt(s);
}

}  // namespace

struct SaeEncoder::Impl {
    // device weights
    __half* d_w_enc_t = nullptr;    // [d_sae * d_in]
    __half* d_b_enc = nullptr;      // [d_sae]
    __half* d_b_dec = nullptr;      // [d_in]
    float* d_threshold = nullptr;   // [d_sae]

    // grow-only per-call workspace (guarded by mu)
    float* d_x = nullptr;           // [rows * d_in] input upload
    __half* d_xh = nullptr;         // [rows * d_in]
    float* d_gated = nullptr;       // [rows * d_sae]
    int32_t* d_idx = nullptr;       // [rows * k]
    float* d_val = nullptr;         // [rows * k]
    char* d_picked = nullptr;       // [rows * d_sae] sae_topk's "already selected" scratch mask --
                                    // hoisted here so encode_topk's hot path
                                    // never pays sae_topk()'s own cudaMalloc/cudaFree + forced sync.
    int ws_rows = 0, ws_k = 0;

    int d_in = 0, d_sae = 0, layer = -1;
    bool ready = false;
    std::string error;
    size_t weight_bytes = 0, ws_bytes = 0;
    cudaStream_t stream = nullptr;
    std::mutex mu;

    bool cuda_ok(cudaError_t e, const char* what) {
        if (e == cudaSuccess) return true;
        error = std::string("CUDA error (") + what + "): " + cudaGetErrorString(e);
        return false;
    }

    // Upload one host blob to a fresh device buffer.
    template <typename T>
    bool upload(const std::vector<uint8_t>& host, T** dev, const char* what) {
        if (!cuda_ok(cudaMalloc(dev, host.size()), what)) return false;
        weight_bytes += host.size();
        return cuda_ok(cudaMemcpy(*dev, host.data(), host.size(), cudaMemcpyHostToDevice), what);
    }

    // Ensure the workspace covers [rows, k] (grow-only; freed with the encoder).
    bool reserve(int rows, int k) {
        if (rows <= ws_rows && k <= ws_k) return true;
        const int nr = rows > ws_rows ? rows : ws_rows;
        const int nk = k > ws_k ? k : ws_k;
        release_ws();
        const size_t rd = static_cast<size_t>(nr) * d_in;
        const size_t rs = static_cast<size_t>(nr) * d_sae;
        const size_t rk = static_cast<size_t>(nr) * nk;
        if (!cuda_ok(cudaMalloc(&d_x, rd * sizeof(float)), "ws x") ||
            !cuda_ok(cudaMalloc(&d_xh, rd * sizeof(__half)), "ws xh") ||
            !cuda_ok(cudaMalloc(&d_gated, rs * sizeof(float)), "ws gated") ||
            !cuda_ok(cudaMalloc(&d_idx, rk * sizeof(int32_t)), "ws idx") ||
            !cuda_ok(cudaMalloc(&d_val, rk * sizeof(float)), "ws val") ||
            !cuda_ok(cudaMalloc(&d_picked, rs * sizeof(char)), "ws picked")) {
            release_ws();
            return false;
        }
        ws_rows = nr; ws_k = nk;
        ws_bytes = rd * (sizeof(float) + sizeof(__half)) + rs * sizeof(float) +
                   rk * (sizeof(int32_t) + sizeof(float)) + rs * sizeof(char);
        return true;
    }

    // Upload x and run prep + encode into d_gated. Caller holds mu.
    bool encode_to_gated(const float* x, int rows) {
        const size_t rd = static_cast<size_t>(rows) * d_in;
        if (!cuda_ok(cudaMemcpyAsync(d_x, x, rd * sizeof(float), cudaMemcpyHostToDevice, stream), "H2D x"))
            return false;
        const int threads = 256;
        const int prep_blocks = static_cast<int>((rd + threads - 1) / threads) < 4096
                                    ? static_cast<int>((rd + threads - 1) / threads) : 4096;
        prep_input_kernel<<<prep_blocks, threads, 0, stream>>>(d_x, d_b_dec, d_xh, rows, d_in);
        const int enc_blocks = (d_sae + kWarpsPerBlock - 1) / kWarpsPerBlock;
        encode_jumprelu_kernel<<<enc_blocks, 32 * kWarpsPerBlock, 0, stream>>>(
            d_w_enc_t, d_b_enc, d_threshold, d_xh, d_gated, rows, d_in, d_sae);
        return cuda_ok(cudaGetLastError(), "encode launch");
    }

    void release_ws() {
        cudaFree(d_x); cudaFree(d_xh); cudaFree(d_gated); cudaFree(d_idx); cudaFree(d_val);
        cudaFree(d_picked);
        d_x = nullptr; d_xh = nullptr; d_gated = nullptr; d_idx = nullptr; d_val = nullptr;
        d_picked = nullptr;
        ws_rows = 0; ws_k = 0; ws_bytes = 0;
    }

    ~Impl() {
        release_ws();
        cudaFree(d_w_enc_t); cudaFree(d_b_enc); cudaFree(d_b_dec); cudaFree(d_threshold);
        if (stream) cudaStreamDestroy(stream);
    }
};

SaeEncoder::SaeEncoder() : impl_(new Impl()) {}
SaeEncoder::~SaeEncoder() = default;

bool SaeEncoder::ready() const { return impl_->ready; }
const std::string& SaeEncoder::error() const { return impl_->error; }
int SaeEncoder::d_in() const { return impl_->d_in; }
int SaeEncoder::d_sae() const { return impl_->d_sae; }
int SaeEncoder::layer() const { return impl_->layer; }
size_t SaeEncoder::device_bytes() const { return impl_->weight_bytes + impl_->ws_bytes; }

bool SaeEncoder::load(const std::string& dir) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    im.ready = false;
    im.error.clear();

    // meta.txt: whitespace-separated `key value` lines (export_sae_weights.py). Parsed with a plain
    // map — no JSON dependency, so clozn_sae links into anything.
    std::ifstream mf(dir + "/meta.txt");
    if (!mf) { im.error = "cannot open " + dir + "/meta.txt"; return false; }
    std::map<std::string, std::string> meta;
    std::string key, value;
    while (mf >> key >> value) meta[key] = value;
    for (const char* need : {"format", "kind", "d_in", "d_sae", "layer",
                             "w_enc_t", "b_enc", "b_dec", "threshold"}) {
        if (!meta.count(need)) { im.error = std::string("meta.txt missing key: ") + need; return false; }
    }
    if (meta["format"] != "clozn-sae-v1") { im.error = "unknown format: " + meta["format"]; return false; }
    if (meta["kind"] != "jumprelu") { im.error = "unsupported SAE kind: " + meta["kind"]; return false; }
    im.d_in = std::atoi(meta["d_in"].c_str());
    im.d_sae = std::atoi(meta["d_sae"].c_str());
    im.layer = std::atoi(meta["layer"].c_str());
    if (im.d_in <= 0 || im.d_sae <= 0) { im.error = "bad dims in meta.txt"; return false; }

    struct BlobSpec { const char* key; size_t bytes; bool f16; };
    const BlobSpec specs[] = {
        {"w_enc_t", static_cast<size_t>(im.d_sae) * im.d_in * 2, true},
        {"b_enc", static_cast<size_t>(im.d_sae) * 2, true},
        {"b_dec", static_cast<size_t>(im.d_in) * 2, true},
        {"threshold", static_cast<size_t>(im.d_sae) * 4, false},
    };
    im.weight_bytes = 0;
    for (const BlobSpec& s : specs) {
        const std::string path = dir + "/" + meta[s.key];
        const std::vector<uint8_t> blob = read_blob(path, s.bytes);
        if (blob.empty()) {
            std::ostringstream os;
            os << "blob " << path << " missing or wrong size (want " << s.bytes << " bytes)";
            im.error = os.str();
            return false;
        }
        // L2-norm receipt: recompute and compare against the exporter's value (rel 1e-3).
        const std::string norm_key = std::string("norm_") + s.key;
        if (meta.count(norm_key)) {
            const double want = std::atof(meta[norm_key].c_str());
            const double got = blob_l2(blob, s.f16);
            if (std::fabs(got - want) > 1e-3 * (std::fabs(want) > 1.0 ? std::fabs(want) : 1.0)) {
                std::ostringstream os;
                os << "norm mismatch for " << s.key << ": file " << got << " vs meta " << want;
                im.error = os.str();
                return false;
            }
        }
        bool ok = false;
        if (std::strcmp(s.key, "w_enc_t") == 0) ok = im.upload(blob, &im.d_w_enc_t, s.key);
        else if (std::strcmp(s.key, "b_enc") == 0) ok = im.upload(blob, &im.d_b_enc, s.key);
        else if (std::strcmp(s.key, "b_dec") == 0) ok = im.upload(blob, &im.d_b_dec, s.key);
        else ok = im.upload(blob, &im.d_threshold, s.key);
        if (!ok) return false;
    }
    if (!im.stream && !im.cuda_ok(cudaStreamCreate(&im.stream), "stream")) return false;
    im.ready = true;
    return true;
}

bool SaeEncoder::encode_topk(const float* x, int rows, int k,
                             std::vector<int32_t>& out_indices, std::vector<float>& out_values) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (!im.ready) { im.error = "encoder not loaded"; return false; }
    if (rows < 1 || k < 1) { im.error = "rows and k must be >= 1"; return false; }
    if (!im.reserve(rows, k) || !im.encode_to_gated(x, rows)) return false;

    SaeTopKParams p{};
    p.rows = rows;
    p.n_features = im.d_sae;
    p.k = k;
    p.relu = true;  // gated values are already >= 0; relu keeps the reference's rank/pad contract
    SaeTopKOutputs o{im.d_idx, im.d_val};
    // Pass the workspace's persistent picked-mask buffer (item 10b): sae_topk no longer
    // cudaMalloc/cudaFree/syncs on our behalf. The kernel launch is stream-ordered, so the
    // blocking cudaMemcpy below (same stream) still correctly waits for it to finish.
    sae_topk(im.d_gated, p, o, im.stream, im.d_picked);

    const size_t rk = static_cast<size_t>(rows) * k;
    out_indices.resize(rk);
    out_values.resize(rk);
    if (!im.cuda_ok(cudaMemcpy(out_indices.data(), im.d_idx, rk * sizeof(int32_t),
                               cudaMemcpyDeviceToHost), "D2H idx") ||
        !im.cuda_ok(cudaMemcpy(out_values.data(), im.d_val, rk * sizeof(float),
                               cudaMemcpyDeviceToHost), "D2H val"))
        return false;
    return true;
}

bool SaeEncoder::encode_dense(const float* x, int rows, std::vector<float>& gated) {
    Impl& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mu);
    if (!im.ready) { im.error = "encoder not loaded"; return false; }
    if (rows < 1) { im.error = "rows must be >= 1"; return false; }
    if (!im.reserve(rows, im.ws_k > 0 ? im.ws_k : 1) || !im.encode_to_gated(x, rows)) return false;

    const size_t rs = static_cast<size_t>(rows) * im.d_sae;
    gated.resize(rs);
    if (!im.cuda_ok(cudaStreamSynchronize(im.stream), "sync") ||
        !im.cuda_ok(cudaMemcpy(gated.data(), im.d_gated, rs * sizeof(float),
                               cudaMemcpyDeviceToHost), "D2H gated"))
        return false;
    return true;
}

}  // namespace clozn
