// test_sae_encoder.cpp — the engine-vs-torch parity gate for the on-device SAE readout (clozn/sae.hpp).
//
// Two receipts, both against artifacts produced by the research-side torch oracle:
//   1. LOAD: the exported weight dir (tools/export_sae_weights.py) loads, shapes/layer match, and
//      the meta.txt L2-norm receipts verify (a corrupt blob refuses to load).
//   2. ENCODE: on the dumped parity vectors (tools/dump_sae_vectors.py), the CUDA JumpReLU encode +
//      sae_topk must reproduce research/sae7b.py GpuSAE.encode + the numpy top-k reference:
//      the FULL [rows x 131072] gated matrix to fp16-rounding tolerance (both sides round the GEMM
//      result and the bias add to fp16; only accumulation order differs), gate flips (a feature
//      within one rounding step of its learned threshold) counted and bounded, and the top-k sparse
//      code compared as per-row index SETS with values matched by feature id.
//
// SKIPS (returns 0) when the weight dir is absent — CI boxes without the 0.94 GB export stay green,
// like test_ggml_state_write without a GGUF. Runs fully on this repo's RTX 5080 box:
//   test_sae_encoder [sae_dir] [vectors_dir]
//   (defaults: CLOZN_SAE_DIR / CLOZN_SAE_VECTORS, then ~/.clozn/sae/andyrdt_l15[/vectors])
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "clozn/sae.hpp"

using namespace clozn;

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    std::printf("  %-52s : %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

std::string default_dir(const char* env_key, const std::string& fallback_tail) {
    if (const char* v = std::getenv(env_key)) return v;
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    return home ? std::string(home) + fallback_tail : std::string();
}

bool file_exists(const std::string& path) { return std::ifstream(path).good(); }

template <typename T>
std::vector<T> read_bin(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || static_cast<size_t>(f.tellg()) != count * sizeof(T)) return {};
    std::vector<T> v(count);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(T)));
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string sae_dir =
        argc > 1 ? argv[1] : default_dir("CLOZN_SAE_DIR", "/.clozn/sae/andyrdt_l15");
    if (sae_dir.empty() || !file_exists(sae_dir + "/meta.txt")) {
        std::printf("test_sae_encoder: SKIPPED (no exported SAE at '%s'; run "
                    "tools/export_sae_weights.py or set CLOZN_SAE_DIR)\n", sae_dir.c_str());
        return 0;
    }

    std::printf("test_sae_encoder: torch-oracle parity for the on-device SAE readout\n");
    std::printf("[1] load %s\n", sae_dir.c_str());
    SaeEncoder sae;
    const bool loaded = sae.load(sae_dir);
    if (!loaded) std::printf("  load error: %s\n", sae.error().c_str());
    expect(loaded && sae.ready(), "load + norm receipts verify");
    expect(sae.d_in() == 3584 && sae.d_sae() == 131072 && sae.layer() == 15,
           "dims/layer match the andyrdt L15 SAE");
    std::printf("  device_bytes: %.1f MB (weights + workspace)\n", sae.device_bytes() / 1e6);
    if (failures) return 1;

    const std::string vec_dir =
        argc > 2 ? argv[2] : default_dir("CLOZN_SAE_VECTORS", "/.clozn/sae/andyrdt_l15/vectors");
    if (vec_dir.empty() || !file_exists(vec_dir + "/manifest.txt")) {
        std::printf("[2] encode parity: SKIPPED (no vectors at '%s'; run tools/dump_sae_vectors.py)\n"
                    "load-only PASS\n", vec_dir.c_str());
        return 0;
    }

    // Manifest + reference artifacts (see dump_sae_vectors.py for the exact torch ops).
    std::map<std::string, std::string> man;
    {
        std::ifstream mf(vec_dir + "/manifest.txt");
        std::string k, v;
        while (mf >> k >> v) man[k] = v;
    }
    const int rows = std::atoi(man["rows"].c_str());
    const int d_in = std::atoi(man["d_in"].c_str());
    const int d_sae = std::atoi(man["d_sae"].c_str());
    const int k = std::atoi(man["k"].c_str());
    std::printf("[2] encode parity vs %s (rows=%d k=%d)\n", vec_dir.c_str(), rows, k);
    if (rows < 1 || d_in != sae.d_in() || d_sae != sae.d_sae() || k < 1) {
        std::printf("  manifest/encoder mismatch (rows=%d d_in=%d d_sae=%d k=%d)\n", rows, d_in, d_sae, k);
        return 1;
    }
    const auto x = read_bin<float>(vec_dir + "/" + man["x"], static_cast<size_t>(rows) * d_in);
    const auto ref_gated = read_bin<float>(vec_dir + "/" + man["ref_gated"], static_cast<size_t>(rows) * d_sae);
    const auto ref_idx = read_bin<int32_t>(vec_dir + "/" + man["ref_idx"], static_cast<size_t>(rows) * k);
    const auto ref_val = read_bin<float>(vec_dir + "/" + man["ref_val"], static_cast<size_t>(rows) * k);
    if (x.empty() || ref_gated.empty() || ref_idx.empty() || ref_val.empty()) {
        std::printf("  vector blobs missing or wrong size under %s\n", vec_dir.c_str());
        return 1;
    }

    // --- dense receipt: the full gated feature matrix, all rows x 131072 ---
    std::vector<float> gated;
    if (!sae.encode_dense(x.data(), rows, gated)) {
        std::printf("  encode_dense failed: %s\n", sae.error().c_str());
        return 1;
    }
    // fp16 error is RELATIVE (ulp at 13000 is 8.0; real first-token attention-sink rows reach that),
    // so the contract is |d| <= max(0.05, 4e-3 * magnitude) — ~4 fp16 ulps, loose enough for
    // accumulation-order noise, tight enough that a wrong bias/layer/transpose (O(1) relative)
    // fails loudly. Track the worst absolute AND worst relative excess.
    const double kAbsTol = 0.05, kRelTol = 4e-3;
    double max_d = 0.0, sum_d = 0.0, max_rel = 0.0;
    long long out_of_tol = 0;
    long long flips = 0;           // one side zero (gated off), the other live — a threshold flip
    double max_flip_mag = 0.0;     // magnitude of the live side at the worst flip
    for (size_t i = 0; i < gated.size(); ++i) {
        const double r = ref_gated[i], g = gated[i];
        if ((r > 0.0) != (g > 0.0)) {
            ++flips;
            max_flip_mag = std::max(max_flip_mag, std::max(r, g));
            continue;  // flips are counted, not folded into the numeric diff
        }
        const double d = std::fabs(r - g);
        const double mag = std::max(std::fabs(r), std::fabs(g));
        sum_d += d;
        if (d > max_d) max_d = d;
        if (mag > 1.0) max_rel = std::max(max_rel, d / mag);
        if (d > std::max(kAbsTol, kRelTol * mag)) ++out_of_tol;
    }
    std::printf("  gated matrix: max|d|=%.4g max rel=%.3g mean|d|=%.3g over %zu elems; "
                "out-of-tol=%lld gate flips=%lld (worst mag %.3g)\n",
                max_d, max_rel, sum_d / static_cast<double>(gated.size()), gated.size(),
                out_of_tol, flips, max_flip_mag);
    expect(out_of_tol == 0, "gated values within fp16 tolerance (|d| <= max(0.05, 4e-3*mag))");
    expect(flips <= 20, "threshold gate flips rare (<= 20 of ~1M)");
    expect(max_flip_mag <= 4.0, "any flip is a near-threshold feature (mag <= 4.0)");

    // --- sparse receipt: the top-k code the engine will actually emit ---
    std::vector<int32_t> idx;
    std::vector<float> val;
    if (!sae.encode_topk(x.data(), rows, k, idx, val)) {
        std::printf("  encode_topk failed: %s\n", sae.error().c_str());
        return 1;
    }
    double min_overlap = 1.0, sum_overlap = 0.0;
    double max_val_d = 0.0, max_val_rel = 0.0;
    long long val_out_of_tol = 0;
    for (int r = 0; r < rows; ++r) {
        std::map<int32_t, float> ref_live, got_live;  // live = value > 0 (pad slots carry 0)
        for (int c = 0; c < k; ++c) {
            const size_t i = static_cast<size_t>(r) * k + c;
            if (ref_val[i] > 0.0f) ref_live[ref_idx[i]] = ref_val[i];
            if (val[i] > 0.0f) got_live[idx[i]] = val[i];
        }
        int inter = 0;
        for (const auto& kv : ref_live) {
            const auto it = got_live.find(kv.first);
            if (it == got_live.end()) continue;
            ++inter;
            const double d = std::fabs(kv.second - it->second);
            const double mag = std::max(std::fabs(kv.second), std::fabs(it->second));
            max_val_d = std::max(max_val_d, d);
            if (mag > 1.0) max_val_rel = std::max(max_val_rel, d / mag);
            if (d > std::max(kAbsTol, kRelTol * mag)) ++val_out_of_tol;
        }
        const size_t denom = std::max(ref_live.size(), got_live.size());
        const double overlap = denom == 0 ? 1.0 : static_cast<double>(inter) / static_cast<double>(denom);
        min_overlap = std::min(min_overlap, overlap);
        sum_overlap += overlap;
        std::printf("  row %d: live ref=%zu got=%zu inter=%d overlap=%.3f\n",
                    r, ref_live.size(), got_live.size(), inter, overlap);
    }
    std::printf("  top-k: mean overlap=%.4f min=%.4f, matched-value max|d|=%.4g max rel=%.3g\n",
                sum_overlap / rows, min_overlap, max_val_d, max_val_rel);
    expect(sum_overlap / rows >= 0.97, "top-k index sets agree (mean overlap >= 0.97)");
    expect(min_overlap >= 0.9, "worst row still agrees (min overlap >= 0.90)");
    expect(val_out_of_tol == 0, "matched top-k values within fp16 tolerance (rel-aware)");

    // --- perf receipt (workspace already grown; the steady-state per-readout cost) ---
    const auto t0 = std::chrono::steady_clock::now();
    sae.encode_topk(x.data(), rows, k, idx, val);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  steady-state encode_topk(%d rows, k=%d): %.1f ms; device %.1f MB\n",
                rows, k, ms, sae.device_bytes() / 1e6);

    // --- GEMV-only split (encode_dense skips sae_topk entirely) -- isolates the vectorized
    // encoder's own cost from the top-k reduction that follows it, since both are folded
    // together in the encode_topk number above.
    {
        std::vector<float> gated_only;
        const auto td0 = std::chrono::steady_clock::now();
        sae.encode_dense(x.data(), rows, gated_only);
        const double gemv_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
        std::printf("  [split] GEMV-only encode_dense(%d rows): %.1f ms (encode_topk total above: %.1f ms)\n",
                    rows, gemv_ms, ms);
    }

    // --- perf-only receipt at 2x rows (the before/after target size) ---
    // Tiles the same `x` rows to synthesize a 2*rows-row batch. Timing only -- no new expect()s,
    // so this cannot change PASS/FAIL; it exists to track the GEMV vectorization's effect at a
    // second row count without touching dump_sae_vectors.py or the oracle artifacts.
    {
        std::vector<float> x2(x.size() * 2);
        std::copy(x.begin(), x.end(), x2.begin());
        std::copy(x.begin(), x.end(), x2.begin() + static_cast<long>(x.size()));
        std::vector<int32_t> idx2;
        std::vector<float> val2;
        sae.encode_topk(x2.data(), rows * 2, k, idx2, val2);  // warm the workspace at the new size
        const auto t2 = std::chrono::steady_clock::now();
        sae.encode_topk(x2.data(), rows * 2, k, idx2, val2);
        const double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
        std::printf("  steady-state encode_topk(%d rows, k=%d): %.1f ms; device %.1f MB\n",
                    rows * 2, k, ms2, sae.device_bytes() / 1e6);
    }

    if (failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d check(s) FAILED\n", failures);
    return 1;
}
