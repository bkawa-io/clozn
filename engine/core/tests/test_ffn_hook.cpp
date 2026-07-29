// test_ffn_hook.cpp -- the ffn_out-<il> hook (feed-forward/MLP contribution, additive analogue of
// kqv_out) AND the 2026-07-28 head_write/ffn_write honesty fix, proven against a REAL model.
//
// Uses GgmlAdapter::ar_forward_score, the EXACT method POST /score (routes_whitebox.cpp) calls --
// not the generic diffusion-oriented forward(), which applies the Dream-family logit SHIFT
// (src_of(m) = m-1) and made an earlier version of this test show a false "zero effect" for a write
// at the sequence's last position (the shift means there is no output row that ever reads a
// last-position write -- confirmed as a test-harness artifact, not a product bug, by reproducing the
// identical false-zero on the ALREADY-PROVEN residual write_state path under the same conditions).
// ar_forward_score has no shift (AR heads are in-place), matching how /score actually scores.
//
// Two things are gated here:
//   1. ffn_write actually moves the output, and writes at two DIFFERENT layers COMPOSE (both !=
//      either single-site arm) -- the same additive-not-overwrite shape kqv_out/head_write already
//      have, now proven for ffn_out too.
//   2. add_ffn_write is fully fail-closed (returns false for an out-of-range layer or a mismatched
//      values length, BEFORE any forward runs) and head_write_landed()/ffn_write_landed() correctly
//      report false -- never true -- for a spec that could not actually be applied. This is the
//      adapter-level half of the routes_whitebox.cpp fix for the measured bug: a malformed head_write
//      used to report {"head_write_applied": true, "n_head_writes": 1} while writing nothing
//      (model_ggml.cpp's old set_head_writes/eval_cb silently dropped invalid specs with no record).
//
// Uses a CHECK() macro instead of assert(): this repo's build_gpu.bat configures Release, which
// defines NDEBUG and makes plain assert() a silent no-op -- a "regression test" whose checks
// evaporate under the box's own normal build would be worse than none. CHECK always aborts on
// failure, in every build config.
//
// Needs a GGUF (pass as argv[1]); without one it SKIPS (returns 0) so the backend-free CI stays
// green -- same convention as test_ggml_state_write.cpp.
//   build-gpu/test_ffn_hook.exe  <model.gguf>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "clozn/model_ggml.hpp"

using namespace clozn;

#define CHECK(cond) do { \
        if (!(cond)) { \
            std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1); \
        } \
    } while (0)

static double l2diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - double(b[i]); s += d * d; }
    return std::sqrt(s);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("test_ffn_hook: SKIPPED (no GGUF arg; pass a model path to run the real loop)\n");
        return 0;
    }
    const std::string path = argv[1];
    const int mask_tok = 151665;  // open-dCoder <M>; harmless for a forward over real (non-diffusion) tokens

    GgmlAdapter adp(path, mask_tok, /*eos*/ -1, /*n_ctx*/ 512, /*n_gpu_layers*/ 0, /*passthrough*/ false);
    adp.set_causal(true);  // AR scoring path, same mode /score uses

    std::vector<int> board = adp.encode("The capital of France is");
    const int n = static_cast<int>(board.size());
    CHECK(n >= 3);
    const std::vector<int> logits_for = {n - 1};   // predict the token after the whole prompt -- /score's own shape
    const int pos = n - 1;   // the position whose OWN residual produces logits_for[0] (no shift, unlike forward())
    const int n_embd = adp.n_embd();
    const int n_layer = adp.n_layer();
    std::printf("  n=%d n_embd=%d n_layer=%d pos=%d\n", n, n_embd, n_layer, pos);
    CHECK(n_layer >= 6);
    const int L1 = 1;
    const int L2 = n_layer - 2;

    // ================= 1) ffn_write moves the output, and composes across two layers =================
    ForwardResult base = adp.ar_forward_score(board, logits_for);
    CHECK(!base.logits.empty());

    // Capture the real ffn_out row at L1 and L2, then re-inject each as a WRITE at the SAME
    // position, scaled so the perturbation is large and unmistakable (not a no-op self-write).
    adp.set_ffn_capture({L1, L2}, {pos});
    ForwardResult cap = adp.ar_forward_score(board, logits_for);
    (void)cap;
    auto rows = adp.ffn_rows();
    CHECK(rows.count(L1) && rows.at(L1).count(pos));
    CHECK(rows.count(L2) && rows.at(L2).count(pos));
    std::vector<float> row_L1 = rows.at(L1).at(pos);
    std::vector<float> row_L2 = rows.at(L2).at(pos);
    CHECK(static_cast<int>(row_L1.size()) == n_embd);
    CHECK(static_cast<int>(row_L2.size()) == n_embd);
    double norm_L1 = 0.0, norm_L2 = 0.0;
    for (float x : row_L1) norm_L1 += double(x) * x;
    for (float x : row_L2) norm_L2 += double(x) * x;
    std::printf("  captured ffn_out row norms: L1=%.6f L2=%.6f\n", std::sqrt(norm_L1), std::sqrt(norm_L2));
    for (float& x : row_L1) x = x * 1.7f + 0.05f;
    for (float& x : row_L2) x = x * 1.7f - 0.05f;
    adp.clear_ffn_capture();

    auto score_with_writes = [&](bool use_l1, bool use_l2) -> std::vector<float> {
        adp.clear_ffn_write();
        if (use_l1) { const bool ok = adp.add_ffn_write(L1, {pos}, row_L1); CHECK(ok); }
        if (use_l2) { const bool ok = adp.add_ffn_write(L2, {pos}, row_L2); CHECK(ok); }
        ForwardResult r = adp.ar_forward_score(board, logits_for);
        // every requested write must have LANDED (this model's architecture does name "ffn_out" --
        // qwen2/llama/gemma4/qwen35/deepseek2(mistral4)/dream all verified by direct file read).
        for (bool ok : adp.ffn_write_landed()) CHECK(ok);
        adp.clear_ffn_write();
        return r.logits;
    };

    std::vector<float> arm_L1 = score_with_writes(true, false);
    std::vector<float> arm_L2 = score_with_writes(false, true);
    std::vector<float> arm_both = score_with_writes(true, true);

    const double d_L1_base = l2diff(arm_L1, base.logits);
    const double d_L2_base = l2diff(arm_L2, base.logits);
    const double d_both_L1 = l2diff(arm_both, arm_L1);
    const double d_both_L2 = l2diff(arm_both, arm_L2);
    std::printf("  ffn_write L1-only moved logits by L2 = %.6f (from baseline)\n", d_L1_base);
    std::printf("  ffn_write L2-only moved logits by L2 = %.6f (from baseline)\n", d_L2_base);
    std::printf("  ffn_write both vs L1-only  L2 = %.6f\n", d_both_L1);
    std::printf("  ffn_write both vs L2-only  L2 = %.6f\n", d_both_L2);
    CHECK(d_L1_base > 1e-4);   // each single-site write has SOME effect
    CHECK(d_L2_base > 1e-4);
    CHECK(d_both_L1 > 1e-4);  // COMPOSITION: "both" differs from either single-site arm
    CHECK(d_both_L2 > 1e-4);

    // ================= 2) fail-closed: add_ffn_write rejects a bad spec BEFORE any forward =================
    adp.clear_ffn_write();
    CHECK(adp.add_ffn_write(n_layer + 50, {pos}, row_L1) == false);        // layer out of range
    CHECK(adp.add_ffn_write(-1, {pos}, row_L1) == false);                  // layer negative
    std::vector<float> short_values(row_L1.begin(), row_L1.end() - 1);
    CHECK(adp.add_ffn_write(L1, {pos}, short_values) == false);            // values.size() mismatch
    CHECK(adp.add_ffn_write(L1, {pos}, row_L1) == true);                   // sanity: a GOOD spec still works
    adp.clear_ffn_write();
    std::printf("  add_ffn_write correctly rejects out-of-range layer and mismatched values.size(), "
                "and still accepts a well-formed spec\n");

    // ================= 3) honesty: head_write_landed() is false for the EXACT measured bug repro =====
    // Re-derive the SAME "malformed values length" and "out-of-range layer" repro cases the spike
    // (scripts/spike/additive_writes_probe.py) measured as a silent no-op that still reported
    // head_write_applied:true. GgmlAdapter itself (not the HTTP route) must now report the truth via
    // head_write_landed(), never a stale "true" left over from a route-level assumption.
    adp.set_head_capture({L1}, {pos}, /*rows=*/true);
    ForwardResult hcap = adp.ar_forward_score(board, logits_for);
    (void)hcap;
    auto head_dims = adp.head_dims();
    auto head_rows = adp.head_rows();
    adp.clear_head_capture();
    const int d_head = head_dims.count("d_head") ? head_dims.at("d_head") : 0;
    const int n_head = head_dims.count("n_head") ? head_dims.at("n_head") : 0;
    std::printf("  head_dims: d_head=%d n_head=%d\n", d_head, n_head);
    CHECK(d_head > 0 && n_head > 0);
    CHECK(head_rows.count(L1) && head_rows.at(L1).count(pos));
    std::vector<float> full_row = head_rows.at(L1).at(pos);
    std::vector<float> slice(full_row.begin(), full_row.begin() + d_head);

    // 3a) malformed values length: d_head-1 floats instead of d_head.
    {
        GgmlAdapter::HeadWrite bad;
        bad.layer = L1; bad.head = 0; bad.positions = {pos};
        bad.values = std::vector<float>(slice.begin(), slice.end() - 1);
        adp.set_head_writes({bad});
        ForwardResult r = adp.ar_forward_score(board, logits_for);
        (void)r;
        const auto& landed = adp.head_write_landed();
        CHECK(landed.size() == 1);
        CHECK(landed[0] == false);   // MUST be false -- this is the exact measured-bug repro
        adp.clear_head_writes();
        std::printf("  head_write_landed() == false for values.size() == d_head-1 (matches the "
                    "measured bug repro; used to silently report applied:true)\n");
    }

    // 3b) out-of-range layer: n_layer+50.
    {
        GgmlAdapter::HeadWrite bad;
        bad.layer = n_layer + 50; bad.head = 0; bad.positions = {pos}; bad.values = slice;
        adp.set_head_writes({bad});
        ForwardResult r = adp.ar_forward_score(board, logits_for);
        (void)r;
        const auto& landed = adp.head_write_landed();
        CHECK(landed.size() == 1);
        CHECK(landed[0] == false);   // MUST be false -- the second measured-bug repro
        adp.clear_head_writes();
        std::printf("  head_write_landed() == false for layer == n_layer+50 (matches the measured "
                    "bug repro)\n");
    }

    // 3c) control: a WELL-FORMED head_write DOES land (landed_ isn't just permanently false), and
    // moves the output -- the same read->edit->write->observe bar test_ggml_state_write.cpp holds
    // residual write_state to.
    {
        GgmlAdapter::HeadWrite good;
        good.layer = L1; good.head = 0; good.positions = {pos}; good.values = slice;
        for (float& v : good.values) v = v * 1.7f + 0.05f;
        adp.set_head_writes({good});
        ForwardResult r = adp.ar_forward_score(board, logits_for);
        const auto& landed = adp.head_write_landed();
        CHECK(landed.size() == 1);
        CHECK(landed[0] == true);
        const double moved = l2diff(base.logits, r.logits);
        std::printf("  head_write_landed() == true for a well-formed spec, and it moved logits by "
                    "L2 = %.6f (the fix does not just always report false)\n", moved);
        CHECK(moved > 1e-4);
        adp.clear_head_writes();
    }

    // ================= 4) cross-hook composition: ffn_write (L1) + head_write (L2, different layer)
    // both being additive, they must COMPOSE exactly like two ffn_write layers did above.
    {
        adp.set_head_capture({L2}, {pos}, /*rows=*/true);
        ForwardResult hc = adp.ar_forward_score(board, logits_for);
        (void)hc;
        auto hd = adp.head_dims();
        auto hr = adp.head_rows();
        adp.clear_head_capture();
        const int dh2 = hd.count("d_head") ? hd.at("d_head") : 0;
        CHECK(dh2 > 0);
        std::vector<float> hslice(hr.at(L2).at(pos).begin(), hr.at(L2).at(pos).begin() + dh2);
        for (float& v : hslice) v = v * 1.7f - 0.05f;

        GgmlAdapter::HeadWrite hw; hw.layer = L2; hw.head = 0; hw.positions = {pos}; hw.values = hslice;

        adp.clear_ffn_write(); adp.clear_head_writes();
        CHECK(adp.add_ffn_write(L1, {pos}, row_L1));
        ForwardResult ffn_only = adp.ar_forward_score(board, logits_for);
        adp.clear_ffn_write();

        adp.set_head_writes({hw});
        ForwardResult head_only = adp.ar_forward_score(board, logits_for);
        adp.clear_head_writes();

        CHECK(adp.add_ffn_write(L1, {pos}, row_L1));
        adp.set_head_writes({hw});
        ForwardResult both = adp.ar_forward_score(board, logits_for);
        CHECK(adp.ffn_write_landed()[0]);
        CHECK(adp.head_write_landed()[0]);
        adp.clear_ffn_write(); adp.clear_head_writes();

        const double d_both_ffn = l2diff(both.logits, ffn_only.logits);
        const double d_both_head = l2diff(both.logits, head_only.logits);
        std::printf("  cross-hook: ffn_write(L1) + head_write(L2) both vs ffn-only  L2 = %.6f\n", d_both_ffn);
        std::printf("  cross-hook: ffn_write(L1) + head_write(L2) both vs head-only L2 = %.6f\n", d_both_head);
        CHECK(d_both_ffn > 1e-4);
        CHECK(d_both_head > 1e-4);
    }

    std::printf("test_ffn_hook: OK (ffn_write composes across layers and with head_write; "
                "add_ffn_write/head_write_landed are honest about what actually landed)\n");
    return 0;
}
