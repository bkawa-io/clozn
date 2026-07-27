// serve/readout_plane.hpp — the Phase 2.3 multi-observer readout plane.
//
// One capture, N observers: the adapter's capture set snapshots several layers' residuals in a
// SINGLE forward (one D2H per layer per token — ~n_embd*4 bytes each, negligible next to the
// decode itself), and this plane fans the frame out to every requested observer — J-lens
// "disposed to say" top-k, per-layer residual norms, concept-probe projections — on its OWN
// worker thread. The decode thread only queues frames; it never runs observer math. That is the
// structural change vs the old jlens_live path, where the J-lens GEMV ran inside on_event ON the
// decode thread, stalling the next llama_decode (and the tap was single-owner: SAE or lens,
// never both).
//
// Honesty contract ("no silent caps"): a bounded queue protects decode latency when observers
// are slower than generation — but every dropped or skipped frame is COUNTED and reported in the
// final readout_stats frame. A consumer can always tell exactly which tokens were observed.
//
// Threading: push() is called by the decode thread (single producer). drain() and finish() are
// called by the SSE emitter thread — the same thread that owns the sink, so ALL stream writes
// stay single-threaded (cpp-httplib's DataSink is not thread-safe). The worker thread only
// computes and parks results in ready_. JlensServe::readout is internally mutex-serialized and
// GgmlModel::decode is a pure vocab lookup, so both are safe off-thread.
#pragma once

#include "server_shared.hpp"

#include <chrono>
#include <deque>

namespace clozn {

struct ReadoutObserverConfig {
    std::vector<int> layers;     // capture set (validated + defaulted by the route)
    bool jlens = true;           // per-layer J-lens top-k (skipped per-layer if no sidecar)
    bool norms = true;           // per-row |residual| per layer
    bool probes = false;         // concept-probe projection (at probe_layer only)
    int probe_layer = 0;         // layer the concept probes are calibrated at
    int topk = 8;                // J-lens top-k per position
    int every = 1;               // observe every Nth generated token (1 = all)
    bool include_prompt = false; // also observe the prefill frame (rows = prompt length)
};

class ReadoutPlane {
public:
    ReadoutPlane(const ReadoutObserverConfig& cfg, JlensServe* lens,
                 const ConceptProbes* probes, const GgmlModel* model)
        : cfg_(cfg), lens_(lens), probes_(probes), model_(model) {
        worker_ = std::thread([this] { loop(); });
    }
    ~ReadoutPlane() { stop(); }
    ReadoutPlane(const ReadoutPlane&) = delete;
    ReadoutPlane& operator=(const ReadoutPlane&) = delete;

    // Decode thread: hand a frame over. NEVER blocks — a full queue drops the frame (counted).
    void push(CaptureFrame&& f) {
        if (f.rows > 1 && !cfg_.include_prompt) { ++skipped_prompt_; return; }  // the prefill frame
        if (f.rows == 1 && (seen_gen_++ % cfg_.every) != 0) { ++skipped_every_; return; }
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (queue_.size() >= kQueueCap) { ++dropped_; return; }
            queue_.push_back(std::move(f));
        }
        cv_.notify_one();
    }

    // Emitter thread: completed readout frames ready to stream (non-blocking; may be empty).
    std::vector<json> drain() {
        std::vector<json> out;
        std::lock_guard<std::mutex> lk(mtx_);
        out.swap(ready_);
        return out;
    }

    // Emitter thread, after generation ends: wait for the worker to finish the tail of the queue,
    // then return the remaining frames + the honest stats frame. The queue is bounded (kQueueCap),
    // so this wait is bounded too.
    std::vector<json> finish() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            flushing_ = true;         // worker: stop accumulating, drain what's queued now
            cv_.notify_all();
            done_cv_.wait(lk, [this] { return queue_.empty() && !busy_; });
        }
        stop();
        std::vector<json> out;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            out.swap(ready_);
        }
        out.push_back(json{{"type", "readout_stats"},
                           {"observed", observed_},
                           {"dropped_queue_full", dropped_},
                           {"skipped_by_every", skipped_every_},
                           {"skipped_prompt_frames", skipped_prompt_},
                           {"queue_cap", static_cast<int>(kQueueCap)}});
        return out;
    }

private:
    static constexpr size_t kQueueCap = 32;
    static constexpr int kBatchMax = 16;      // tokens per readout graph (amortizes head read + syncs)
    static constexpr int kBatchWaitMs = 140;  // max accumulation lag before observing what's queued

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (stop_) return;
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    // Worker: pop a BATCH of frames per pass. One readout graph covers the whole batch, so the
    // head weight read AND the per-graph H2D/compute/D2H syncs amortize across kBatchMax tokens
    // (per-token graphs measured ~7-9 ms each on Windows — mostly WDDM sync + a full head read
    // per token; batched, both divide by the batch size). The accumulation wait (kBatchWaitMs)
    // trades a bounded readout lag for that amortization; finish() flushes immediately.
    void loop() {
        for (;;) {
            std::vector<CaptureFrame> batch;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) { if (stop_) return; continue; }
                if (!stop_ && !flushing_)  // let a few more tokens land; bounded by the wait
                    cv_.wait_for(lk, std::chrono::milliseconds(kBatchWaitMs),
                                 [this] { return stop_ || flushing_ ||
                                                 queue_.size() >= static_cast<size_t>(kBatchMax); });
                while (!queue_.empty() && static_cast<int>(batch.size()) < kBatchMax) {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                }
                busy_ = true;
            }
            std::vector<json> frames = observe_batch(batch);
            {
                std::lock_guard<std::mutex> lk(mtx_);
                for (auto& fr : frames) ready_.push_back(std::move(fr));
                busy_ = false;
                observed_ += static_cast<long long>(batch.size());
            }
            done_cv_.notify_all();
        }
    }

    // The fan-out: every observer reads the SAME captured residuals. Runs on the worker thread.
    // The whole BATCH goes through one readout_multi call — one graph, one head read, one round of
    // syncs for every frame and every layer in the batch (the amortization loop() exists for).
    // Emits one `readout` json per input frame, in order, so the wire shape is batch-invariant.
    std::vector<json> observe_batch(const std::vector<CaptureFrame>& batch) {
        // Per-layer concatenation across frames: layer -> [total_rows * n_embd], frame-major.
        // Frames in a batch come from one request, so they share the same capture layer set.
        int total_rows = 0;
        for (const auto& f : batch) total_rows += f.rows;
        const int n_embd = batch.empty() ? 0 : batch.front().n_embd;
        std::map<int, std::vector<float>> cat;
        for (const auto& f : batch) {
            for (const auto& lv : f.layers) {
                std::vector<float>& dst = cat[lv.first];
                dst.reserve(static_cast<size_t>(total_rows) * n_embd);
                dst.insert(dst.end(), lv.second.begin(), lv.second.end());
            }
        }
        std::map<int, std::vector<std::vector<std::pair<int, float>>>> jl_out;
        std::string jl_err;
        if (cfg_.jlens && lens_ && lens_->on && total_rows > 0) {
            std::vector<std::pair<int, const float*>> h_by_layer;
            for (const auto& kv : cat)
                if (kv.second.size() == static_cast<size_t>(total_rows) * n_embd)  // present in every frame
                    h_by_layer.emplace_back(kv.first, kv.second.data());
            lens_->readout_multi(h_by_layer, total_rows, cfg_.topk, jl_out, jl_err);
        }
        std::vector<json> out;
        out.reserve(batch.size());
        int row_base = 0;
        for (const auto& f : batch) {
            json layers = json::object();
            for (const auto& lv : f.layers) {
                const int il = lv.first;
                const std::vector<float>& h = lv.second;
                json lj = json::object();
                if (cfg_.norms) {
                    json ns = json::array();
                    for (int r = 0; r < f.rows; ++r) {
                        const float* row = h.data() + static_cast<size_t>(r) * f.n_embd;
                        double ss = 0.0;
                        for (int i = 0; i < f.n_embd; ++i) ss += static_cast<double>(row[i]) * row[i];
                        ns.push_back(std::sqrt(ss));
                    }
                    lj["norm"] = std::move(ns);
                }
                if (cfg_.jlens && lens_ && lens_->on && lens_->has(il)) {
                    auto it = jl_out.find(il);
                    if (it != jl_out.end() && static_cast<int>(it->second.size()) == total_rows) {
                        json rd = json::array();
                        for (int r = 0; r < f.rows; ++r) {  // this frame's slice of the batch rows
                            const auto& row = it->second[static_cast<size_t>(row_base + r)];
                            json rr = json::array();
                            for (const auto& pr : row)
                                rr.push_back({{"id", pr.first},
                                              {"piece", model_->decode({pr.first})},
                                              {"score", pr.second}});
                            rd.push_back(std::move(rr));
                        }
                        lj["jlens"] = std::move(rd);
                    } else {
                        lj["jlens_error"] = jl_err.empty() ? "readout failed" : jl_err;
                    }
                }
                if (cfg_.probes && probes_ && probes_->ready() && il == cfg_.probe_layer && f.rows > 0) {
                    // project the LAST row (the just-committed token — same semantics as StepFeatures)
                    const float* last = h.data() + static_cast<size_t>(f.rows - 1) * f.n_embd;
                    lj["probes"] = {{"names", probes_->names}, {"scores", probes_->project(last)}};
                }
                layers[std::to_string(il)] = std::move(lj);
            }
            json positions = json::array();
            for (int r = 0; r < f.rows; ++r) positions.push_back(f.from + r);
            out.push_back(json{{"type", "readout"}, {"positions", std::move(positions)},
                               {"layers", std::move(layers)}});
            row_base += f.rows;
        }
        return out;
    }

    ReadoutObserverConfig cfg_;
    JlensServe* lens_;
    const ConceptProbes* probes_;
    const GgmlModel* model_;

    std::mutex mtx_;
    std::condition_variable cv_;        // wakes the worker (new frame / stop)
    std::condition_variable done_cv_;   // wakes finish() (queue drained + worker idle)
    std::deque<CaptureFrame> queue_;
    std::vector<json> ready_;
    std::thread worker_;
    bool stop_ = false;
    bool busy_ = false;
    bool flushing_ = false;  // finish() called: drain without the accumulation wait

    // Coverage accounting (reported in readout_stats; push-side counters are single-producer).
    long long seen_gen_ = 0;
    long long observed_ = 0;
    long long dropped_ = 0;
    long long skipped_every_ = 0;
    long long skipped_prompt_ = 0;
};

}  // namespace clozn
