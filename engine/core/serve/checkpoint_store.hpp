// serve/checkpoint_store.hpp -- process-generation-scoped, bounded in-memory checkpoint storage.
//
// Checkpoints are private worker state. Their references must therefore identify the exact worker
// process that owns them: a bare process-local counter can collide after restart and make a persisted
// reference appear to name unrelated state. The worker creates one opaque generation id at startup,
// and every id issued by this store embeds it:
//
//     ckpt-<opaque worker generation id>-<monotonic in-process counter>
//
// The id remains one opaque string to existing in-process callers. The generation id is also returned
// separately on the wire so future persistence can retain and compare the compound identity without
// parsing checkpoint_id.
//
// Capacity eviction is insertion-order FIFO. std::map::erase(begin()) is lexical-key eviction, which
// diverges from insertion order once counters cross digit boundaries (ckpt-10 sorts before ckpt-2).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace clozn {

inline std::string make_worker_generation_id() {
    // 128 bits from the platform random source. This value is an identity nonce, not a secret; failure
    // to obtain randomness aborts worker startup rather than falling back to a restart-colliding counter.
    std::random_device source;
    std::array<unsigned char, 16> bytes{};
    for (unsigned char& byte : bytes)
        byte = static_cast<unsigned char>(source() & 0xffU);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : bytes)
        out << std::setw(2) << static_cast<unsigned int>(byte);
    return out.str();
}

template <typename Value>
class CheckpointStore {
public:
    CheckpointStore(std::string worker_generation_id, std::size_t capacity)
        : worker_generation_id_(std::move(worker_generation_id)), capacity_(capacity) {
        if (worker_generation_id_.empty())
            throw std::invalid_argument("worker_generation_id must not be empty");
        if (capacity_ == 0)
            throw std::invalid_argument("checkpoint capacity must be positive");
    }

    const std::string& worker_generation_id() const noexcept {
        return worker_generation_id_;
    }

    std::string insert(Value value) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string id = "ckpt-" + worker_generation_id_ + "-" +
                               std::to_string(next_checkpoint_number_++);
        while (insertion_order_.size() >= capacity_) {
            records_.erase(insertion_order_.front());
            insertion_order_.pop_front();
        }
        records_.emplace(id, std::move(value));
        insertion_order_.push_back(id);
        return id;
    }

    std::optional<Value> find_copy(const std::string& id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = records_.find(id);
        if (found == records_.end())
            return std::nullopt;
        return found->second;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.size();
    }

private:
    const std::string worker_generation_id_;
    const std::size_t capacity_;
    std::uint64_t next_checkpoint_number_ = 0;
    mutable std::mutex mutex_;
    std::map<std::string, Value> records_;
    std::deque<std::string> insertion_order_;
};

}  // namespace clozn
