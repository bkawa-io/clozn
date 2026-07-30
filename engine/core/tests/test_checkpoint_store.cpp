// Model-free contract test for process-generation-scoped checkpoint ids and true FIFO eviction.
#include <cstdio>
#include <string>

#include "checkpoint_store.hpp"

using clozn::CheckpointStore;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition); \
            return 1;                                                                     \
        }                                                                                 \
    } while (false)

int main() {
    CheckpointStore<std::string> store("generation-a", 2);
    CHECK(store.worker_generation_id() == "generation-a");

    const std::string first = store.insert("first");
    const std::string second = store.insert("second");
    CHECK(first == "ckpt-generation-a-0");
    CHECK(second == "ckpt-generation-a-1");
    CHECK(store.find_copy(first).value() == "first");

    const std::string third = store.insert("third");
    CHECK(third == "ckpt-generation-a-2");
    CHECK(!store.find_copy(first).has_value());
    CHECK(store.find_copy(second).value() == "second");
    CHECK(store.find_copy(third).value() == "third");
    CHECK(store.size() == 2);

    // A new worker generation may restart its local counter at zero without colliding.
    CheckpointStore<std::string> restarted("generation-b", 2);
    CHECK(restarted.insert("new process") == "ckpt-generation-b-0");

    // The historical lexical-erase bug becomes visible at the 9 -> 10 boundary. FIFO must retain
    // the latest two insertions regardless of how their textual suffixes sort.
    CheckpointStore<int> decimal_boundary("generation-c", 2);
    std::string previous;
    std::string latest;
    for (int i = 0; i < 12; ++i) {
        previous = latest;
        latest = decimal_boundary.insert(i);
    }
    CHECK(previous == "ckpt-generation-c-10");
    CHECK(latest == "ckpt-generation-c-11");
    CHECK(decimal_boundary.find_copy(previous).value() == 10);
    CHECK(decimal_boundary.find_copy(latest).value() == 11);
    CHECK(!decimal_boundary.find_copy("ckpt-generation-c-9").has_value());

    std::puts("checkpoint store tests passed");
    return 0;
}
