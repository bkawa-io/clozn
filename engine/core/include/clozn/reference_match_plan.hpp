// Exact-token radix planning for the request-wide reference matcher.
//
// The plan is deliberately independent of llama.cpp.  It describes the order in which a
// single seq-0 executor should decode and roll back prompt prefixes; it never assigns a
// prompt to a model sequence and it never merges non-identical token histories.
#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace clozn {

struct ReferenceMatchTraversalPlan {
    struct Node {
        int begin = 0;       // inclusive index into order
        int end = 0;         // exclusive index into order
        int depth = 0;       // shared prefix already resident on seq 0 at entry
        int common_end = 0;  // maximal shared prefix for this range
        int terminal_begin = 0;
        int terminal_end = 0;
        std::vector<int> children;
    };

    std::vector<int> order;
    std::vector<Node> nodes;
    int root = -1;
};

// Build a sorted-index radix plan.  Literal token IDs are the only equality relation used here.
// Empty prompts are representable by the planner, although the native matcher rejects them before
// execution because there is no prompt-final logits row to probe.
inline ReferenceMatchTraversalPlan make_reference_match_traversal_plan(
    const std::vector<std::vector<int>>& prompts) {
    if (prompts.empty()) throw std::invalid_argument("reference-match traversal needs prompts");

    ReferenceMatchTraversalPlan plan;
    plan.order.resize(prompts.size());
    std::iota(plan.order.begin(), plan.order.end(), 0);
    std::sort(plan.order.begin(), plan.order.end(), [&](int a, int b) {
        return prompts[static_cast<size_t>(a)] < prompts[static_cast<size_t>(b)];
    });

    auto visit = [&](auto&& self, int begin, int end, int depth) -> int {
        const auto& first = prompts[static_cast<size_t>(plan.order[static_cast<size_t>(begin)])];
        const auto& last = prompts[static_cast<size_t>(plan.order[static_cast<size_t>(end - 1)])];
        const int limit = std::min(static_cast<int>(first.size()), static_cast<int>(last.size()));
        int common_end = depth;
        while (common_end < limit &&
               first[static_cast<size_t>(common_end)] == last[static_cast<size_t>(common_end)]) {
            ++common_end;
        }

        const int node_id = static_cast<int>(plan.nodes.size());
        plan.nodes.push_back(ReferenceMatchTraversalPlan::Node{
            begin, end, depth, common_end, begin, begin, {}});

        int terminal_end = begin;
        while (terminal_end < end &&
               static_cast<int>(prompts[static_cast<size_t>(plan.order[static_cast<size_t>(terminal_end)])].size()) ==
                   common_end) {
            ++terminal_end;
        }
        plan.nodes[static_cast<size_t>(node_id)].terminal_end = terminal_end;

        int child_begin = terminal_end;
        while (child_begin < end) {
            const auto& child_prompt =
                prompts[static_cast<size_t>(plan.order[static_cast<size_t>(child_begin)])];
            if (static_cast<int>(child_prompt.size()) <= common_end)
                throw std::logic_error("reference-match radix plan lost a prompt child");
            const int child_token = child_prompt[static_cast<size_t>(common_end)];
            int child_end = child_begin + 1;
            while (child_end < end) {
                const auto& next_prompt =
                    prompts[static_cast<size_t>(plan.order[static_cast<size_t>(child_end)])];
                if (static_cast<int>(next_prompt.size()) <= common_end ||
                    next_prompt[static_cast<size_t>(common_end)] != child_token)
                    break;
                ++child_end;
            }
            const int child_id = self(self, child_begin, child_end, common_end);
            plan.nodes[static_cast<size_t>(node_id)].children.push_back(child_id);
            child_begin = child_end;
        }
        return node_id;
    };

    plan.root = visit(visit, 0, static_cast<int>(plan.order.size()), 0);
    return plan;
}

inline long long reference_match_plan_physical_rows(const ReferenceMatchTraversalPlan& plan) {
    long long rows = 0;
    for (const auto& node : plan.nodes)
        rows += static_cast<long long>(node.common_end - node.depth);
    return rows;
}

// Execute the plan with model-independent callbacks.  `decode` receives the representative arm,
// absolute [from,to), and whether the final row needs logits.  `probe` receives the sorted-order
// terminal range.  `evict` runs after every recursive node returns, so a fake adapter can assert
// the lifecycle invariant without loading a GGUF.
template <typename Decode, typename Probe, typename Evict>
void walk_reference_match_traversal(const ReferenceMatchTraversalPlan& plan,
                                    Decode&& decode, Probe&& probe, Evict&& evict) {
    if (plan.root < 0) throw std::invalid_argument("reference-match traversal has no root");
    std::function<void(int)> visit = [&](int node_id) {
        const auto& node = plan.nodes[static_cast<size_t>(node_id)];
        const bool has_terminal = node.terminal_begin < node.terminal_end;
        if (node.common_end > node.depth) {
            decode(plan.order[static_cast<size_t>(node.begin)],
                   node.depth, node.common_end, has_terminal);
        }
        if (has_terminal)
            probe(node.terminal_begin, node.terminal_end, node.common_end);
        for (int child_id : node.children)
            visit(child_id);
        evict(node.depth, node.common_end - node.depth);
    };
    visit(plan.root);
}

}  // namespace clozn
