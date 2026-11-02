#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace clozn {

// Model-free capacity accounting for native teacher-forced multi-arm scoring.
//
// Prompt rows are materialized through an exact-token trie: one physical row per
// distinct token prefix, regardless of model family. Continuation rows cannot be
// shared after prompt arms diverge because their hidden states are different, even
// when the forced token IDs are identical. The final continuation token is a target
// only, so it does not need its own resident KV row.
struct TeacherForcedScoreManyPlan {
    int arm_count = 0;
    long long logical_prompt_rows = 0;
    long long physical_prompt_rows = 0;
    long long prefix_rows_reused = 0;
    long long continuation_rows = 0;
    long long total_physical_rows = 0;
};

inline TeacherForcedScoreManyPlan plan_teacher_forced_score_many(
        const std::vector<std::vector<int>>& prompts,
        std::size_t continuation_length,
        int n_ctx,
        int max_arms = 16) {
    if (prompts.empty())
        throw std::invalid_argument("teacher-forced score-many needs at least one arm");
    if (prompts.size() > static_cast<std::size_t>(max_arms))
        throw std::invalid_argument("teacher-forced score-many exceeds the arm limit");
    if (continuation_length == 0)
        throw std::invalid_argument("teacher-forced score-many needs a continuation");
    if (n_ctx <= 0)
        throw std::invalid_argument("teacher-forced score-many has an invalid context window");

    struct Node {
        std::unordered_map<int, std::unique_ptr<Node>> children;
    };
    Node root;
    TeacherForcedScoreManyPlan plan;
    plan.arm_count = static_cast<int>(prompts.size());

    for (const auto& prompt : prompts) {
        if (prompt.empty())
            throw std::invalid_argument("teacher-forced score-many prompt is empty");
        // Keep the same per-arm context contract as scalar /score. The native path
        // physically needs one fewer row, but accepting more here would make fallback
        // behavior depend on which execution path happened to run.
        if (prompt.size() + continuation_length > static_cast<std::size_t>(n_ctx))
            throw std::invalid_argument(
                "teacher-forced score-many prompt + continuation exceeds n_ctx");

        plan.logical_prompt_rows += static_cast<long long>(prompt.size());
        Node* node = &root;
        for (int token : prompt) {
            auto& child = node->children[token];
            if (!child) {
                child = std::make_unique<Node>();
                ++plan.physical_prompt_rows;
            }
            node = child.get();
        }
    }

    plan.prefix_rows_reused =
        plan.logical_prompt_rows - plan.physical_prompt_rows;
    plan.continuation_rows = static_cast<long long>(prompts.size()) *
        static_cast<long long>(continuation_length - 1);
    plan.total_physical_rows =
        plan.physical_prompt_rows + plan.continuation_rows;
    if (plan.total_physical_rows > n_ctx)
        throw std::invalid_argument(
            "teacher-forced score-many physical rows exceed n_ctx; use fewer arms, "
            "shorter prompts, or a shorter continuation");
    return plan;
}

}  // namespace clozn
