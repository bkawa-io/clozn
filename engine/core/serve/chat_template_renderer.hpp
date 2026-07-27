#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;

namespace clozn {

inline constexpr const char* NATIVE_CHAT_EXECUTOR_ID =
    "clozn.chat_io.atomic_executor.v1";
inline constexpr const char* NATIVE_CHAT_RENDERER_ID =
    "clozn.chat_io.llama_common.renderer.v1";
inline constexpr const char* NATIVE_CHAT_GRAMMAR_ID =
    "clozn.chat_io.ar_grammar.v1";
inline constexpr const char* NATIVE_CHAT_PARSER_ID =
    "clozn.chat_io.llama_common.parser.v1";

// Private worker-side chat I/O contract. These structs intentionally contain no llama.cpp or
// nlohmann types: the Python compatibility server must not treat this as a qualified public API
// until a model/template pair has passed its structured-output conformance battery.
struct ChatTemplateRequest {
    // OpenAI-compatible JSON values. tool_choice_json accepts "auto", "required", "none", or
    // a named function choice object. A named choice is lowered to one required tool.
    std::string messages_json;
    std::string tools_json = "[]";
    std::string tool_choice_json = "\"auto\"";
    std::string json_schema_json;
    bool parallel_tool_calls = false;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::string reasoning_format = "none";
};

struct ChatGrammarTrigger {
    std::string type;
    std::string value;
    std::int32_t token = -1;
};

struct PreparedChat {
    std::string prompt;
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<ChatGrammarTrigger> grammar_triggers;
    std::vector<std::string> preserved_tokens;
    std::vector<std::string> additional_stops;

    // The generation prefix and serialized PEG parser are an inseparable parse descriptor.
    std::string generation_prompt;
    std::string parser;
    std::string format;
    std::map<std::string, bool> capabilities;
    bool supports_thinking = false;
    std::string thinking_start_tag;
    std::string thinking_end_tag;

    // Retained so parse() can faithfully reconstruct llama.cpp's parser parameters.
    std::string reasoning_format = "none";
    bool parse_tool_calls = false;
};

struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

struct ParsedChat {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::string tool_name;
    std::string tool_call_id;
    std::vector<ParsedToolCall> tool_calls;
    // Complete OpenAI-compatible message JSON, including fields not represented above.
    std::string openai_json;
};

// Keeps llama-common's Jinja/chat types out of the worker's public headers. In particular,
// chat.h defines its own global JSON alias, which must not bleed into server_shared.hpp.
class ChatTemplateRenderer {
public:
    explicit ChatTemplateRenderer(const llama_model* model);
    // Model-free constructor for native seam tests and template qualification tooling.
    explicit ChatTemplateRenderer(std::string template_source,
                                  std::string bos_token = "",
                                  std::string eos_token = "");
    ~ChatTemplateRenderer();

    ChatTemplateRenderer(const ChatTemplateRenderer&) = delete;
    ChatTemplateRenderer& operator=(const ChatTemplateRenderer&) = delete;

    bool available() const noexcept;
    std::string apply(const std::vector<std::pair<std::string, std::string>>& messages,
                      bool add_assistant) const;
    PreparedChat prepare(const ChatTemplateRequest& request) const;
    ParsedChat parse(const PreparedChat& prepared,
                     const std::string& model_output,
                     bool is_partial = false) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace clozn
