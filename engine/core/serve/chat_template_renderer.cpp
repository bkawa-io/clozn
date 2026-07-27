#include "chat_template_renderer.hpp"

#include "chat.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <stdexcept>

namespace clozn {

struct ChatTemplateRenderer::Impl {
    common_chat_templates_ptr templates;
};

ChatTemplateRenderer::ChatTemplateRenderer(const llama_model* model) : impl_(std::make_unique<Impl>()) {
    if (model != nullptr && llama_model_chat_template(model, /*name=*/nullptr) != nullptr) {
        impl_->templates = common_chat_templates_init(model, /*chat_template_override=*/"");
    }
}

ChatTemplateRenderer::ChatTemplateRenderer(
    std::string template_source,
    std::string bos_token,
    std::string eos_token) : impl_(std::make_unique<Impl>()) {
    if (!template_source.empty()) {
        impl_->templates = common_chat_templates_init(
            /*model=*/nullptr,
            template_source,
            bos_token,
            eos_token);
    }
}

ChatTemplateRenderer::~ChatTemplateRenderer() = default;

bool ChatTemplateRenderer::available() const noexcept {
    return impl_ != nullptr && impl_->templates != nullptr;
}

std::string ChatTemplateRenderer::apply(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_assistant) const {
    if (!available()) {
        throw std::runtime_error("model has no embedded chat template");
    }
    common_chat_templates_inputs inputs;
    inputs.messages.reserve(messages.size());
    for (const auto& message : messages) {
        inputs.messages.push_back(common_chat_msg{message.first, message.second});
    }
    inputs.add_generation_prompt = add_assistant;
    inputs.use_jinja = true;
    return common_chat_templates_apply(impl_->templates.get(), inputs).prompt;
}

namespace {

std::string grammar_trigger_type_name(common_grammar_trigger_type type) {
    switch (type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            return "token";
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            return "word";
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            return "pattern";
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            return "pattern_full";
    }
    throw std::runtime_error("unknown llama.cpp grammar trigger type");
}

common_chat_tool_choice parse_tool_choice(
    const nlohmann::ordered_json& value,
    std::vector<common_chat_tool>& tools) {
    if (value.is_string()) {
        return common_chat_tool_choice_parse_oaicompat(value.get<std::string>());
    }
    if (!value.is_object() || value.value("type", "") != "function" ||
        !value.contains("function") || !value.at("function").is_object() ||
        !value.at("function").contains("name") || !value.at("function").at("name").is_string()) {
        throw std::invalid_argument(
            "tool_choice must be auto, required, none, or a named function choice");
    }

    const std::string name = value.at("function").at("name").get<std::string>();
    const auto selected = std::find_if(tools.begin(), tools.end(), [&](const common_chat_tool& tool) {
        return tool.name == name;
    });
    if (selected == tools.end()) {
        throw std::invalid_argument("named tool_choice does not match a declared tool: " + name);
    }
    common_chat_tool tool = *selected;
    tools.assign(1, std::move(tool));
    return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
}

}  // namespace

PreparedChat ChatTemplateRenderer::prepare(const ChatTemplateRequest& request) const {
    if (!available()) {
        throw std::runtime_error("model has no embedded chat template");
    }

    const auto messages_json = nlohmann::ordered_json::parse(request.messages_json);
    const auto tools_json = nlohmann::ordered_json::parse(request.tools_json.empty() ? "[]" : request.tools_json);
    const auto tool_choice_json = nlohmann::ordered_json::parse(
        request.tool_choice_json.empty() ? "\"auto\"" : request.tool_choice_json);

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(messages_json);
    inputs.tools = common_chat_tools_parse_oaicompat(tools_json);
    inputs.tool_choice = parse_tool_choice(tool_choice_json, inputs.tools);
    inputs.parallel_tool_calls = request.parallel_tool_calls;
    inputs.add_generation_prompt = request.add_generation_prompt;
    inputs.enable_thinking = request.enable_thinking;
    inputs.reasoning_format = common_reasoning_format_from_name(request.reasoning_format);
    inputs.use_jinja = true;

    if (!request.json_schema_json.empty()) {
        const auto schema = nlohmann::ordered_json::parse(request.json_schema_json);
        if (!schema.is_object()) {
            throw std::invalid_argument("json_schema must be a JSON object");
        }
        if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            throw std::invalid_argument("json_schema and active tools are mutually exclusive");
        }
        inputs.json_schema = schema.dump();
    }
    if (inputs.tools.empty() && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
        throw std::invalid_argument("required tool_choice requires at least one declared tool");
    }

    const common_chat_params params = common_chat_templates_apply(impl_->templates.get(), inputs);
    const bool structured_requested =
        (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) ||
        !request.json_schema_json.empty();
    if (structured_requested && params.grammar.empty()) {
        throw std::invalid_argument(
            "loaded chat template did not emit a grammar for the requested structured output");
    }
    if (structured_requested && params.grammar.rfind("%llguidance", 0) == 0) {
        throw std::invalid_argument(
            "loaded chat template requires llguidance, which this native structured path does not support");
    }
    PreparedChat result;
    result.prompt = params.prompt;
    result.grammar = params.grammar;
    result.grammar_lazy = params.grammar_lazy;
    result.preserved_tokens = params.preserved_tokens;
    result.additional_stops = params.additional_stops;
    result.generation_prompt = params.generation_prompt;
    result.parser = params.parser;
    result.format = common_chat_format_name(params.format);
    result.capabilities = common_chat_templates_get_caps(impl_->templates.get());
    result.supports_thinking = params.supports_thinking;
    result.thinking_start_tag = params.thinking_start_tag;
    result.thinking_end_tag = params.thinking_end_tag;
    result.reasoning_format = request.reasoning_format;
    result.parse_tool_calls = !inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    result.grammar_triggers.reserve(params.grammar_triggers.size());
    for (const auto& trigger : params.grammar_triggers) {
        result.grammar_triggers.push_back(ChatGrammarTrigger{
            grammar_trigger_type_name(trigger.type),
            trigger.value,
            static_cast<std::int32_t>(trigger.token),
        });
    }
    return result;
}

ParsedChat ChatTemplateRenderer::parse(
    const PreparedChat& prepared,
    const std::string& model_output,
    bool is_partial) const {
    common_chat_parser_params params;
    params.format = [&]() {
        for (int value = 0; value < static_cast<int>(COMMON_CHAT_FORMAT_COUNT); ++value) {
            auto format = static_cast<common_chat_format>(value);
            if (prepared.format == common_chat_format_name(format)) {
                return format;
            }
        }
        throw std::invalid_argument("unknown prepared chat format: " + prepared.format);
    }();
    params.reasoning_format = common_reasoning_format_from_name(prepared.reasoning_format);
    params.generation_prompt = prepared.generation_prompt;
    params.parse_tool_calls = prepared.parse_tool_calls;
    if (!prepared.parser.empty()) {
        params.parser.load(prepared.parser);
    }

    const common_chat_msg message = common_chat_parse(model_output, is_partial, params);
    ParsedChat result;
    result.role = message.role;
    result.content = message.content;
    result.reasoning_content = message.reasoning_content;
    result.tool_name = message.tool_name;
    result.tool_call_id = message.tool_call_id;
    result.tool_calls.reserve(message.tool_calls.size());
    for (const auto& tool_call : message.tool_calls) {
        result.tool_calls.push_back(ParsedToolCall{
            tool_call.id,
            tool_call.name,
            tool_call.arguments,
        });
    }
    result.openai_json = message.to_json_oaicompat().dump();
    return result;
}

}  // namespace clozn
