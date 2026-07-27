#include "chat_template_renderer.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open test template: " + path);
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
}

clozn::ChatTemplateRenderer ministral_renderer() {
    const std::string source_dir = CLOZN_SOURCE_DIR;
    return clozn::ChatTemplateRenderer(read_file(
        source_dir + "/third_party/llama.cpp/models/templates/"
                     "mistralai-Ministral-3-14B-Reasoning-2512.jinja"));
}

const char* weather_tools = R"json([
  {
    "type": "function",
    "function": {
      "name": "weather",
      "description": "Get the weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
      }
    }
  }
])json";

void test_plain_content() {
    auto renderer = ministral_renderer();
    require(renderer.available(), "template override should be available");

    clozn::ChatTemplateRequest request;
    request.messages_json = R"json([{"role":"user","content":"Hello"}])json";
    const auto prepared = renderer.prepare(request);
    require(!prepared.prompt.empty(), "plain request should render a prompt");
    require(!prepared.parser.empty(), "plain request should retain a parser descriptor");
    require(prepared.format == "peg-native", "Ministral should use its native PEG parser");

    const auto parsed = renderer.parse(prepared, "A plain answer");
    require(parsed.role == "assistant", "parsed role should be assistant");
    require(parsed.content == "A plain answer", "plain content should round-trip");
    require(parsed.tool_calls.empty(), "plain content must not invent tool calls");
    require(parsed.openai_json.find("A plain answer") != std::string::npos,
            "complete OpenAI JSON should include content");
}

void test_tools_and_history() {
    auto renderer = ministral_renderer();
    clozn::ChatTemplateRequest request;
    request.messages_json = R"json([
      {"role":"system","content":"Be concise."},
      {"role":"user","content":"Weather in Rome?"},
      {"role":"assistant","content":null,"tool_calls":[{
        "id":"call_previous","type":"function",
        "function":{"name":"weather","arguments":"{\"city\":\"Rome\"}"}
      }]},
      {"role":"tool","tool_call_id":"call_previous","content":"sunny"}
    ])json";
    request.tools_json = weather_tools;
    request.tool_choice_json =
        R"json({"type":"function","function":{"name":"weather"}})json";
    request.reasoning_format = "auto";
    const auto prepared = renderer.prepare(request);

    require(prepared.prompt.find("[TOOL_CALLS]weather[ARGS]") != std::string::npos &&
                prepared.prompt.find("sunny") != std::string::npos,
            "rich assistant/tool history should reach the model template");
    require(!prepared.grammar.empty(), "required tool choice should produce a grammar");
    require(!prepared.grammar_lazy, "required tool choice grammar must be eager");
    require(!prepared.grammar_triggers.empty(), "tool grammar should expose lazy triggers");
    require(prepared.grammar_triggers.front().type == "word",
            "trigger type should be portable and explicit");
    require(prepared.grammar_triggers.front().value == "[TOOL_CALLS]",
            "Ministral tool trigger should be preserved");
    require(prepared.capabilities.at("supports_tool_calls"),
            "template capabilities should be exposed");
    require(prepared.parse_tool_calls, "tool parser should be enabled");

    const auto parsed = renderer.parse(
        prepared,
        R"([THINK]I should check Paris.[/THINK][TOOL_CALLS]weather[ARGS]{"city":"Paris"})");
    require(parsed.reasoning_content == "I should check Paris.",
            "reasoning should be separated by llama.cpp's parser");
    require(parsed.content.empty(), "a pure tool call should have no content");
    require(parsed.tool_calls.size() == 1, "one tool call should be parsed");
    require(parsed.tool_calls[0].name == "weather", "tool name should round-trip");
    require(parsed.tool_calls[0].arguments == R"({"city":"Paris"})",
            "tool arguments should remain a JSON string");
    require(parsed.openai_json.find("tool_calls") != std::string::npos,
            "complete OpenAI JSON should include tool calls");
}

void test_json_schema() {
    auto renderer = ministral_renderer();
    clozn::ChatTemplateRequest request;
    request.messages_json = R"json([{"role":"user","content":"Give an invoice."}])json";
    request.json_schema_json = R"json({
      "type":"object",
      "properties":{"amount":{"type":"number"}},
      "required":["amount"],
      "additionalProperties":false
    })json";
    request.reasoning_format = "auto";
    const auto prepared = renderer.prepare(request);
    require(!prepared.grammar.empty(), "JSON schema should produce a grammar");
    require(!prepared.grammar_lazy, "response schema grammar should be eager");
    require(!prepared.parser.empty(), "response schema should produce a parser descriptor");

    const auto parsed = renderer.parse(
        prepared,
        "[THINK]Structured answer.[/THINK]```json{\"amount\":12.5}```");
    require(parsed.reasoning_content == "Structured answer.",
            "schema parse should retain separated reasoning");
    if (parsed.content != R"({"amount":12.5})") {
        throw std::runtime_error("schema parse should extract JSON content, got: " + parsed.content);
    }
    require(parsed.tool_calls.empty(), "schema output must not contain tool calls");
}

void test_fail_closed_request_validation() {
    auto renderer = ministral_renderer();
    clozn::ChatTemplateRequest request;
    request.messages_json = R"json([{"role":"user","content":"Hello"}])json";
    request.tools_json = weather_tools;
    request.tool_choice_json =
        R"json({"type":"function","function":{"name":"missing"}})json";
    bool rejected_missing = false;
    try {
        (void)renderer.prepare(request);
    } catch (const std::invalid_argument&) {
        rejected_missing = true;
    }
    require(rejected_missing, "unknown named tool choice should fail closed");

    request.tool_choice_json = R"json("auto")json";
    request.json_schema_json = R"json({"type":"object"})json";
    bool rejected_conflict = false;
    try {
        (void)renderer.prepare(request);
    } catch (const std::invalid_argument&) {
        rejected_conflict = true;
    }
    require(rejected_conflict, "active tools and JSON schema should be mutually exclusive");

    // A syntactically valid template that ignores tool metadata must not silently turn an active
    // structured request into unconstrained text generation.
    clozn::ChatTemplateRenderer plain_renderer(
        R"jinja({% for message in messages %}{{ message['content'] }}{% endfor %})jinja");
    request.messages_json = R"json([{"role":"user","content":"Hello"}])json";
    request.tools_json = weather_tools;
    request.tool_choice_json = R"json("auto")json";
    request.json_schema_json.clear();
    bool rejected_missing_grammar = false;
    try {
        (void)plain_renderer.prepare(request);
    } catch (const std::invalid_argument&) {
        rejected_missing_grammar = true;
    }
    require(rejected_missing_grammar,
            "structured request must fail when the template emits no grammar");
}

}  // namespace

int main() {
    try {
        test_plain_content();
        test_tools_and_history();
        test_json_schema();
        test_fail_closed_request_validation();
        std::cout << "chat template renderer tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "chat template renderer test failed: " << error.what() << '\n';
        return 1;
    }
}
