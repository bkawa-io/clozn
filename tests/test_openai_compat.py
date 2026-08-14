"""Model-free contract tests for the explicit OpenAI endpoint/field matrix."""
import pytest

from clozn.server.openai_compat import (
    CompatibilityError,
    normalize_chat_request,
    normalize_responses_request,
)


def test_chat_normalizes_current_token_limit_and_developer_role():
    out = normalize_chat_request({
        "model": "local",
        "messages": [{"role": "developer", "content": "Be terse."},
                     {"role": "user", "content": "Hello"}],
        "max_completion_tokens": 12,
        "temperature": 0,
        "top_p": 0.75,
        "seed": 9,
        "n": 1,
        "user": "sdk-user",
    })
    assert out["messages"][0] == {"role": "system", "content": "Be terse."}
    assert out["max_tokens"] == 12
    assert out["temperature"] == 0.0 and out["top_p"] == 0.75 and out["seed"] == 9
    assert "max_completion_tokens" not in out
    assert "n" not in out and "user" not in out


@pytest.mark.parametrize("field,value", [
    ("n", 2),
    ("frequency_penalty", 0.5),
])
def test_chat_rejects_behavior_it_cannot_honor(field, value):
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({"messages": [{"role": "user", "content": "hi"}], field: value})
    assert caught.value.param == field


def test_chat_normalizes_native_stop_and_stream_usage():
    normalized = normalize_chat_request({
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "END",
        "stream_options": {"include_usage": True},
    })
    assert normalized["stop"] == ["END"]
    assert normalized["stream_options"] == {"include_usage": True}


def test_responses_normalizes_text_input_to_chat_messages():
    out = normalize_responses_request({
        "model": "local",
        "instructions": "Be concise.",
        "input": "Explain mutexes.",
        "max_output_tokens": 12,
    })
    assert out["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain mutexes."},
    ]
    assert out["max_tokens"] == 12
    assert out["stream"] is False


def test_responses_streaming_and_tools_fail_closed():
    with pytest.raises(CompatibilityError) as stream_error:
        normalize_responses_request({"input": "hi", "stream": True})
    assert stream_error.value.code == "responses_streaming_not_supported"
    with pytest.raises(CompatibilityError) as tools_error:
        normalize_responses_request({"input": "hi", "tools": [{"type": "function"}]})
    assert tools_error.value.code == "responses_tools_not_supported"


@pytest.mark.parametrize("value", [[], [""], ["END"] * 5, ["END", "END"]])
def test_chat_rejects_invalid_stop_shape(value):
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": value,
        })
    assert caught.value.param.startswith("stop")


def test_chat_rejects_unknown_field_instead_of_silently_dropping_it():
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({"messages": [{"role": "user", "content": "hi"}], "magic": 1})
    assert caught.value.param == "magic"


def test_chat_accepts_explicit_unique_source_metadata_without_putting_it_in_messages():
    out = normalize_chat_request({
        "messages": [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Document text"},
        ],
        "clozn_sources": [
            {"message_index": 1, "source_id": "customer-handbook", "label": "Customer handbook"},
        ],
    })
    assert out["messages"][1] == {"role": "user", "content": "Document text"}
    assert out["_clozn_sources"] == [{
        "message_index": 1,
        "source_id": "customer-handbook",
        "label": "Customer handbook",
    }]


def test_chat_accepts_multiple_exact_unicode_sources_on_one_message_and_derives_utf8_ranges():
    out = normalize_chat_request({
        "messages": [{"role": "user", "content": "x 😀 policy + question"}],
        "clozn_sources": [
            {"message_index": 0, "source_id": "policy", "unicode_range": [4, 10],
             "provenance_kind": "retrieved_document"},
            {"message_index": 0, "source_id": "question", "unicode_range": [13, 21],
             "provenance_kind": "conversation_turn"},
        ],
    })
    sources = out["_clozn_sources"]
    assert [item["unicode_range"] for item in sources] == [[4, 10], [13, 21]]
    assert sources[0]["byte_range"] == [7, 13]
    assert out["messages"] == [{"role": "user", "content": "x 😀 policy + question"}]


def test_chat_rejects_non_nested_overlapping_exact_source_ranges():
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({
            "messages": [{"role": "user", "content": "abcdefgh"}],
            "clozn_sources": [
                {"message_index": 0, "source_id": "a", "unicode_range": [1, 5]},
                {"message_index": 0, "source_id": "b", "unicode_range": [4, 7]},
            ],
        })
    assert caught.value.param == "clozn_sources"


def test_chat_rejects_source_byte_range_that_does_not_match_unicode_utf8_boundary():
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({
            "messages": [{"role": "user", "content": "😀 abc"}],
            "clozn_sources": [{
                "message_index": 0, "source_id": "emoji", "unicode_range": [0, 1],
                "byte_range": [0, 1],
            }],
        })
    assert caught.value.param == "clozn_sources[0].byte_range"


def test_chat_accepts_explicit_structural_parent_for_nested_exact_source_ranges():
    out = normalize_chat_request({
        "messages": [{"role": "user", "content": "[section paragraph]"}],
        "clozn_sources": [
            {"message_index": 0, "source_id": "section", "unicode_range": [0, 19]},
            {"message_index": 0, "source_id": "paragraph", "unicode_range": [1, 18],
             "parent_source_id": "section"},
        ],
    })
    assert out["_clozn_sources"][1]["parent_source_id"] == "section"


@pytest.mark.parametrize("sources", [
    [{"message_index": 0, "source_id": "self", "unicode_range": [0, 4],
      "parent_source_id": "self"}],
    [
        {"message_index": 0, "source_id": "a", "unicode_range": [0, 4],
         "parent_source_id": "b"},
        {"message_index": 0, "source_id": "b", "unicode_range": [0, 4],
         "parent_source_id": "a"},
    ],
])
def test_chat_rejects_cyclic_source_hierarchies(sources):
    with pytest.raises(CompatibilityError, match="acyclic"):
        normalize_chat_request({
            "messages": [{"role": "user", "content": "text"}],
            "clozn_sources": sources,
        })


@pytest.mark.parametrize("sources,param", [
    (
        [
            {"message_index": 0, "source_id": "duplicate"},
            {"message_index": 1, "source_id": "duplicate"},
        ],
        "clozn_sources[1].source_id",
    ),
    ([{"message_index": 4, "source_id": "missing"}], "clozn_sources[0].message_index"),
])
def test_chat_rejects_ambiguous_source_metadata(sources, param):
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({
            "messages": [
                {"role": "system", "content": "Policy"},
                {"role": "user", "content": "Question"},
            ],
            "clozn_sources": sources,
        })
    assert caught.value.param == param


def test_chat_accepts_and_strips_documented_neutral_values():
    out = normalize_chat_request({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [], "tool_choice": "none", "store": False, "metadata": {},
        "response_format": {"type": "text"}, "frequency_penalty": 0,
        "presence_penalty": 0, "logprobs": False,
    })
    assert out == {"messages": [{"role": "user", "content": "hi"}]}


def test_nullable_supported_options_are_treated_as_absent():
    chat = normalize_chat_request({"messages": [{"role": "user", "content": "hi"}],
                                   "max_tokens": None, "temperature": None, "stream": None})
    assert chat == {"messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("field,value", [("temperature", float("nan")), ("top_p", float("inf"))])
def test_chat_rejects_non_finite_sampling_numbers(field, value):
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({"messages": [{"role": "user", "content": "hi"}], field: value})
    assert caught.value.param == field


def test_chat_rejects_multimodal_tool_and_extra_message_fields_precisely():
    cases = [
        ({"role": "user", "content": [{"type": "text", "text": "hi"}]}, "messages[0].content"),
        ({"role": "tool", "content": "result", "tool_call_id": "call_1"}, "messages[0].tool_call_id"),
        ({"role": "user", "content": "hi", "name": "alice"}, "messages[0].name"),
    ]
    for message, param in cases:
        with pytest.raises(CompatibilityError) as caught:
            normalize_chat_request({"messages": [message]})
        assert caught.value.param == param


def test_chat_rejects_conflicting_token_limit_aliases():
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({"messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 8, "max_completion_tokens": 9})
    assert caught.value.param == "max_completion_tokens"
