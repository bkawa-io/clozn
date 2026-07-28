"""Model-free contract tests for the explicit OpenAI endpoint/field matrix."""
import pytest

from clozn.server.openai_compat import (
    CompatibilityError,
    normalize_chat_request,
    normalize_completion_request,
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
    ("stop", ["END"]),
    ("n", 2),
    ("frequency_penalty", 0.5),
    ("stream_options", {"include_usage": True}),
])
def test_chat_rejects_behavior_it_cannot_honor(field, value):
    with pytest.raises(CompatibilityError) as caught:
        normalize_chat_request({"messages": [{"role": "user", "content": "hi"}], field: value})
    assert caught.value.param == field
    assert caught.value.code == "unsupported_parameter"


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
    completion = normalize_completion_request({"prompt": "hi", "max_tokens": None,
                                               "temperature": None, "stream": None})
    assert chat == {"messages": [{"role": "user", "content": "hi"}]}
    assert completion == {"prompt": "hi"}


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


def test_completion_normalizes_extensions_and_neutral_legacy_fields():
    out = normalize_completion_request({
        "model": "local", "prompt": "Hello", "max_tokens": 7, "stream": False,
        "temperature": 0.5, "top_p": 0.8, "seed": 4, "top_k": 20,
        "repeat_penalty": 1.1, "n": 1, "best_of": 1, "echo": False, "user": "sdk-user",
    })
    assert out == {"model": "local", "prompt": "Hello", "max_tokens": 7, "stream": False,
                   "temperature": 0.5, "top_p": 0.8, "seed": 4, "top_k": 20,
                   "rep_penalty": 1.1}


@pytest.mark.parametrize("field,value", [("prompt", ["a", "b"]), ("echo", True),
                                           ("logprobs", 5), ("best_of", 2), ("stop", "END")])
def test_completion_rejects_unsupported_shapes_and_behavior(field, value):
    body = {"prompt": "hi", field: value}
    with pytest.raises(CompatibilityError) as caught:
        normalize_completion_request(body)
    assert caught.value.param == field
