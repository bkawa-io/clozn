"""Phase 2.5: model scratch text is evidence, never assistant history/content."""
from clozn.runs.think_tags import (
    ThinkTagStream,
    prompt_opens_think,
    sanitize_messages,
    sanitize_reply,
    sanitize_steps,
)


def test_plain_and_incomplete_lookalike_are_byte_exact():
    for text in ("", "plain answer", "show <thi literally", "&lt;think&gt; escaped"):
        result = sanitize_reply(text)
        assert result.public_text == text
        assert result.journal() == {}


def test_closed_multiple_nested_and_unclosed_blocks_are_hidden():
    raw = "A<think>one</think>B<THINK>two<think>deep</think>end</THINK>C"
    result = sanitize_reply(raw)
    assert result.public_text == "ABC"
    assert [block["text"] for block in result.blocks] == ["one", "two<think>deep</think>end"]
    assert all(block["closed"] for block in result.blocks)

    cut = sanitize_reply("before<think>unfinished tool call")
    assert cut.public_text == "before"
    assert cut.blocks == ({"text": "unfinished tool call", "closed": False},)


def test_prompt_prefilled_and_legacy_inferred_reasoning_are_hidden():
    assert prompt_opens_think("assistant\n<think>\n") is True
    raw = "consider the options</think>\n\nanswer"
    explicit = sanitize_reply(raw, implicit_open=True, infer_implicit=False)
    inferred = sanitize_reply(raw)
    assert explicit.public_text == inferred.public_text == "\n\nanswer"
    assert explicit.blocks[0] == {"text": "consider the options", "closed": True}
    assert explicit.implicit_open is True


def test_stream_matches_batch_at_every_chunk_boundary():
    raw = "<think>private {\"tool\":\"delete\"}</think>public JSON"
    expected = sanitize_reply(raw)
    for split in range(len(raw) + 1):
        stream = ThinkTagStream()
        out = stream.feed(raw[:split]) + stream.feed(raw[split:])
        tail, result = stream.finish()
        assert out + tail == expected.public_text
        assert result.blocks == expected.blocks
        assert "delete" not in out + tail


def test_assistant_history_only_is_sanitized_and_external_thinking_is_dropped():
    messages = [
        {"role": "system", "content": "keep <think> literally"},
        {"role": "user", "content": "what does </think> mean?"},
        {"role": "assistant", "content": "old scratch</think>answer", "thinking": "old scratch"},
    ]
    clean = sanitize_messages(messages)
    assert clean[:2] == messages[:2]
    assert clean[2] == {"role": "assistant", "content": "answer"}
    assert messages[2]["thinking"] == "old scratch"  # pure: input was not mutated


def test_trace_is_partitioned_and_public_tokens_reconstruct_answer():
    steps = [
        {"index": 0, "piece": "<thi", "confidence": .9},
        {"index": 1, "piece": "nk>why</think>an", "confidence": .8},
        {"index": 2, "piece": "swer", "confidence": .7},
    ]
    public, reasoning, result = sanitize_steps(steps)
    assert "".join(step["piece"] for step in public) == result.public_text == "answer"
    assert reasoning
    assert all("why" not in step["piece"] for step in public)


def test_run_store_keeps_clean_answer_and_separate_reasoning_trace(tmp_path, monkeypatch):
    import clozn.runs.store as store

    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    raw = "scratch</think>\nanswer"
    steps = [
        {"index": 0, "piece": "scratch", "confidence": .6},
        {"index": 1, "piece": "</think>\n", "confidence": .7},
        {"index": 2, "piece": "answer", "confidence": .9},
    ]
    rid = store.record(
        source="test",
        messages=[{"role": "user", "content": "question"}],
        response=raw,
        trace=steps,
        final_prompt="assistant\n<think>\n",
    )
    rec = store.get_run(rid)
    assert rec["response"] == "\nanswer"
    assert rec["response_summary"] == "answer"
    assert rec["reasoning"]["schema"] == "clozn.reasoning_trace.v1"
    assert rec["reasoning"]["blocks"] == [{"text": "scratch", "closed": True}]
    assert "reasoning-captured" in rec["flags"]
    assert "".join(rec["trace"]["tokens"]) == rec["response"]
    assert rec["trace"]["reasoning_steps"]
