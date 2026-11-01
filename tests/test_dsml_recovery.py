"""DSML recovery: leaked markup in content becomes OpenAI tool_calls.

The wire formats mirror Libertai/conscio's DSML cases (tests/test_streaming.py
constant DSML_LEAK, test_runtime.py's DsmlToolCallParserTests) plus the
`<｜DSML｜tool_calls>` invoke/parameter form seen in real leaked artifacts.
"""

import json

import pytest

from src.dsml_recovery import (
    StreamRecovery,
    has_dsml_markup,
    recover_chat_completion,
    recover_content,
    recovers_dsml,
)

# Role-token form: the tool name sits after <｜tool_sep｜>, the "function" role
# marker before it.
DSML_ROLE = (
    "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>echo_tool\n"
    '{"input": "hi"}<｜tool_call_end｜><｜tool_calls_end｜>'
)
DSML_FENCED = (
    "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>search_memory\n"
    '```json\n{"query": "vitamin b1"}\n```'
    "<｜tool_call_end｜><｜tool_calls_end｜>"
)
DSML_TRUNCATED = (
    '<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>note_progress\n{"note": "finished reading"}'
)
DSML_XML = (
    "<｜DSML｜tool_calls>\n"
    '<｜DSML｜invoke name="store_fact">\n'
    '<｜DSML｜parameter name="fact" string="true">BBC news</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>"
)


def _body(content: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }
    ).encode()


# --- gating ---------------------------------------------------------------


def test_gate_covers_the_two_deepseek_builds_case_insensitively():
    assert recovers_dsml("deepseek-v4.1-flash")
    assert recovers_dsml("deepseek-v4.1-flash-1017")
    assert recovers_dsml("DeepSeek-V4.1-Flash")
    assert not recovers_dsml("qwen3.6-35b-a3b")


def test_plain_content_is_not_markup():
    assert not has_dsml_markup("Hello, world.")
    assert not has_dsml_markup("")
    assert not has_dsml_markup("I considered using <｜some marker｜> but decided not to.")


# --- parsing (ported conscio cases) ---------------------------------------


@pytest.mark.parametrize(
    ("content", "name", "args"),
    [
        (DSML_ROLE, "echo_tool", {"input": "hi"}),
        (DSML_FENCED, "search_memory", {"query": "vitamin b1"}),
        (DSML_TRUNCATED, "note_progress", {"note": "finished reading"}),
        (DSML_XML, "store_fact", {"fact": "BBC news"}),
    ],
)
def test_dsml_block_becomes_a_tool_call(content, name, args):
    cleaned, calls = recover_content(content)
    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == name
    assert json.loads(calls[0]["function"]["arguments"]) == args


def test_missing_function_name_returns_no_call_but_strips_markup():
    cleaned, calls = recover_content(
        "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
        '{"x": 1}<｜tool_call_end｜><｜tool_calls_end｜>'
    )
    assert calls == []
    assert cleaned == ""


def test_ascii_pipe_variant_is_recovered():
    cleaned, calls = recover_content('<|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>echo_tool{"a": 1}')
    assert cleaned == ""
    assert [call["function"]["name"] for call in calls] == ["echo_tool"]


def test_prose_before_a_leak_is_kept():
    cleaned, calls = recover_content("Sure thing, calling now.\n" + DSML_ROLE)
    assert cleaned == "Sure thing, calling now."
    assert len(calls) == 1


# --- non-streaming body rewrite -------------------------------------------


def test_non_streaming_full_block_becomes_tool_calls():
    body, changed = recover_chat_completion(_body(DSML_ROLE))
    assert changed is True
    choice = json.loads(body)["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "echo_tool"


def test_non_streaming_unparseable_dsml_is_stripped(caplog):
    with caplog.at_level("WARNING", logger="src.dsml_recovery"):
        body, changed = recover_chat_completion(_body("Oops <｜tool_calls_begin｜> not a call"))
    assert changed is True
    message = json.loads(body)["choices"][0]["message"]
    assert message["content"] == "Oops"
    assert "tool_calls" not in message
    assert "DSML markup in response content" in caplog.text


@pytest.mark.parametrize("content", ["Hello, world.", "Multi\nline <b>html</b> stays.", ""])
def test_normal_content_is_byte_identical_non_streaming(content):
    body = _body(content)
    assert recover_chat_completion(body) == (body, False)


def test_native_tool_calls_win_but_markup_is_still_stripped():
    payload = json.loads(_body(DSML_ROLE))
    native = [{"id": "call_native", "type": "function", "function": {"name": "other", "arguments": "{}"}}]
    payload["choices"][0]["message"]["tool_calls"] = native
    body, changed = recover_chat_completion(json.dumps(payload).encode())
    message = json.loads(body)["choices"][0]["message"]
    assert changed is True
    assert message["tool_calls"] == native
    assert message["content"] == ""


# --- streaming ------------------------------------------------------------


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


def _chunk(text=None, finish=None, **extra) -> bytes:
    delta = {"content": text} if text is not None else {}
    obj = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-v4.1-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    obj.update(extra)
    return _sse(obj)


def _run(events: list[bytes], cuts: list[int] | None = None) -> list[bytes]:
    """Feed the concatenated events through the gate, split at byte offsets."""
    raw = b"".join(events)
    cuts = cuts or []
    gate = StreamRecovery()
    out: list[bytes] = []
    position = 0
    for cut in cuts:
        out.extend(gate.feed(raw[position:cut]))
        position = cut
    out.extend(gate.feed(raw[position:]))
    out.extend(gate.finish())
    return out


def _content(out: list[bytes]) -> str:
    text = ""
    for event in out:
        payload = event.removeprefix(b"data: ").strip()
        if payload == b"[DONE]":
            continue
        obj = json.loads(payload)
        for choice in obj.get("choices", []):
            text += choice.get("delta", {}).get("content") or ""
    return text


def _clean_stream() -> list[bytes]:
    return [
        _chunk("Hello streaming world, this is a longer sentence to cross the sniff window."),
        _chunk(" And a second sentence that keeps going."),
        _chunk(None, "stop"),
        b"data: [DONE]\n\n",
    ]


def test_streaming_without_dsml_is_byte_identical():
    events = _clean_stream()
    out = _run(events)
    assert b"".join(out) == b"".join(events)


def test_streaming_dsml_split_across_chunks_is_recovered():
    events = [
        _chunk("<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>echo_tool\n"),
        _chunk('{"input": "hi"}'),
        _chunk("<｜tool_call_end｜><｜tool_calls_end｜>"),
        _chunk(None, "stop"),
        b"data: [DONE]\n\n",
    ]
    raw = b"".join(events)
    # Split on awkward boundaries too: mid-marker, mid-UTF-8, mid-SSE frame.
    for cuts in ([], [7], [1, 60, 61, 200], list(range(1, len(raw), 5))):
        out = _run(events, cuts)
        text = b"".join(out).decode()
        assert "<｜" not in text and "tool_calls_begin" not in text
        assert _content(out) == ""
        assert '"finish_reason":"tool_calls"' in text
        assert text.rstrip().endswith("data: [DONE]")


def test_streaming_recovered_call_has_openai_delta_shape():
    out = _run([_chunk(DSML_ROLE), _chunk(None, "stop"), b"data: [DONE]\n\n"])
    chunks = [json.loads(e.removeprefix(b"data: ")) for e in out if not e.startswith(b"data: [DONE]")]
    call_chunk = next(c for c in chunks if c["choices"][0]["delta"].get("tool_calls"))
    call = call_chunk["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["type"] == "function"
    assert call["function"]["name"] == "echo_tool"
    assert json.loads(call["function"]["arguments"]) == {"input": "hi"}


def test_streaming_prose_prefix_survives_a_later_leak():
    prose = "x" * 80
    out = _run(
        [_chunk(prose), _chunk("abc<"), _chunk("｜tool_calls_begin｜>"), _chunk(None, "stop"), b"data: [DONE]\n\n"]
    )
    text = b"".join(out).decode()
    assert _content(out) == prose + "abc"
    assert "tool_calls_begin" not in text


def test_streaming_usage_and_done_pass_through_after_recovery():
    usage = _sse({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [], "usage": {"total_tokens": 3}})
    out = _run([_chunk(DSML_ROLE), _chunk(None, "stop"), usage, b"data: [DONE]\n\n"])
    assert out[-2:] == [usage, b"data: [DONE]\n\n"]


def test_streaming_native_tool_calls_are_not_buffered_or_rewritten():
    # A stream that carries structured tool_calls needs no recovery; it must be
    # forwarded byte-identically (the gate must not hold it until the end).
    native = _chunk(None)
    native_obj = json.loads(native.removeprefix(b"data: "))
    native_obj["choices"][0]["delta"] = {
        "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    }
    events = [
        b"data: " + json.dumps(native_obj, ensure_ascii=False).encode() + b"\n\n",
        _chunk(None, "tool_calls"),
        b"data: [DONE]\n\n",
    ]
    out = _run(events)
    assert b"".join(out) == b"".join(events)


def test_streaming_no_done_event_still_finishes_cleanly():
    # Some upstreams end without [DONE]; recovery must still emit a final chunk.
    out = _run([_chunk("Hello there, a short clean answer.")])
    text = b"".join(out).decode()
    assert _content(out) == "Hello there, a short clean answer."
    assert '"finish_reason":"stop"' in text


def test_streaming_recovered_leak_without_done_still_terminates():
    # Truncated leak, no final chunk and no [DONE]: the recovered call still gets
    # a closing chunk and a [DONE] terminator.
    out = _run([_chunk(DSML_TRUNCATED)])
    text = b"".join(out).decode()
    assert _content(out) == ""
    assert '"name":"note_progress"' in text
    assert text.count('"finish_reason":"tool_calls"') == 1
    assert text.rstrip().endswith("data: [DONE]")
