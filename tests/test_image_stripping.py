from src.image_stripping import strip_images


def test_openai_chat_image_url_stripped():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
    }
    out, stripped = strip_images("v1/chat/completions", body)
    assert stripped is True
    assert out["messages"][0]["content"] == [{"type": "text", "text": "what is this?"}]


def test_openai_responses_input_image_stripped():
    body = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hi"},
                    {"type": "input_image", "image_url": "https://x/y.jpg"},
                    {"type": "input_image", "file_id": "file-1"},
                ],
            }
        ]
    }
    out, stripped = strip_images("v1/responses", body)
    assert stripped is True
    assert out["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]


def test_anthropic_image_block_stripped():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                ],
            }
        ]
    }
    out, stripped = strip_images("v1/messages", body)
    assert stripped is True
    assert out["messages"][0]["content"] == [{"type": "text", "text": "look"}]


def test_anthropic_nested_tool_result_image_stripped():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "result"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                        ],
                    }
                ],
            }
        ]
    }
    out, stripped = strip_images("v1/messages", body)
    assert stripped is True
    tool_result = out["messages"][0]["content"][0]
    assert tool_result["content"] == [{"type": "text", "text": "result"}]


def test_content_collapses_to_empty_string_when_only_image():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}
        ]
    }
    out, stripped = strip_images("v1/chat/completions", body)
    assert stripped is True
    assert out["messages"][0]["content"] == ""


def test_plain_string_content_untouched():
    body = {"messages": [{"role": "user", "content": "just text"}]}
    out, stripped = strip_images("v1/chat/completions", body)
    assert stripped is False
    assert out == body


def test_no_images_unchanged():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    out, stripped = strip_images("v1/chat/completions", body)
    assert stripped is False
    assert out == body


def test_image_generation_path_not_stripped():
    # Image generation legitimately carries images and must never be stripped.
    body = {"image_url": {"url": "data:image/png;base64,AA"}, "type": "image_url"}
    out, stripped = strip_images("v1/images/edits", body)
    assert stripped is False
    assert out == body
