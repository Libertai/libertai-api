from src.thinking import disable_thinking


def test_default_model_gets_enable_thinking_kwarg():
    body = disable_thinking("qwen3.6-35b-a3b", {"model": "qwen3.6-35b-a3b"})
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body


def test_reasoning_effort_model_gets_effort_and_no_kwarg():
    body = disable_thinking("glm-5.3-flash", {"model": "glm-5.3-flash"})
    assert body["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in body


def test_model_name_matched_case_insensitively():
    body = disable_thinking("GLM-5.3-Flash", {"model": "GLM-5.3-Flash"})
    assert body["reasoning_effort"] == "low"


def test_client_supplied_effort_wins():
    body = disable_thinking("glm-5.3-flash", {"reasoning_effort": "high"})
    assert body["reasoning_effort"] == "high"


def test_client_supplied_kwarg_wins():
    body = disable_thinking("qwen3.6-35b-a3b", {"chat_template_kwargs": {"enable_thinking": True}})
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_other_chat_template_kwargs_preserved():
    body = disable_thinking("qwen3.6-35b-a3b", {"chat_template_kwargs": {"clear_thinking": True}})
    assert body["chat_template_kwargs"] == {"clear_thinking": True, "enable_thinking": False}
