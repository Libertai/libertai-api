"""Thinking directives for reasoning models, which disagree on the mechanism."""

# GLM chat templates drive thinking with `reasoning_effort` and ignore `enable_thinking`.
# They recognise only "low" and "high"; every other value, the field's absence included,
# renders as the "Max" default, whose reasoning runs several times longer than "high".
REASONING_EFFORT_MODELS = frozenset({"glm-5.3", "glm-5.3-flash"})

# DeepSeek V4.1 renders `reasoning_effort` as a numeric budget (low 50, high 75, max 100). Its `-thinking`
# variant sends high explicitly: a 3,738-job benchmark (agentic coding, tau-bench, LiveCodeBench, AIME, chat)
# found max no better than high on quality, at up to 1.8x the tokens and the most runaway generations.
# Thinking-off still uses `enable_thinking`. Clients can still ask for "max" themselves.
HIGH_EFFORT_THINKING_MODELS = frozenset({"deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"})


def disable_thinking(model: str, body_json: dict) -> dict:
    """Add the thinking-off directive this model understands. An explicit client value wins."""
    if model.lower() in REASONING_EFFORT_MODELS:
        body_json.setdefault("reasoning_effort", "low")
    else:
        body_json.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
    return body_json


def request_thinking(model: str, body_json: dict) -> dict:
    """Set the thinking level the `-thinking` variant means. An explicit client value wins."""
    if model.lower() in REASONING_EFFORT_MODELS or model.lower() in HIGH_EFFORT_THINKING_MODELS:
        body_json.setdefault("reasoning_effort", "high")
    return body_json
