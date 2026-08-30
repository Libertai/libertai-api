"""Thinking-off directives for reasoning models, which disagree on the mechanism."""

# GLM chat templates drive thinking with `reasoning_effort` and ignore `enable_thinking`.
# They recognise only "low" and "high"; every other value, the field's absence included,
# renders as the "Max" default, so there is nothing to send to turn thinking back up.
REASONING_EFFORT_MODELS = frozenset({"glm-5.3", "glm-5.3-flash"})


def disable_thinking(model: str, body_json: dict) -> dict:
    """Add the thinking-off directive this model understands. An explicit client value wins."""
    if model.lower() in REASONING_EFFORT_MODELS:
        body_json.setdefault("reasoning_effort", "low")
    else:
        body_json.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
    return body_json
