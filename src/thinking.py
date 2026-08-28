"""Thinking-off directives for reasoning models, which disagree on the mechanism."""

# These models' chat templates have no `enable_thinking`: sending it drops the opening `<think>`
# tag while the model still reasons, so the trace lands in `content`. `reasoning_effort: "none"`
# fails the same way; "low" is the only working off switch (it does not scale).
REASONING_EFFORT_MODELS = frozenset({"glm-5.3-flash"})


def disable_thinking(model: str, body_json: dict) -> dict:
    """Add the thinking-off directive this model understands. An explicit client value wins."""
    if model.lower() in REASONING_EFFORT_MODELS:
        body_json.setdefault("reasoning_effort", "low")
    else:
        body_json.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
    return body_json
