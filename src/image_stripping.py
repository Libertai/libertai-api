"""Strip image parts from chat requests when the target model has no vision support.

vLLM (and other engines) error out when handed image content for a text-only model, so
we remove image parts before forwarding rather than letting the request crash. This is a
best-effort, silent operation: text is preserved, only the image parts are dropped.

Three request shapes carry images, all handled by the recursive signature match below:
- OpenAI Chat Completions (`v1/chat/completions`): {"type": "image_url", "image_url": {...}}
- OpenAI Responses        (`v1/responses`):        {"type": "input_image", "image_url"|"file_id": ...}
- Anthropic Messages      (`v1/messages`):         {"type": "image", "source": {...}}
  (images can be nested inside `tool_result.content` arrays — recursion covers that)
"""

# Only text/chat endpoints are stripped. Image generation/edit and audio endpoints
# legitimately carry images and must be left untouched.
IMAGE_STRIP_PATHS = frozenset(
    {
        "v1/chat/completions",
        "v1/completions",
        "v1/messages",
        "v1/responses",
    }
)


def _is_image_part(item) -> bool:
    if not isinstance(item, dict):
        return False
    part_type = item.get("type")
    if part_type in ("image_url", "input_image"):
        return True
    # Anthropic image block; guard on `source` to avoid matching unrelated "image" types.
    if part_type == "image" and "source" in item:
        return True
    return False


def _strip(obj):
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            stripped = _strip(value)
            # A content array that held only images becomes empty after stripping; collapse
            # it to an empty string so engines don't choke on an empty-list message content.
            if key == "content" and isinstance(stripped, list) and len(stripped) == 0:
                stripped = ""
            result[key] = stripped
        return result
    if isinstance(obj, list):
        return [_strip(item) for item in obj if not _is_image_part(item)]
    return obj


def strip_images(full_path: str, body_json: dict) -> tuple[dict, bool]:
    """Return (possibly-stripped body, whether anything was removed).

    Caller is responsible for only invoking this when the model lacks vision support.
    """
    if full_path not in IMAGE_STRIP_PATHS:
        return body_json, False
    stripped = _strip(body_json)
    return stripped, stripped != body_json
