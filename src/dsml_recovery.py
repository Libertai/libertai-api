"""Recover leaked DeepSeek DSML tool calls from assistant `content`.

DeepSeek-V4.1-Flash (and the -1017 build) sometimes renders its chat template's
tool-call tokens into the `content` stream instead of returning structured
`tool_calls`. Clients then see raw markup such as

    <｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>echo
    {"input": "hi"}<｜tool_call_end｜><｜tool_calls_end｜>

or the XML-ish variant

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="store_fact">
    <｜DSML｜parameter name="fact" string="true">...</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

and tool calling breaks entirely. The root cause is on the serving side (chat
template / parser), so this module is a stopgap in the proxy: it parses the
markup back into OpenAI-shaped tool calls on the response path and strips the
markup so it never reaches a client.

The parsing logic is ported from Libertai/conscio (`src/conscio/core/tool_loop.py`,
`_parse_dsml_tool_call` and its marker regexes); the streaming gate follows the
same shape as conscio's `_StreamGate` but works on SSE events so `[DONE]`,
usage chunks and every other event pass through untouched.

Limits, documented rather than solved here:
- Only the first choice of a stream is recovered; multi-choice streams are
  forwarded untouched.
- After a DSML marker is detected mid-stream the rest of that message's content
  is withheld (it is either the call's arguments or trailing prose); the
  non-streaming path keeps prose outside a closed block.
- A prose message that literally prints a marker *and* parses as a call is
  indistinguishable from a leak; call names are not checked against the request.
"""

import json
import re
from dataclasses import dataclass

from src.logger import setup_logger

logger = setup_logger(__name__)

# Models whose serving-side parser leaks DSML into content. Adding a model is a
# one-line change here.
DSML_RECOVERY_MODELS = frozenset({"deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"})

# Recovers DeepSeek-style native tool-call markers. Ported from conscio; the
# `｜` is U+FF5C, the ascii `|` variant is accepted too.
_DSML_SEP = re.compile(r"<\s*[｜|]\s*tool[_ ]?sep\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALL_BEGIN = re.compile(r"<\s*[｜|]\s*tool[_ ]?call[_ ]?begin\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALL_END = re.compile(r"<\s*[｜|]\s*tool[_ ]?call[_ ]?end\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALLS_BEGIN = re.compile(r"<\s*[｜|]\s*tool[_ ]?calls[_ ]?begin\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALLS_END = re.compile(r"<\s*[｜|]\s*tool[_ ]?calls[_ ]?end\s*[｜|]\s*>", re.IGNORECASE)
_DSML_MARKER = re.compile(r"<\s*[｜|][^<>]*>")
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

# Any marker that opens a tool-call block: the role-token form and the DSML
# invoke/parameter form. Used as the "is this a leak" test everywhere.
_DSML_HINT = re.compile(
    r"<\s*[｜|]\s*(?:(?:dsml\s*[｜|]?\s*)?tool[_ ]?calls?|(?:dsml\s*[｜|]\s*)?(?:invoke|parameter)\b)",
    re.IGNORECASE,
)

# Whole closed blocks, removed when cleaning content.
_DSML_XML_BLOCK = re.compile(
    r"<\s*[｜|]\s*DSML\s*[｜|]\s*tool_calls\s*>.*?</\s*[｜|]\s*DSML\s*[｜|]\s*tool_calls\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DSML_ROLE_BLOCK = re.compile(
    r"<\s*[｜|]\s*tool[_ ]?calls?[_ ]?begin\s*[｜|]\s*>.*?(?:<\s*[｜|]\s*tool[_ ]?calls?[_ ]?end\s*[｜|]\s*>|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Openers left after closed blocks are gone: a truncated block runs to the end.
_DSML_TRUNCATED_OPENERS = (
    re.compile(r"<\s*[｜|]\s*DSML\s*[｜|]", re.IGNORECASE),
    re.compile(r"<\s*[｜|]\s*tool[_ ]?calls?[_ ]?begin\s*[｜|]", re.IGNORECASE),
)

# DSML parameter/invoke tags (name attribute is order-independent in practice).
_DSML_XML_INVOKE_OPEN = re.compile(
    r"<\s*[｜|]\s*DSML\s*[｜|]\s*invoke\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>",
    re.IGNORECASE,
)
_DSML_XML_INVOKE_CLOSE = re.compile(r"</\s*(?:[｜|]\s*DSML\s*[｜|]\s*)?invoke\s*>", re.IGNORECASE)
_DSML_XML_PARAM = re.compile(
    r"<\s*[｜|]\s*DSML\s*[｜|]\s*parameter\s+name\s*=\s*[\"']([^\"']+)[\"']([^>]*)>"
    r"(.*?)</\s*(?:[｜|]\s*DSML\s*[｜|]\s*)?parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DSML_STRING_ATTR = re.compile(r"string\s*=\s*[\"']true[\"']", re.IGNORECASE)

# Streaming: the gate holds back a short prefix to decide, then only a trailing
# partial opener so a marker split across chunk boundaries is still caught.
_SNIFF_CHARS = 64
_MAX_PARTIAL = 64
_PARTIAL_OPENER = re.compile(r"<\s*(?:[｜|][A-Za-z0-9_ \t｜|]*)?")


@dataclass
class _Call:
    name: str
    args: dict


def recovers_dsml(model: str) -> bool:
    """True when this model's responses may leak DSML tool-call markup."""
    return model.lower() in DSML_RECOVERY_MODELS


def has_dsml_markup(content: str) -> bool:
    """True when content contains a tool-call marker that may need recovery."""
    return bool(content) and bool(_DSML_HINT.search(content))


def recover_content(content: str) -> tuple[str, list[dict]]:
    """Split leaked markup out of a message's content.

    Returns the cleaned content and OpenAI-shaped tool calls (empty when the
    markup contained no parseable call). Markup is stripped either way so raw
    tokens never reach a client; an unparseable block is logged at warning
    level.
    """
    if not has_dsml_markup(content):
        return content, []
    calls = _parse_calls(content)
    if not calls:
        logger.warning(f"DSML markup in response content but no tool call could be parsed: {content[:200]!r}")
    cleaned = _strip_markup(content)
    return cleaned, [_openai_call(call, index) for index, call in enumerate(calls)]


def recover_chat_completion(body: bytes) -> tuple[bytes, bool]:
    """Rewrite a non-streaming chat-completion body, returning (body, changed).

    Only choices whose `message.content` carries DSML are touched. A body with
    no markup is returned byte-identical.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body, False
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        return body, False
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not has_dsml_markup(content):
            continue
        cleaned, calls = recover_content(content)
        message["content"] = cleaned
        # Native tool_calls win; the markup is still stripped from content.
        if calls and not message.get("tool_calls"):
            message["tool_calls"] = calls
            choice["finish_reason"] = "tool_calls"
        changed = True
    if not changed:
        return body, False
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), True


def _parse_calls(content: str) -> list[_Call]:
    xml_calls = _parse_xml_calls(content)
    if xml_calls:
        return xml_calls
    return _parse_role_token_calls(content)


def _parse_role_token_calls(content: str) -> list[_Call]:
    begins = list(_DSML_CALL_BEGIN.finditer(content))
    if begins:
        calls = []
        for index, match in enumerate(begins):
            region_end = begins[index + 1].start() if index + 1 < len(begins) else len(content)
            call = _parse_role_body(content[match.end() : region_end])
            if call is not None:
                calls.append(call)
        return calls
    # Truncated/older form: only the plural wrapper, one call inside.
    wrapper = _DSML_CALLS_BEGIN.search(content)
    if wrapper:
        call = _parse_role_body(content[wrapper.end() :])
        return [call] if call is not None else []
    return []


def _parse_role_body(region: str) -> _Call | None:
    """Ported from conscio's `_parse_dsml_tool_call`: `function<sep>name<args>`."""
    end_match = _DSML_CALL_END.search(region) or _DSML_CALLS_END.search(region)
    if end_match:
        region = region[: end_match.start()]
    sep_match = _DSML_SEP.search(region)
    # The role token ("function") sits before <｜tool_sep｜>; the name after it.
    name_and_args = region[sep_match.end() :] if sep_match else region
    fence = _JSON_FENCE.search(name_and_args)
    if fence:
        prefix = name_and_args[: fence.start()].strip()
        args_raw = fence.group(1)
    else:
        brace = name_and_args.find("{")
        if brace == -1:
            return None
        prefix = name_and_args[:brace].strip()
        args_raw = _extract_balanced_json(name_and_args, brace)
        if args_raw is None:
            return None
    prefix = _DSML_MARKER.sub("\n", prefix)
    name = next((line.strip().strip("`").strip() for line in prefix.splitlines() if line.strip()), "")
    if not name:
        return None
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        args = {"input": args_raw}
    if not isinstance(args, dict):
        args = {"input": args}
    return _Call(name=name, args=args)


def _parse_xml_calls(content: str) -> list[_Call]:
    """Parse the `<｜DSML｜invoke>` / `<｜DSML｜parameter>` form."""
    calls: list[_Call] = []
    position = 0
    while True:
        match = _DSML_XML_INVOKE_OPEN.search(content, position)
        if match is None:
            break
        close = _DSML_XML_INVOKE_CLOSE.search(content, match.end())
        body_end = close.start() if close else len(content)
        body = content[match.end() : body_end]
        args: dict = {}
        for param in _DSML_XML_PARAM.finditer(body):
            value = param.group(3).strip()
            if not _DSML_STRING_ATTR.search(param.group(2)):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            args[param.group(1)] = value
        name = match.group(1).strip()
        if name:
            calls.append(_Call(name=name, args=args))
        position = close.end() if close else len(content)
    return calls


def _extract_balanced_json(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_markup(content: str) -> str:
    text = _DSML_XML_BLOCK.sub("", content)
    text = _DSML_ROLE_BLOCK.sub("", text)
    # A truncated block has no end marker: drop from the earliest leftover opener.
    cut: int | None = None
    for opener in _DSML_TRUNCATED_OPENERS:
        match = opener.search(text)
        if match is not None and (cut is None or match.start() < cut):
            cut = match.start()
    if cut is not None:
        text = text[:cut]
    text = _DSML_MARKER.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _openai_call(call: _Call, index: int) -> dict:
    arguments = json.dumps(
        call.args,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "id": f"call_dsml_{index}",
        "type": "function",
        "function": {"name": call.name, "arguments": arguments},
    }


class StreamRecovery:
    """Recover DSML calls from an SSE byte stream without buffering it whole.

    Feed raw upstream chunks to `feed` and forward the returned events; call
    `finish` once the upstream stream ends. While the first `_SNIFF_CHARS` of
    content are undecided, whole raw events are held so a clean stream is
    replayed byte-identically. After that, content flows through except for a
    short trailing partial opener (so a marker split across chunks is caught).
    From a detected marker on, content is withheld and accumulated for parsing;
    `[DONE]`, usage and non-content events are held only until the end so the
    recovered `tool_calls` delta can precede the final chunk.
    """

    def __init__(self) -> None:
        self._buffer = b""
        # Raw content blocks held during the sniff window, released (or dropped)
        # when the window decides. Non-content events never wait on the window.
        self._held: list[bytes] = []
        # Final/[DONE] events seen before the window decides; replayed after the
        # held content so a clean stream keeps its order.
        self._tail: list[bytes] = []
        self._sniff = ""
        self._parse = ""
        self._carry = ""
        self._decided = False
        self._suppressed = False
        self._disabled = False
        self._meta: dict = {}
        self._final_block: bytes | None = None
        self._usage_blocks: list[bytes] = []
        self._done_block: bytes | None = None
        # Whether the upstream has already produced a terminating chunk and/or
        # `[DONE]`; used by `finish` to avoid emitting a duplicate terminator.
        self._seen_final = False
        self._seen_done = False

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume a raw upstream chunk and return events to forward."""
        if self._disabled:
            return [chunk]
        self._buffer += chunk
        out: list[bytes] = []
        while True:
            block, self._buffer = _pop_event(self._buffer)
            if block is None:
                break
            out.extend(self._handle_block(block))
        return out

    def finish(self) -> list[bytes]:
        """Flush at end of upstream stream."""
        out: list[bytes] = []
        if self._buffer:
            # A final event with no trailing blank line; route it like any other.
            out.extend(self._handle_block(self._buffer))
            self._buffer = b""
        if not self._decided:
            self._decide()
            if not self._suppressed:
                # Nothing was held back as a leak: replay verbatim, supplying a
                # terminator only if the upstream never sent one.
                out.extend([*self._held, *self._tail])
                self._held, self._tail = [], []
                if not self._seen_final:
                    out.append(self._final_event(None))
                if not self._seen_done:
                    out.append(b"data: [DONE]\n\n")
                return out
            out.extend(self._prefix_event())
        self._held = []
        if self._suppressed:
            for block in self._tail:
                payload = _event_payload(block)
                if payload is not None and payload.strip() == b"[DONE]":
                    self._done_block = block
                elif self._final_block is None:
                    self._final_block = block
            self._tail = []
            _, calls = recover_content(self._parse)
            if calls:
                out.append(self._event(self._tool_call_chunk(calls)))
                out.append(self._final_event("tool_calls"))
            else:
                out.append(self._final_event(None))
            out.extend(self._usage_blocks)
            out.append(self._done_block if self._done_block is not None else b"data: [DONE]\n\n")
        else:
            out.extend([*self._tail, *self._flush_carry()])
            self._tail = []
            # A short clean stream that ended without a final/[DONE] still needs
            # a terminator, or clients that wait on one hang.
            if not self._seen_final:
                out.append(self._final_event(None))
            if not self._seen_done:
                out.append(b"data: [DONE]\n\n")
        return out

    def _handle_block(self, block: bytes) -> list[bytes]:
        payload = _event_payload(block)
        if payload is None:
            return [block]
        if payload.strip() == b"[DONE]":
            self._seen_done = True
            if self._suppressed:
                self._done_block = block
            elif not self._decided:
                self._tail.append(block)
            else:
                return [block]
            return []
        try:
            obj = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return [block]
        if not isinstance(obj, dict):
            return [block]
        self._remember_meta(obj)
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            if isinstance(obj.get("usage"), dict) and self._suppressed:
                self._usage_blocks.append(block)
                return []
            return [block]
        # Native structured calls mean the serving parser worked; stop gating
        # and let the rest of the stream through untouched.
        if any(
            isinstance(choice, dict) and isinstance(choice.get("delta"), dict) and choice["delta"].get("tool_calls")
            for choice in choices
        ):
            return self._release(block)
        content_choices = [
            choice
            for choice in choices
            if isinstance(choice, dict)
            and isinstance(choice.get("delta"), dict)
            and isinstance(choice["delta"].get("content"), str)
            and choice["delta"]["content"] != ""
        ]
        if len(content_choices) > 1:
            # Multi-choice recovery would need a gate per choice; forward the
            # stream untouched rather than corrupt it.
            return self._release(block)
        if content_choices:
            return self._content_event(block, obj, content_choices[0]["delta"]["content"])
        has_finish = any(isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices)
        if has_finish:
            self._seen_final = True
            if not self._decided:
                self._tail.append(block)
                return []
            if self._suppressed:
                self._final_block = block
                return []
            return [*self._flush_carry(), block]
        return [block]

    def _release(self, block: bytes) -> list[bytes]:
        """Stop gating: flush anything held, then hand the stream through raw."""
        self._disabled = True
        self._suppressed = False
        self._parse = ""
        self._carry = ""
        held, self._held = self._held, []
        usage, self._usage_blocks = self._usage_blocks, []
        done, self._done_block = self._done_block, None
        # During the sniff window content arrived in order, so held blocks
        # precede the current one; other deferred events follow it.
        return [*held, block, *usage, *([done] if done is not None else [])]

    def _flush_carry(self) -> list[bytes]:
        if not self._carry:
            return []
        text, self._carry = self._carry, ""
        return [self._event(self._content_chunk(text))]

    def _content_event(self, block: bytes, obj: dict, text: str) -> list[bytes]:
        if not self._decided:
            self._held.append(block)
            self._sniff += text
            # Decide only once the window cannot end inside a partial marker
            # opener; otherwise a marker split exactly at the boundary would be
            # replayed raw and then missed.
            if len(self._sniff) < _SNIFF_CHARS or _partial_start(self._sniff) != len(self._sniff):
                return []
            self._decide()
            held, self._held = self._held, []
            if self._suppressed:
                # The held blocks are the leak's own prefix; re-emit only the
                # clean prose that preceded it.
                return self._prefix_event()
            return held
        if self._suppressed:
            self._parse += text
            return []
        emitted = self._feed_content(text)
        if self._suppressed:
            return self._prefix_chunk(emitted)
        if emitted == text and not self._carry:
            return [block]  # untouched: byte-identical pass-through
        if not emitted:
            return []
        return [self._event(self._content_chunk(emitted))]

    def _feed_content(self, text: str) -> str:
        """Gate one content delta once sniffing is done; may switch to suppression."""
        combined = self._carry + text
        self._carry = ""
        hint = _DSML_HINT.search(combined)
        if hint is not None:
            self._suppressed = True
            self._parse = combined[hint.start() :]
            return combined[: hint.start()]
        cut = _partial_start(combined)
        self._carry = combined[cut:]
        return combined[:cut]

    def _decide(self) -> None:
        self._decided = True
        hint = _DSML_HINT.search(self._sniff)
        if hint is not None:
            self._suppressed = True
            self._parse = self._sniff[hint.start() :]

    def _prefix_event(self) -> list[bytes]:
        """Emit the clean prose that preceded a detected leak, if any."""
        hint = _DSML_HINT.search(self._sniff)
        prefix = self._sniff[: hint.start()] if hint is not None else self._sniff
        return self._prefix_chunk(prefix)

    def _prefix_chunk(self, text: str) -> list[bytes]:
        if not text:
            return []
        return [self._event(self._content_chunk(text))]

    def _remember_meta(self, obj: dict) -> None:
        for key in ("id", "object", "created", "model", "system_fingerprint"):
            if key in obj:
                self._meta[key] = obj[key]

    def _content_chunk(self, text: str) -> dict:
        chunk = self._chunk_base()
        chunk["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
        return chunk

    def _tool_call_chunk(self, calls: list[dict]) -> dict:
        chunk = self._chunk_base()
        delta_calls = [
            {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": call["function"],
            }
            for index, call in enumerate(calls)
        ]
        chunk["choices"] = [{"index": 0, "delta": {"tool_calls": delta_calls}, "finish_reason": None}]
        return chunk

    def _chunk_base(self) -> dict:
        base = {
            "id": self._meta.get("id", "chatcmpl-dsml"),
            "object": "chat.completion.chunk",
            "created": self._meta.get("created", 0),
            "model": self._meta.get("model", ""),
        }
        if "system_fingerprint" in self._meta:
            base["system_fingerprint"] = self._meta["system_fingerprint"]
        return base

    def _final_event(self, finish_reason: str | None) -> bytes:
        """Emit the held final chunk, optionally replacing its finish_reason."""
        if self._final_block is not None:
            block = self._final_block
            self._final_block = None
            payload = _event_payload(block)
            if finish_reason is not None and payload is not None:
                try:
                    obj = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    obj = None
                if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                    for choice in obj["choices"]:
                        if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                            choice["finish_reason"] = finish_reason
                    return self._event(obj)
            return block
        chunk = self._chunk_base()
        chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}]
        return self._event(chunk)

    @staticmethod
    def _event(obj: dict) -> bytes:
        return b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"


def _pop_event(buffer: bytes) -> tuple[bytes | None, bytes]:
    """Split off one complete SSE event, keeping the remainder buffered."""
    lf = buffer.find(b"\n\n")
    crlf = buffer.find(b"\r\n\r\n")
    if lf == -1 and crlf == -1:
        return None, buffer
    if crlf != -1 and (lf == -1 or crlf < lf):
        return buffer[: crlf + 4], buffer[crlf + 4 :]
    return buffer[: lf + 2], buffer[lf + 2 :]


def _event_payload(block: bytes) -> bytes | None:
    """Return the concatenated `data:` payload of an SSE event, if any."""
    lines = block.split(b"\n")
    data = [line[5:].lstrip(b" ") for line in lines if line.startswith(b"data:")]
    if not data:
        return None
    return b"\n".join(data)


def _partial_start(text: str) -> int:
    """Index of a trailing suffix that may still grow into a DSML marker opener."""
    limit = max(0, len(text) - _MAX_PARTIAL)
    index = text.rfind("<", limit)
    if index == -1:
        return len(text)
    tail = text[index:]
    if ">" in tail:
        return len(text)
    return index if _PARTIAL_OPENER.fullmatch(tail) else len(text)
