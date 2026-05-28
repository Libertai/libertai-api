"""Telegram helpers — pure functions and handlers shared by the bot worker.

Lifecycle (Application building, polling, alerts loop) lives in src/bot.py.
The web `api` replicas no longer import this module; only the bot worker does."""

from datetime import datetime
from typing import Awaitable, Callable

from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError
from telegram.ext import ContextTypes

from src.config import config
from src.health import server_health_monitor
from src.logger import setup_logger

logger = setup_logger(__name__)

# Module-level ``Bot`` for one-shot sends from the web replicas (no polling).
# Used only by ``send_message`` below for rare critical alerts; bot-worker code
# uses its own ``Application.bot`` and never touches this.
_bot: Bot | None = None


def _get_bot() -> Bot | None:
    global _bot
    if _bot is None and config.TELEGRAM_BOT_TOKEN:
        _bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    return _bot


async def send_message(text: str) -> None:
    """One-shot Telegram send to the configured chat — for critical alerts from
    the web replicas (e.g. payment-settled-but-credit-failed). No polling, no
    Application; retry on transient NetworkError handles stale-keepalive sockets."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured; cannot send message")
        return
    bot = _get_bot()
    if bot is None:
        return
    try:
        kwargs: dict = {"chat_id": config.TELEGRAM_CHAT_ID, "text": text}
        if config.TELEGRAM_TOPIC_ID:
            kwargs["message_thread_id"] = int(config.TELEGRAM_TOPIC_ID)
        await _send_with_retry(lambda: bot.send_message(**kwargs))
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)


def _classify_servers(model: str, urls: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Classify servers into (loaded, available, down) for a given model."""
    healthy = set(server_health_monitor.healthy_model_urls.get(model, []))
    capable = set(server_health_monitor.capable_model_urls.get(model, []))
    loaded = [url for url in urls if url in healthy]
    available = [url for url in urls if url in capable]
    down = [url for url in urls if url not in healthy and url not in capable]
    return loaded, available, down


def generate_health_report() -> str:
    """Build the /status / periodic-alert health report (Markdown)."""
    total_down = 0
    total_urls = 0
    for model, urls in server_health_monitor.model_urls.items():
        if not urls:
            continue
        _loaded, _available, down = _classify_servers(model, urls)
        total_down += len(down)
        total_urls += len(urls)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if total_down == 0 and total_urls > 0:
        message = f"✅ *LibertAI Health Report* ({now})\n\n*All servers are UP*\n\n"
    else:
        message = f"🚨 *LibertAI Health Report* ({now})\n\n*{total_down} of {total_urls} servers are DOWN*\n\n"

    for model, urls in server_health_monitor.model_urls.items():
        if not urls:
            continue
        loaded, available, down = _classify_servers(model, urls)
        message += f"*Model: {model}*\n"
        if loaded:
            message += f"✅ Loaded ({len(loaded)}/{len(urls)}):\n"
            for url in loaded:
                message += f"- `{url}`\n"
        if available:
            message += f"🔄 Available ({len(available)}/{len(urls)}):\n"
            for url in available:
                message += f"- `{url}`\n"
        if down:
            message += f"❌ Down ({len(down)}/{len(urls)}):\n"
            for url in down:
                message += f"- `{url}`\n"
        message += "\n"
    return message


async def _send_with_retry(send: Callable[[], Awaitable[object]]) -> object:
    """Run a Telegram send coroutine; retry once on NetworkError.

    A long-lived PTB ``Bot`` keeps an ``httpx.AsyncClient`` with pooled keep-alive
    connections; over time Telegram's edge can drop them while they sit idle in
    the pool. The next send grabs a dead socket and httpx raises
    ``RemoteProtocolError``, which PTB wraps as ``NetworkError``. Retrying once
    forces httpx to take a fresh connection, which is the canonical mitigation
    for this class of transient failure."""
    try:
        return await send()
    except NetworkError as e:
        logger.warning(f"Telegram send NetworkError ({e!r}); retrying once")
        return await send()


async def status_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply with the health report. Always replies once invoked: report on
    success, error text on failure. Falls back to plain text only on
    ``BadRequest`` (content/parse problem) — transient ``NetworkError`` is
    retried so the user still gets formatted output."""
    if not update.effective_chat or not update.message:
        logger.warning("Received status command with missing chat or message")
        return

    message = update.message  # narrowed for type-checker
    logger.info(f"/status received (chat={update.effective_chat.id}, thread={message.message_thread_id})")

    try:
        text = generate_health_report()
    except Exception as e:
        logger.error(f"Failed to generate status report: {e}", exc_info=True)
        # Bind err out of the except scope so the lambda's closure is well-defined.
        err = f"Error generating status report: {e}"
        await _send_with_retry(lambda: message.reply_text(err))
        return

    try:
        await _send_with_retry(lambda: message.reply_text(text, parse_mode=ParseMode.MARKDOWN))
    except BadRequest as e:
        # Markdown parse problem — send as plain so the report still lands.
        logger.warning(f"Markdown reply failed ({e}); resending as plain text")
        await _send_with_retry(lambda: message.reply_text(text))


async def on_error(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all so handler exceptions surface in logs instead of being dropped."""
    logger.error(f"Unhandled error in Telegram handler: {context.error}", exc_info=context.error)


async def send_health_report(bot: Bot) -> None:
    """Periodic alert: send a report iff some servers are truly down.
    Available (capable but not loaded) servers are NOT considered down."""
    if not config.TELEGRAM_CHAT_ID:
        return

    try:
        down_by_model: dict[str, list[str]] = {}
        total_down = 0
        total_urls = 0
        for model, urls in server_health_monitor.model_urls.items():
            if not urls:
                continue
            _loaded, _available, down = _classify_servers(model, urls)
            total_urls += len(urls)
            if down:
                down_by_model[model] = down
                total_down += len(down)

        if total_down == 0 or total_urls == 0:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🚨 *LibertAI Health Alert* ({now})\n\n*{total_down} of {total_urls} servers are DOWN*\n\n"
        for model, down_urls in down_by_model.items():
            all_urls = server_health_monitor.model_urls[model]
            message += f"*Model: {model}* ({len(down_urls)} down / {len(all_urls)} total)\n"
            for url in down_urls:
                message += f"- `{url}`\n"
            message += "\n"

        kwargs: dict = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": ParseMode.MARKDOWN}
        if config.TELEGRAM_TOPIC_ID:
            kwargs["message_thread_id"] = int(config.TELEGRAM_TOPIC_ID)
        await _send_with_retry(lambda: bot.send_message(**kwargs))
        logger.info(f"Health alert sent to Telegram: {total_down} servers are down")
    except Exception as e:
        logger.error(f"Failed to send Telegram health report: {e}", exc_info=True)
