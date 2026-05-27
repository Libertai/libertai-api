import asyncio
from datetime import datetime

from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from src.config import config
from src.health import server_health_monitor
from src.logger import setup_logger

logger = setup_logger(__name__)


class TelegramReporter:
    def __init__(self) -> None:
        """Initialize the Telegram reporter."""
        self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN) if config.TELEGRAM_BOT_TOKEN else None
        self.app: Application | None = None
        self._bot_started = False
        # Serializes start/stop/ensure so the on_acquire callback and the
        # supervisor loop can't drive the Application lifecycle concurrently.
        self._lock = asyncio.Lock()

        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram bot token or chat ID not set. Telegram reporting disabled.")
        else:
            self._build_app()

    def _build_app(self) -> None:
        """(Re)build the PTB Application. PTB's Application is single-shot — once
        shut down, it cannot be re-initialized, so we rebuild on each start."""
        self.app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(CommandHandler("status", TelegramReporter.status_command))
        self.app.add_error_handler(TelegramReporter._on_error)
        # Fail loud if a PTB upgrade ever removes the internal polling-task slot our
        # liveness check relies on — otherwise is_polling() silently degrades to the
        # unreliable running flag (the original bug).
        if self.app.updater is not None and not hasattr(self.app.updater, self._POLLING_TASK_ATTR):
            logger.warning(
                "PTB Updater lacks %s; poller liveness degraded to the running flag", self._POLLING_TASK_ATTR
            )

    # PTB (pinned 22.x) runs polling in this name-mangled background task on the
    # Updater. ``updater.running`` is only a flag set True on start and reset
    # solely by a clean ``stop()`` — it stays True even after the task dies
    # abnormally (cancellation, InvalidToken, unexpected error). So the task's
    # own state, not the flag, is the real liveness signal.
    _POLLING_TASK_ATTR = "_Updater__polling_task"

    def _polling_task(self) -> "asyncio.Task | None":
        updater = self.app.updater if self.app else None
        task = getattr(updater, self._POLLING_TASK_ATTR, None) if updater else None
        return task if isinstance(task, asyncio.Task) else None

    def is_polling(self) -> bool:
        """Whether this replica has a genuinely live command-polling loop.

        Requires the dispatch processor (``app.running``) AND the update fetcher
        flag (``updater.running``) AND — crucially — that the fetcher's background
        task is still running. Trusting ``updater.running`` alone is what let a
        dead poller masquerade as healthy: the flag stayed True after the task
        died, so the supervisor never restarted it and updates piled up unread."""
        if not (self.app and self._bot_started and self.app.running and self.app.updater and self.app.updater.running):
            return False
        task = self._polling_task()
        if task is None:
            # Attribute missing (PTB internals changed) — fall back to the flag
            # rather than restart-loop forever. The <23 pin guards this.
            return True
        return not task.done()

    async def start_bot(self) -> None:
        """Start the bot polling for commands. Should only be called on the leader replica.

        Idempotent and self-healing: a no-op when already polling, otherwise it
        discards any stale Application, builds a fresh one, and — crucially — leaves
        clean state if startup fails so the supervisor can retry on its next tick."""
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.info("Telegram bot not configured, skipping startup")
            return

        async with self._lock:
            if self.is_polling():
                return

            # Discard any partial/stale instance first — a previously-initialized
            # Application cannot be re-initialized.
            await self._teardown_app()
            self._build_app()

            try:
                assert self.app is not None
                await self.app.initialize()
                await self.app.start()
                if self.app.updater:
                    await self.app.updater.start_polling(drop_pending_updates=True)

                self._bot_started = True
                logger.info("Telegram bot started and listening for commands")
            except Exception as e:
                logger.error(f"Error starting Telegram bot: {e}", exc_info=True)
                # Leave clean state so a later start_bot() rebuilds from scratch.
                await self._teardown_app()

    async def ensure_polling(self) -> None:
        """Supervisor entry point: (re)start polling if it isn't running. Heals a
        poller that failed to start or died while this replica remained leader."""
        if self.is_polling():
            return
        self._log_dead_polling_task()
        logger.warning("Telegram bot not polling while leader; (re)starting")
        await self.start_bot()

    def _log_dead_polling_task(self) -> None:
        """If the polling task died while leaving ``updater.running`` stale-True,
        surface why — this is the trigger we couldn't see before."""
        task = self._polling_task()
        if task is None or not task.done():
            return
        if task.cancelled():
            logger.error("Telegram polling task was cancelled but updater.running stayed True")
            return
        exc = task.exception()  # safe: task is done and not cancelled
        if exc is not None:
            logger.error(f"Telegram polling task died: {exc!r} (updater.running stayed True)", exc_info=exc)

    async def stop_bot(self) -> None:
        """Stop the bot polling. Called when this replica loses leadership."""
        async with self._lock:
            await self._teardown_app()

    async def _teardown_app(self) -> None:
        """Stop polling and shut down the Application if present. Caller must hold
        ``self._lock`` (or run before any concurrent lifecycle access, e.g. startup)."""
        if self.app is None:
            return

        try:
            if self.app.updater and self.app.updater.running:
                await self.app.updater.stop()
            if self.app.running:
                await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}", exc_info=True)
        finally:
            # Discard the shut-down instance; a future start_bot will rebuild.
            self.app = None
            self._bot_started = False

    @staticmethod
    def _classify_servers(model: str, urls: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Classify servers into loaded, available, and down for a given model.

        Returns:
            Tuple of (loaded_urls, available_urls, down_urls)
        """
        healthy = set(server_health_monitor.healthy_model_urls.get(model, []))
        capable = set(server_health_monitor.capable_model_urls.get(model, []))
        loaded = [url for url in urls if url in healthy]
        available = [url for url in urls if url in capable]
        down = [url for url in urls if url not in healthy and url not in capable]
        return loaded, available, down

    @staticmethod
    async def generate_health_report() -> str:
        """Generate a complete health report showing loaded, available, and down servers."""
        total_down = 0
        total_urls = 0

        for model, urls in server_health_monitor.model_urls.items():
            if not urls:
                continue
            _loaded, _available, down = TelegramReporter._classify_servers(model, urls)
            total_down += len(down)
            total_urls += len(urls)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if total_down == 0 and total_urls > 0:
            message = f"✅ *LibertAI Health Report* ({now})\n\n"
            message += "*All servers are UP*\n\n"
        else:
            message = f"🚨 *LibertAI Health Report* ({now})\n\n"
            message += f"*{total_down} of {total_urls} servers are DOWN*\n\n"

        for model, urls in server_health_monitor.model_urls.items():
            if not urls:
                continue

            loaded, available, down = TelegramReporter._classify_servers(model, urls)
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

    @staticmethod
    async def _on_error(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Catch-all so handler exceptions surface in logs instead of being silently dropped."""
        logger.error(f"Unhandled error in Telegram handler: {context.error}", exc_info=context.error)

    @staticmethod
    async def status_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with a health report when /status is received.

        Once this handler runs it ALWAYS replies — report on success, the error
        text on failure — so a silent "read but no response" can only mean the
        update never reached the handler (dispatch/poller layer), not a swallowed
        exception here. The Markdown reply falls back to plain text so a parse
        error never eats the response."""
        if not update.effective_chat or not update.message:
            logger.warning("Received status command with missing chat or message")
            return

        thread_id = update.message.message_thread_id
        logger.info(f"/status received (chat={update.effective_chat.id}, thread={thread_id})")

        try:
            message = await TelegramReporter.generate_health_report()
        except Exception as e:
            logger.error(f"Failed to generate status report: {e}", exc_info=True)
            await update.message.reply_text(f"Error generating status report: {e}")
            return

        try:
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            # Most likely a Markdown parse error — resend as plain text so the
            # report still gets through, and record what tripped the parser.
            logger.warning(f"Markdown reply failed ({e}); resending as plain text")
            await update.message.reply_text(message)

    async def send_message(self, text: str) -> None:
        """Send a plain text message to the configured Telegram channel."""
        if not self.bot or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured, cannot send message")
            return

        try:
            kwargs: dict = {"chat_id": config.TELEGRAM_CHAT_ID, "text": text}
            if config.TELEGRAM_TOPIC_ID:
                kwargs["message_thread_id"] = int(config.TELEGRAM_TOPIC_ID)
            await self.bot.send_message(**kwargs)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}", exc_info=True)

    async def send_health_report(self) -> None:
        """
        Send a health report to the Telegram channel, but only if some servers are truly down.
        Available (capable but not loaded) servers are NOT considered down.
        """
        if not self.bot or not config.TELEGRAM_CHAT_ID:
            return

        try:
            down_by_model: dict[str, list[str]] = {}
            total_down = 0
            total_urls = 0

            for model, urls in server_health_monitor.model_urls.items():
                if not urls:
                    continue

                _loaded, _available, down = self._classify_servers(model, urls)
                total_urls += len(urls)

                if down:
                    down_by_model[model] = down
                    total_down += len(down)

            # Only send message if there are truly down servers
            if total_down == 0 or total_urls == 0:
                return

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"🚨 *LibertAI Health Alert* ({now})\n\n"
            message += f"*{total_down} of {total_urls} servers are DOWN*\n\n"

            for model, down_urls in down_by_model.items():
                all_urls = server_health_monitor.model_urls[model]
                message += f"*Model: {model}* ({len(down_urls)} down / {len(all_urls)} total)\n"
                for url in down_urls:
                    message += f"- `{url}`\n"
                message += "\n"

            # Send the message with topic ID if provided
            if config.TELEGRAM_TOPIC_ID:
                await self.bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    message_thread_id=int(config.TELEGRAM_TOPIC_ID),
                )
            else:
                await self.bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN
                )
            logger.info(f"Health alert sent to Telegram: {total_down} servers are down")

        except Exception as e:
            logger.error(f"Failed to send Telegram health report: {e}", exc_info=True)


telegram_reporter = TelegramReporter()
