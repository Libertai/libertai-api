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

        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram bot token or chat ID not set. Telegram reporting disabled.")
        else:
            self._build_app()

    def _build_app(self) -> None:
        """(Re)build the PTB Application. PTB's Application is single-shot — once
        shut down, it cannot be re-initialized, so we rebuild on each start."""
        self.app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(CommandHandler("status", TelegramReporter.status_command))

    async def start_bot(self) -> None:
        """Start the bot polling for commands. Should only be called on the leader replica."""
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.info("Telegram bot not configured, skipping startup")
            return

        if self._bot_started:
            logger.warning("Telegram bot already started, skipping")
            return

        # Rebuild a fresh Application — a previously-shut-down one cannot be re-initialized.
        if self.app is None:
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

    async def stop_bot(self) -> None:
        """Stop the bot polling. Called when this replica loses leadership."""
        if not self.app or not self._bot_started:
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
    async def status_command(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a health status report when the /status command is received."""
        try:
            # Ensure we have the necessary objects
            if not update.effective_chat or not update.message:
                logger.warning("Received status command with missing chat or message")
                return

            # Generate health report
            message = await TelegramReporter.generate_health_report()

            # Reply to the command with the health report
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Failed to process status command: {e}", exc_info=True)
            if update.message:
                await update.message.reply_text("Error generating status report. Check server logs.")

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
