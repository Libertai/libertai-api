import logging
from datetime import datetime

from telegram import Bot
from telegram.constants import ParseMode

from src.config import config
from src.health import server_health_monitor

logger = logging.getLogger(__name__)


class TelegramReporter:
    def __init__(self) -> None:
        """Initialize the Telegram reporter."""
        self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN) if config.TELEGRAM_BOT_TOKEN else None

        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram bot token or chat ID not set. Telegram reporting disabled.")

    async def send_health_report(self) -> None:
        """
        Send a health report to the Telegram channel, but only if some URLs are down.
        Show exactly which URLs are down per model.
        """
        if not self.bot or not config.TELEGRAM_CHAT_ID:
            return

        try:
            # Get health information
            healthy_model_urls = server_health_monitor.get_healthy_model_urls()

            # Find unhealthy URLs
            unhealthy_urls_by_model = {}
            total_unhealthy = 0
            total_urls = 0

            for model, urls in server_health_monitor.model_urls.items():
                if not urls:
                    continue

                healthy = set(healthy_model_urls.get(model, []))
                # Get URLs that are in the model_urls but not in healthy_urls
                unhealthy = [url for url in urls if url not in healthy]

                if unhealthy:
                    unhealthy_urls_by_model[model] = unhealthy
                    total_unhealthy += len(unhealthy)

                total_urls += len(urls)

            # Only send message if there are unhealthy URLs
            if len(unhealthy_urls_by_model.keys()) == 0 or total_urls == 0:
                return

            # Create the message
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"🚨 *LibertAI Health Alert* ({now})\n\n"
            message += f"*{total_unhealthy} of {total_urls} servers are DOWN*\n\n"

            # List unhealthy URLs by model
            for model, urls in unhealthy_urls_by_model.items():
                message += f"*Model: {model}*\n"
                for url in urls:
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
            logger.info(f"Health alert sent to Telegram: {total_unhealthy} servers are down")

        except Exception as e:
            logger.error(f"Failed to send Telegram health report: {e}")


telegram_reporter = TelegramReporter()
