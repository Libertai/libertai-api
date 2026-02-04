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
        self.app = None
        self._bot_started = False

        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram bot token or chat ID not set. Telegram reporting disabled.")
        else:
            # Set up the application with the bot token
            self.app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

            # Add command handler
            self.app.add_handler(CommandHandler("status", TelegramReporter.status_command))

    async def start_bot(self) -> None:
        """Start the bot polling for commands. Should only be called once per application."""
        if not self.app:
            logger.info("Telegram bot not configured, skipping startup")
            return

        if self._bot_started:
            logger.warning("Telegram bot already started, skipping")
            return

        try:
            # Initialize bot
            await self.app.initialize()

            # Start the bot
            await self.app.start()

            # Start polling for updates
            if self.app.updater:
                await self.app.updater.start_polling(drop_pending_updates=True)

            self._bot_started = True
            logger.info("Telegram bot started and listening for commands")
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}")

    @staticmethod
    async def generate_health_report() -> str:
        """Generate a complete health report including both healthy and unhealthy servers."""
        # Get health information
        healthy_model_urls = server_health_monitor.get_healthy_model_urls()

        # Analyze health data
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

        # Create the message
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if total_unhealthy == 0 and total_urls > 0:
            message = f"✅ *LibertAI Health Report* ({now})\n\n"
            message += f"*All {total_urls} servers are UP*\n\n"
        else:
            message = f"🚨 *LibertAI Health Report* ({now})\n\n"
            message += f"*{total_unhealthy} of {total_urls} servers are DOWN*\n\n"

        # List all model statuses, including healthy ones
        for model, urls in server_health_monitor.model_urls.items():
            if not urls:
                continue

            healthy = set(healthy_model_urls.get(model, []))
            message += f"*Model: {model}*\n"

            # Show healthy URLs
            if healthy:
                message += f"✅ Healthy ({len(healthy)}/{len(urls)}):\n"
                for url in healthy:
                    message += f"- `{url}`\n"

            # Show unhealthy URLs
            unhealthy = [url for url in urls if url not in healthy]
            if unhealthy:
                message += f"❌ Unhealthy ({len(unhealthy)}/{len(urls)}):\n"
                for url in unhealthy:
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
            logger.error(f"Failed to process status command: {e}")
            if update.message:
                await update.message.reply_text("Error generating status report. Check server logs.")

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
                message += f"*Model: {model}* ({len(urls)} / {len(server_health_monitor.model_urls[model])})\n"
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
