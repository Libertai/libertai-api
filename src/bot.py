"""Telegram bot worker — its own dokploy service, replicas: 1.

Owns the sole getUpdates poller and the periodic Telegram alerts. Reads
health state from Redis (kept fresh by the api app's leader running
``run_jobs``). Lifecycle is PTB-native (``Application.run_polling``); if the
process ever exits, Docker ``restart: unless-stopped`` brings it back.

By living in its own single-replica service, we no longer need leader
election, an asyncio.Lock around the bot lifecycle, or a flag-based
supervisor — the singleton constraint is the deployment, not the code.

Entry point: ``python -m src.bot``."""

import asyncio
from contextlib import suppress

from telegram.ext import Application, CommandHandler

from src.config import config
from src.health import server_health_monitor
from src.logger import setup_logger
from src.redis_client import close_redis
from src.telegram import on_error, send_health_report, status_command

logger = setup_logger(__name__)

# How often to pull fresh health state from Redis (written by the api leader).
SYNC_INTERVAL = 30  # seconds
# Periodic alert cadence — only fires if some servers are actually down.
ALERT_INTERVAL = 1800  # 30 minutes


async def _sync_loop() -> None:
    """Keep ``server_health_monitor`` fresh from Redis so /status and the alert
    loop see what the api leader has just refreshed."""
    while True:
        try:
            await server_health_monitor.sync_from_redis()
        except Exception as e:
            logger.error(f"Error syncing health from Redis: {e}", exc_info=True)
        await asyncio.sleep(SYNC_INTERVAL)


async def _alert_loop(app: Application) -> None:
    """Broadcast a Telegram alert when servers are down. Quiet otherwise."""
    while True:
        try:
            await send_health_report(app.bot)
        except Exception as e:
            logger.error(f"Error sending health alert: {e}", exc_info=True)
        await asyncio.sleep(ALERT_INTERVAL)


async def _post_init(app: Application) -> None:
    """Schedule background loops once the Application is initialized.

    ``Application.create_task`` ties the task to the Application's lifetime,
    so they're properly cancelled on shutdown without us managing them."""
    app.create_task(_sync_loop())
    app.create_task(_alert_loop(app))
    logger.info("Bot worker background tasks scheduled (sync + alerts)")


async def _post_shutdown(_app: Application) -> None:
    with suppress(Exception):
        await close_redis()


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; refusing to start bot worker")
        return

    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("status", status_command))
    app.add_error_handler(on_error)

    logger.info("Starting Telegram bot worker (PTB run_polling)")
    # Blocking; PTB owns the event loop, signal handling, and graceful shutdown.
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
