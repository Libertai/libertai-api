import logging
import sys
from typing import Optional

from src.config import config


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up and configure a logger

    Args:
        name: Logger name (usually __name__ from the calling module)
        level: Logging level (default: from config.LOG_LEVEL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    log_level = level if level is not None else config.LOG_LEVEL

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(log_level)
    logger.propagate = False

    return logger
