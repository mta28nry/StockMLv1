# log_config.py
import os
import logging
from logging.handlers import TimedRotatingFileHandler

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logger Factory ---
def get_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # Unified formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, f"{name}.log"),
        when="midnight",
        backupCount=14,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
