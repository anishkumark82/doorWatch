# logging_setup.py
import logging
from logging.handlers import RotatingFileHandler
import os
import time

def get_logger(log_path, max_bytes=2_000_000, backup_count=1):
    """A logger that writes to both the console and a size-capped file.
    RotatingFileHandler automatically rotates once max_bytes is hit,
    keeping only `backup_count` old files -- no manual truncation or
    external tail/cron tricks needed."""
    logger = logging.getLogger("door_watchman")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # avoid adding duplicate handlers if called twice

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class ThrottledLogger:
    """Wraps a logger so repeated calls with the same key only actually
    log once per `interval` seconds -- avoids a log file filling up with
    thousands of near-identical 'still idle' lines."""

    def __init__(self, logger, interval=60):
        self.logger = logger
        self.interval = interval
        self._last_logged = {}   # key -> last time it was actually logged

    def info(self, message, key=None):
        key = key or message   # default: throttle identical messages together
        now = time.time()
        last = self._last_logged.get(key, 0)
        if now - last >= self.interval:
            self.logger.info(message)
            self._last_logged[key] = now