# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
import logging
import os

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """A command line formatter with different colors for each level."""

    def __init__(self):
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",  # cyan,
            logging.INFO: f"{reset}\033[32m",  # green
            logging.WARNING: f"{reset}\033[33m",  # yellow
            logging.ERROR: f"{reset}\033[31m",  # red
            logging.CRITICAL: f"{reset}\033[35m",  # magenta
        }
        fmt_str = "{color}%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)4d:{reset} %(message)s"
        self.formatters = {
            level: logging.Formatter(fmt_str.format(color=color, reset=reset))
            for level, color in colors.items()
        }
        self.default_formatter = self.formatters[logging.INFO]

    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)


def get_logger(name, level=logging.INFO):
    """A command line logger."""
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        assert (
            level in LOG_LEVELS
        ), f"Invalid LOG_LEVEL: {level}, must be one of {list(LOG_LEVELS.keys())}"
        level = LOG_LEVELS[level]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)
    return logger
