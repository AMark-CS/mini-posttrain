"""Lightweight logger. Wraps :mod:`logging` so it works inside and outside WandB."""
from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "nano_posttrain", level: int | None = None) -> logging.Logger:
    """Return a stdlib logger that respects :envvar:`LOG_LEVEL`."""
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
