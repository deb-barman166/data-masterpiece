"""
data_masterpiece_v3.utils.logger
─────────────────────────────────
Neon-coloured terminal logger with timestamps and Unicode icons.
Every pipeline stage prints beautifully so even a 12-year-old can follow
exactly what is happening, step by step.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime


# ── ANSI colour palette ──────────────────────────────────────────────────────
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_CYAN    = "\033[96m"
_GREEN   = "\033[92m"
_YELLOW  = "\033[93m"
_RED     = "\033[91m"
_MAGENTA = "\033[95m"
_BLUE    = "\033[94m"
_DIM     = "\033[2m"


class _NeonFormatter(logging.Formatter):
    """Custom formatter that adds colour and icon to every log line."""

    ICONS = {
        logging.DEBUG:    "🔍",
        logging.INFO:     "⚡",
        logging.WARNING:  "⚠️ ",
        logging.ERROR:    "❌",
        logging.CRITICAL: "💀",
    }

    COLOURS = {
        logging.DEBUG:    _DIM,
        logging.INFO:     _CYAN,
        logging.WARNING:  _YELLOW,
        logging.ERROR:    _RED,
        logging.CRITICAL: _RED + _BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        icon   = self.ICONS.get(record.levelno, "•")
        colour = self.COLOURS.get(record.levelno, _RESET)
        ts     = datetime.now().strftime("%H:%M:%S")
        name   = f"{_BLUE}{record.name}{_RESET}"
        msg    = record.getMessage()
        return (
            f"{_DIM}[{ts}]{_RESET} "
            f"{colour}{icon}{_RESET} "
            f"{name} "
            f"{colour}{msg}{_RESET}"
        )


# ── public factory ────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with neon console output (no duplicate handlers)."""
    logger = logging.getLogger(f"dm_v3.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_NeonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def section_banner(title: str, width: int = 60) -> None:
    """Print a neon-bordered section banner to stdout."""
    border = "═" * width
    pad    = (width - len(title) - 2) // 2
    line   = "║" + " " * pad + title + " " * (width - pad - len(title) - 2) + "║"
    print(f"\n{_CYAN}{_BOLD}╔{border}╗")
    print(line)
    print(f"╚{border}╝{_RESET}\n")
