"""
data_masterpiece_v2.utils.logger - Logging System

Provides a customizable logging system for Data Masterpiece V2.
Supports console and file logging with different log levels.

Features:
    - Colored console output
    - File logging with rotation
    - Configurable log levels
    - Module-specific loggers
    - Progress tracking

Usage:
    >>> from data_masterpiece_v2.utils import get_logger
    >>> logger = get_logger("MyModule")
    >>> logger.info("Hello from MyModule!")
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


# Log level colors
LEVEL_COLORS = {
    "DEBUG": Colors.DIM + Colors.WHITE,
    "INFO": Colors.CYAN,
    "WARNING": Colors.YELLOW,
    "ERROR": Colors.RED,
    "CRITICAL": Colors.BRIGHT_RED + Colors.BOLD,
}

# Emoji markers for different log levels
LEVEL_EMOJIS = {
    "DEBUG": "🔍",
    "INFO": "✨",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "🚨",
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors and emojis to log messages."""

    def __init__(self, fmt: Optional[str] = None, use_emoji: bool = True):
        super().__init__(fmt)
        self.use_emoji = use_emoji

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_name = record.levelname
        color = LEVEL_COLORS.get(level_name, Colors.RESET)
        emoji = LEVEL_EMOJIS.get(level_name, "") if self.use_emoji else ""

        # Create colored level name
        record.levelname = f"{color}{level_name}{Colors.RESET}"
        if emoji:
            record.levelname = f"{emoji} {record.levelname}"

        # Format the message
        message = super().format(record)

        return message


class SimpleFormatter(logging.Formatter):
    """Simple formatter without colors for file logging."""

    def __init__(self, fmt: Optional[str] = None):
        super().__init__(fmt)


# Default format strings
CONSOLE_FORMAT = "%(asctime)s │ %(name)-25s │ %(levelname)s │ %(message)s"
FILE_FORMAT = "%(asctime)s │ %(name)-25s │ %(levelname)-8s │ %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "output/logs",
    log_filename: str = "data_masterpiece.log",
    use_colors: bool = True,
    use_emoji: bool = True,
) -> logging.Logger:
    """
    Set up the root logger for Data Masterpiece V2.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_to_file : bool
        Whether to log to file.
    log_dir : str
        Directory for log files.
    log_filename : str
        Name of the log file.
    use_colors : bool
        Whether to use colored output.
    use_emoji : bool
        Whether to use emoji markers.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    # Get root logger
    root_logger = logging.getLogger("data_masterpiece")
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if use_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(CONSOLE_FORMAT, use_emoji=use_emoji)
    else:
        console_formatter = SimpleFormatter(CONSOLE_FORMAT)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = SimpleFormatter(FILE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__ of the module).

    Returns
    -------
    logging.Logger
        Logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("This is an info message")
    """
    # Ensure root logger is set up
    root_logger = logging.getLogger("data_masterpiece")
    if not root_logger.handlers:
        setup_logging()

    return logging.getLogger(f"data_masterpiece.{name}")


class ProgressLogger:
    """
    A context manager for logging progress of long operations.

    Usage:
        >>> with ProgressLogger("Processing data", total=100) as progress:
        ...     for i in range(100):
        ...         # do work
        ...         progress.update(i + 1)

    Parameters
    ----------
    operation : str
        Name of the operation.
    total : int
        Total number of steps.
    logger : logging.Logger, optional
        Logger instance to use.
    """

    def __init__(
        self,
        operation: str,
        total: int,
        logger: Optional[logging.Logger] = None,
        unit: str = "items"
    ):
        self.operation = operation
        self.total = max(1, total)  # Avoid division by zero
        self.logger = logger or get_logger("progress")
        self.unit = unit
        self.current = 0
        self.start_time = None
        self._last_percentage = -1

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(
                f"✅ Completed: {self.operation} "
                f"({self.total} {self.unit}s in {format_duration(elapsed)})"
            )
        else:
            self.logger.error(
                f"❌ Failed: {self.operation} after {self.current}/{self.total} {self.unit}s"
            )

    def update(self, current: int, message: str = ""):
        """
        Update progress.

        Parameters
        ----------
        current : int
            Current step number.
        message : str, optional
            Additional message to log.
        """
        self.current = current
        percentage = int((current / self.total) * 100)

        # Only log at 25%, 50%, 75%, and 100%
        if percentage >= self._last_percentage + 25 or percentage == 100:
            self._last_percentage = (percentage // 25) * 25
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = current / elapsed if elapsed > 0 else 0
            eta = (self.total - current) / rate if rate > 0 else 0

            progress_bar = "█" * (percentage // 5) + "░" * (20 - percentage // 5)

            log_msg = (
                f"⏳ [{progress_bar}] {percentage:3d}% "
                f"({current}/{self.total}) "
                f"ETA: {format_duration(eta)}"
            )

            if message:
                log_msg += f" - {message}"

            self.logger.info(log_msg)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# Initialize default logging on module import
try:
    setup_logging()
except Exception:
    # If logging setup fails, at least configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format=CONSOLE_FORMAT,
        datefmt=DATE_FORMAT
    )
