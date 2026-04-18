"""
data_masterpiece_v2.utils - Utility Functions and Classes

This module provides utility functions used throughout Data Masterpiece V2.
Includes logging, file handling, data manipulation helpers, and more.

Functions:
    get_logger() - Get a configured logger instance
    ensure_dir() - Ensure a directory exists
    format_bytes() - Format bytes to human readable string
    format_duration() - Format seconds to human readable string
    print_progress() - Print a progress bar
"""

from data_masterpiece_v2.utils.logger import get_logger, setup_logging
from data_masterpiece_v2.utils.helpers import (
    ensure_dir,
    format_bytes,
    format_duration,
    print_progress,
    safe_divide,
    detect_task_type,
    infer_problem_type
)

__all__ = [
    "get_logger",
    "setup_logging",
    "ensure_dir",
    "format_bytes",
    "format_duration",
    "print_progress",
    "safe_divide",
    "detect_task_type",
    "infer_problem_type",
]
