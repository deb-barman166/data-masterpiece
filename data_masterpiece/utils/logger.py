"""
data_masterpiece/utils/logger.py  --  Structured, colourised logging.
"""

import logging
import sys

_R  = "\033[0m"
_B  = "\033[1m"
_G  = "\033[32m"
_Y  = "\033[33m"
_RE = "\033[31m"
_C  = "\033[36m"


class _ColourFormatter(logging.Formatter):
    """ANSI-coloured log formatter for terminal output."""

    _LEVEL_COLOUR = {
        logging.DEBUG:    _C,
        logging.INFO:     _G,
        logging.WARNING:  _Y,
        logging.ERROR:    _RE + _B,
        logging.CRITICAL: _RE + _B,
    }

    def format(self, record):
        lc = self._LEVEL_COLOUR.get(record.levelno, "")
        # Build the format string with escaped ANSI codes
        # Use $$ to safely pass through literal % that aren't logging fields
        record.lc = lc
        record.rc = _R
        record.cc = _C
        try:
            msg = (
                "%(asctime)s  [%(lc)s%(levelname)-8s%(rc)s]  "
                "[%(cc)s%(name)s%(rc)s]  %(message)s"
            )
            return (
                logging.Formatter(msg, datefmt="%Y-%m-%d %H:%M:%S")
                .format(record)
            )
        finally:
            # Clean up our custom fields so they don't leak
            del record.lc
            del record.rc
            del record.cc


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a coloured console handler."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_ColourFormatter())
    log.addHandler(handler)
    log.propagate = False
    return log


def get_file_logger(name: str, filepath: str = "data_masterpiece.log") -> logging.Logger:
    """Return a logger that writes DEBUG+ to both console and a file."""
    log = get_logger(name, logging.DEBUG)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler(filepath, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  [%(levelname)-8s]  [%(name)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(fh)
    return log
