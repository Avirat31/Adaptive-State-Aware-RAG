from __future__ import annotations

import logging
import re

_SECRET = re.compile(r"(?i)(bearer\s+|api[_-]?key[\"'=: ]+)([A-Za-z0-9_\-\.]{8,})")


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET.sub(r"\1***REDACTED***", str(record.msg))
        return True


def setup_logging(level: str = "INFO", log_file=None) -> logging.Logger:
    logger = logging.getLogger("adaptive_biorag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(RedactingFilter())
        logger.addHandler(h)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"adaptive_biorag.{name}")
