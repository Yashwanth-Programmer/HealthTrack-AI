"""File + console logging. Every generated message and every error is recorded."""
from __future__ import annotations

import logging
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE_NAME = "healthtrack.log"
_HANDLER_NAME = "healthtrack-file"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | rid=%(request_id)s | %(message)s"

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Stamps every log line with the id of the HTTP request that caused it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(log_dir: Path | str, level: int = logging.INFO, *, console: bool = True) -> Path:
    """Idempotently attach a rotating file handler to the 'healthtrack' logger."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILE_NAME

    root = logging.getLogger("healthtrack")
    root.setLevel(level)
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            if getattr(handler, "baseFilename", None) == str(log_path.resolve()):
                return log_path
            root.removeHandler(handler)
            handler.close()

    file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.set_name(_HANDLER_NAME)
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    file_handler.addFilter(RequestIdFilter())
    root.addHandler(file_handler)

    if console and not any(h.get_name() == "healthtrack-console" for h in root.handlers):
        console = logging.StreamHandler()
        console.set_name("healthtrack-console")
        console.setFormatter(logging.Formatter(_FORMAT))
        console.addFilter(RequestIdFilter())
        root.addHandler(console)
    return log_path
