"""
common.logging — Structured logging for QueryOpt Lab.

Design notes:
- Uses Python stdlib logging (no external logging framework dependency).
- Supports both "text" (human-readable) and "json" (machine-readable) formats.
- Every queryopt package should obtain a logger via get_logger(__name__).
- Query text is NOT logged at INFO level to avoid leaking sensitive data.
  Use DEBUG level only, and only when debug mode is explicitly enabled.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON lines for machine-readable output."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[attr-defined]
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", format_: str = "text") -> None:
    """
    Configure the root logger for QueryOpt Lab.

    Call once at application startup (CLI entry point or FastAPI startup).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger("queryopt")
    root_logger.setLevel(numeric_level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler()

    if format_ == "json":
        handler.setFormatter(JsonFormatter())
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Obtain a named logger scoped under the 'queryopt' namespace.

    Usage:
        logger = get_logger(__name__)
        logger.info("Query received", extra={"query_id": qid})
    """
    if not name.startswith("queryopt"):
        name = f"queryopt.{name}"
    return logging.getLogger(name)


# ── Timing Context Manager ────────────────────────────────────────────────────

class Timer:
    """
    A simple timer context manager for measuring elapsed time in milliseconds.

    Usage:
        with Timer() as t:
            do_work()
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
