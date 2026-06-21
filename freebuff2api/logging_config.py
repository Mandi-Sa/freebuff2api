from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any

from .config import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(ctx)s%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m",
}

# Per-request correlation fields injected into every log line so concurrent
# requests can be told apart at a glance.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "freebuff_request_id", default=""
)
token_label_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "freebuff_token_label", default=""
)
mode_label_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "freebuff_mode_label", default=""
)

MODE_COLORS = {
    "U": "\033[1;32m",  # bold green  -> UNLIMITED
    "P": "\033[1;33m",  # bold yellow -> PREMIUM
}


def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)


def set_token_context(token_label: str, mode_label: str) -> None:
    token_label_var.set(token_label)
    mode_label_var.set(mode_label)


class ContextFilter(logging.Filter):
    """Inject the ``[request token mode]`` prefix into every record."""

    def __init__(self, color: bool) -> None:
        super().__init__()
        self._color = color

    def filter(self, record: logging.LogRecord) -> bool:
        parts = [
            part
            for part in (request_id_var.get(""), token_label_var.get(""))
            if part
        ]
        mode = mode_label_var.get("")
        if mode:
            parts.append(self._mode(mode))
        record.ctx = f"[{' '.join(parts)}] " if parts else ""
        return True

    def _mode(self, mode: str) -> str:
        tint = MODE_COLORS.get(mode) if self._color else None
        if not tint:
            return mode
        return f"{tint}{mode}{RESET}{COLORS[logging.INFO]}"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = COLORS.get(record.levelno)
        if not color:
            return message
        return f"{color}{message}{RESET}"


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter_cls = ColorFormatter if settings.log_color else logging.Formatter
    handler.setFormatter(formatter_cls(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.addFilter(ContextFilter(settings.log_color))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    logging.getLogger("httpx").setLevel(logging.DEBUG if settings.debug else logging.WARNING)
    logging.getLogger("freebuff2api").debug(
        "logging configured debug=%s level=%s body_chars=%s color=%s",
        settings.debug,
        settings.log_level,
        settings.log_body_chars,
        settings.log_color,
    )


def render_debug(value: Any, limit: int) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)

    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = {}
    for key, value in headers.items():
        if key.lower() in {"authorization", "cookie", "set-cookie"}:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted
