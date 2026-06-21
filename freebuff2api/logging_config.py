from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .config import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m",
}

MODE_BADGES = {
    "UNLIMITED": "\033[1;37;42m",  # bold white on green background
    "PREMIUM": "\033[1;30;43m",  # bold black on yellow background
}


def mode_tag(label: str, *, color: bool) -> str:
    """Render an eye-catching ``[LABEL]`` badge for the token-selection mode.

    With color enabled the badge is a bright background block; the trailing
    sequence re-opens the INFO color so the rest of the line keeps its level
    color. Without color it degrades to a plain ``[LABEL]`` tag.
    """
    text = f"[{label.center(9)}]"
    badge = MODE_BADGES.get(label)
    if not color or not badge:
        return text
    return f"{badge}{text}{RESET}{COLORS[logging.INFO]}"


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
