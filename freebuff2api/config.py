from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import timedelta, timezone

from dotenv import load_dotenv


load_dotenv()


HAR_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class Settings:
    codebuff_token: str | None
    local_api_key: str | None
    token_index: int = 1
    codebuff_base_url: str = "https://www.codebuff.com"
    zeroclick_base_url: str = "https://zeroclick.dev"
    session_id: str = ""
    client_id: str = ""
    ad_providers: tuple[str, ...] = ("gravity", "zeroclick")
    request_timeout: float = 60.0
    debug: bool = False
    log_level: str = "INFO"
    log_body_chars: int = 2000
    log_color: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    proxy_enabled: bool = False
    proxy_url: str | None = None
    timezone: str = "Asia/Shanghai"
    locale: str = "zh-CN"
    os_name: str = "windows"
    unlimited_model: str = "deepseek/deepseek-v4-flash"
    premium_model: str = "moonshotai/kimi-k2.6"
    schedule_utc_offset: float = -7.0
    session_block_seconds: float = 360.0
    destroy_lead_seconds: float = 45.0
    quota_file: str = "data/quota.json"

    @property
    def codebuff_api_url(self) -> str:
        return self.codebuff_base_url.strip().rstrip("/")

    @property
    def zeroclick_api_url(self) -> str:
        return self.zeroclick_base_url.rstrip("/")

    @property
    def upstream_proxy_url(self) -> str | None:
        if not self.proxy_enabled:
            return None
        if not self.proxy_url:
            return None
        return self.proxy_url.strip() or None

    @property
    def codebuff_tokens(self) -> tuple[str, ...]:
        if not self.codebuff_token:
            return ()
        values = [item.strip() for item in self.codebuff_token.split(",")]
        return tuple(item for item in values if item)

    @property
    def unlimited_models(self) -> tuple[str, ...]:
        values = [item.strip() for item in self.unlimited_model.split(",")]
        return tuple(item for item in values if item)

    def is_premium(self, model: str) -> bool:
        return model == self.premium_model

    def is_unlimited(self, model: str) -> bool:
        return model in self.unlimited_models

    def is_allowed(self, model: str) -> bool:
        return self.is_premium(model) or self.is_unlimited(model)

    @property
    def schedule_timezone(self) -> timezone:
        return timezone(timedelta(hours=self.schedule_utc_offset))

    @property
    def token_hint(self) -> str:
        if not self.codebuff_token:
            return "no-token"
        suffix = self.codebuff_token[-4:]
        return f"***{suffix}"


def _csv(name: str, default: str) -> tuple[str, ...]:
    values = [item.strip() for item in os.getenv(name, default).split(",")]
    return tuple(item for item in values if item)


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _api_base_url() -> str:
    return (
        os.getenv("FREEBUFF_API_BASE_URL")
        or os.getenv("CODEBUFF_BASE_URL")
        or "https://www.codebuff.com"
    )


def load_settings() -> Settings:
    debug = _bool("FREEBUFF_DEBUG", False)
    log_level = "DEBUG" if debug else os.getenv("FREEBUFF_LOG_LEVEL", "INFO")
    color_default = os.getenv("NO_COLOR") is None
    return Settings(
        codebuff_token=os.getenv("FREEBUFF_TOKEN") or os.getenv("CODEBUFF_TOKEN"),
        local_api_key=os.getenv("FREEBUFF_API_KEY") or os.getenv("OPENAI_API_KEY"),
        codebuff_base_url=_api_base_url(),
        zeroclick_base_url=os.getenv("ZEROCLICK_BASE_URL", "https://zeroclick.dev"),
        session_id=os.getenv("FREEBUFF_SESSION_ID", str(uuid.uuid4())),
        client_id=os.getenv("FREEBUFF_CLIENT_ID", uuid.uuid4().hex[:11]),
        ad_providers=_csv("FREEBUFF_AD_PROVIDERS", "gravity,zeroclick"),
        request_timeout=float(os.getenv("FREEBUFF_TIMEOUT", "60")),
        debug=debug,
        log_level=log_level,
        log_body_chars=_int("FREEBUFF_LOG_BODY_CHARS", 0 if debug else 2000),
        log_color=_bool("FREEBUFF_LOG_COLOR", color_default),
        host=os.getenv("FREEBUFF_HOST", "0.0.0.0"),
        port=_int("FREEBUFF_PORT", 8000),
        proxy_enabled=_bool("FREEBUFF_PROXY_ENABLED", False),
        proxy_url=os.getenv("FREEBUFF_PROXY_URL"),
        timezone=os.getenv("FREEBUFF_TIMEZONE", "Asia/Shanghai"),
        locale=os.getenv("FREEBUFF_LOCALE", "zh-CN"),
        os_name=os.getenv("FREEBUFF_OS", "windows"),
        unlimited_model=os.getenv(
            "FREEBUFF_UNLIMITED_MODEL", "deepseek/deepseek-v4-flash"
        ),
        premium_model=os.getenv("FREEBUFF_PREMIUM_MODEL", "moonshotai/kimi-k2.6"),
        schedule_utc_offset=float(os.getenv("FREEBUFF_SCHEDULE_UTC_OFFSET", "-7")),
        session_block_seconds=float(os.getenv("FREEBUFF_SESSION_BLOCK_SECONDS", "360")),
        destroy_lead_seconds=float(os.getenv("FREEBUFF_DESTROY_LEAD_SECONDS", "45")),
        quota_file=os.getenv("FREEBUFF_QUOTA_FILE", "data/quota.json"),
    )
