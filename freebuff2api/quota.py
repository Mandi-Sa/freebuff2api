from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger("freebuff2api.quota")


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _reset_passed(reset_at: str | None) -> bool:
    if not reset_at:
        return False
    try:
        moment = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return moment <= datetime.now(timezone.utc)


def _fmt(value: float | None) -> str:
    return "?" if value is None else f"{value:g}"


@dataclass
class TokenQuota:
    token_index: int
    token_hint: str
    used: float | None = None
    limit: float | None = None
    reset_at: str | None = None
    updated_at: str | None = None


class QuotaStore:
    """Per-token premium quota, parsed from upstream session responses.

    The premium counter (``recentCount``/``limit``/``resetAt``) is shared across
    premium models. We track the latest seen value per token and persist it so
    it survives restarts.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._quotas: dict[int, TokenQuota] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return
        for item in data.get("tokens", []):
            try:
                quota = TokenQuota(**item)
            except TypeError:
                continue
            self._quotas[quota.token_index] = quota

    def record(
        self,
        *,
        token_index: int,
        token_hint: str,
        used: float | None,
        limit: float | None,
        reset_at: str | None,
    ) -> None:
        with self._lock:
            previous = self._quotas.get(token_index)
            changed = previous is None or (
                previous.used != used
                or previous.limit != limit
                or previous.reset_at != reset_at
            )
            self._quotas[token_index] = TokenQuota(
                token_index=token_index,
                token_hint=token_hint,
                used=used,
                limit=limit,
                reset_at=reset_at,
                updated_at=_now_iso(),
            )
            if changed:
                self._persist_locked()
        # session ops are frequent; only log when the counter actually moves
        if changed:
            logger.info(
                "premium quota token_index=%s token=%s used=%s/%s resetAt=%s",
                token_index,
                token_hint,
                _fmt(used),
                _fmt(limit),
                reset_at,
            )

    def _persist_locked(self) -> None:
        payload = {
            "tokens": [
                asdict(quota)
                for quota in sorted(
                    self._quotas.values(), key=lambda item: item.token_index
                )
            ]
        }
        directory = os.path.dirname(self._path)
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp = f"{self._path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            logger.warning("failed to persist quota file=%s", self._path)

    def get(self, token_index: int) -> TokenQuota | None:
        with self._lock:
            return self._quotas.get(token_index)

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            quotas = sorted(self._quotas.values(), key=lambda item: item.token_index)
        for quota in quotas:
            row = asdict(quota)
            # after the daily reset the upstream counter is back to 0
            row["effective_used"] = (
                0.0 if _reset_passed(quota.reset_at) else quota.used
            )
            rows.append(row)
        return rows
