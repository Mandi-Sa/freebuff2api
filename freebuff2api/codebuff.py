from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from .config import HAR_BROWSER_USER_AGENT, Settings
from .logging_config import (
    redact_headers,
    render_debug,
    set_request_id,
    set_token_context,
)
from .models import agent_validation_payload


logger = logging.getLogger("freebuff2api.codebuff")

CODEBUFF_ACCEPT_ENCODING = "gzip, deflate"
CODEBUFF_JSON_USER_AGENT = "Bun/1.3.11"
FREEBUFF_CLI_USER_AGENT = "Freebuff-CLI/0.0.105"
CHAT_COMPLETIONS_USER_AGENT = (
    "ai-sdk/openai-compatible/0.0.0-test/codebuff "
    "ai-sdk/provider-utils/3.0.20 runtime/browser"
)


class CodebuffError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        *,
        token_index: int | None = None,
        token_hint: str | None = None,
        account_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.token_index = token_index
        self.token_hint = token_hint
        self.account_index = account_index


@dataclass
class FreebuffSession:
    instance_id: str
    model: str
    expires_at: str | None = None
    remaining_ms: int | None = None

    @property
    def is_fresh(self) -> bool:
        return self.remaining_ms is None or self.remaining_ms > 60_000


@dataclass
class FreebuffRun:
    run_id: str
    agent_id: str
    started_at: str
    child_run_id: str | None = None
    chat_run_id: str | None = None
    chat_started_at: str | None = None

    @property
    def payload_run_id(self) -> str:
        return self.chat_run_id or self.run_id


@dataclass
class FreebuffSessionLease:
    session: FreebuffSession
    _lock: asyncio.Lock
    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lock.release()


class CodebuffClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout, read=None),
            follow_redirects=True,
            proxy=settings.upstream_proxy_url,
            trust_env=False,
        )
        self._agents_validated = False
        self._validate_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(
        self,
        *,
        json_body: bool = False,
        user_agent: str = CODEBUFF_JSON_USER_AGENT,
        require_auth: bool = True,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if require_auth and not self.settings.codebuff_token:
            raise CodebuffError("FREEBUFF_TOKEN or CODEBUFF_TOKEN is required", 500)

        headers = {
            "Accept": "*/*",
            "Accept-Encoding": CODEBUFF_ACCEPT_ENCODING,
            "Connection": "keep-alive",
            "Host": _host_header(self.settings.codebuff_api_url),
            "User-Agent": user_agent,
        }
        if require_auth:
            headers["Authorization"] = f"Bearer {self.settings.codebuff_token}"
        if json_body:
            headers["Content-Type"] = "application/json"
        if extra:
            headers.update(extra)
        return headers

    async def _json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.settings.codebuff_api_url}{path}"
        request_headers = headers or self._headers(json_body=body is not None)
        try:
            response = await self._client.request(
                method,
                url,
                json=body,
                headers=request_headers,
            )
        except httpx.RequestError as error:
            raise _network_error(method, url, error) from error
        if self.settings.debug:
            logger.debug(
                "upstream json request method=%s url=%s headers=%s body=%s",
                method,
                url,
                redact_headers(request_headers),
                render_debug(body, self.settings.log_body_chars),
            )
            logger.debug(
                "upstream json response status=%s body=%s",
                response.status_code,
                render_debug(response.text, self.settings.log_body_chars),
            )
        if response.status_code >= 400:
            raise _upstream_error(response)
        if not response.content:
            return {}
        return response.json()

    async def validate_agents(self) -> None:
        if self._agents_validated:
            return
        async with self._validate_lock:
            if self._agents_validated:
                return
            try:
                data = await self._json(
                    "POST",
                    "/api/agents/validate",
                    body=agent_validation_payload(),
                    headers=self._headers(json_body=True, require_auth=False),
                )
            except CodebuffError:
                logger.warning(
                    "agent validation failed; continuing with server configs",
                    exc_info=self.settings.debug,
                )
                self._agents_validated = True
                return
            error_count = int(data.get("errorCount") or 0)
            if error_count:
                logger.warning(
                    "agent validation returned errors count=%s body=%s",
                    error_count,
                    render_debug(data, self.settings.log_body_chars),
                )
            else:
                logger.info(
                    "agent validation completed configs=%s",
                    len(data.get("configs") or []),
                )
            self._agents_validated = True

    async def health(self) -> dict[str, Any]:
        return await self._json(
            "GET",
            "/api/healthz",
            headers=self._headers(require_auth=False),
        )

    async def get_session(self, instance_id: str | None = None) -> dict[str, Any]:
        headers_extra = {}
        if instance_id:
            headers_extra["x-freebuff-instance-id"] = instance_id
        return await self._json(
            "GET",
            "/api/v1/freebuff/session",
            headers=self._headers(extra=headers_extra),
        )

    async def create_session(self, model: str) -> FreebuffSession:
        data = await self._json(
            "POST",
            "/api/v1/freebuff/session",
            headers=self._headers(extra={"x-freebuff-model": model}),
        )
        if data.get("status") == "queued":
            return await self._wait_for_active_session(data, model)
        return self._session_from_data(data, model)

    def _session_from_data(
        self,
        data: dict[str, Any],
        model: str,
        instance_id: str | None = None,
    ) -> FreebuffSession:
        resolved_instance_id = data.get("instanceId") or instance_id
        if data.get("status") != "active" or not resolved_instance_id:
            raise CodebuffError(f"Freebuff session is not active: {data}", 502)
        return FreebuffSession(
            instance_id=resolved_instance_id,
            model=data.get("model") or model,
            expires_at=data.get("expiresAt"),
            remaining_ms=data.get("remainingMs"),
        )

    async def _wait_for_active_session(
        self,
        data: dict[str, Any],
        model: str,
    ) -> FreebuffSession:
        instance_id = data.get("instanceId")
        if not instance_id:
            raise CodebuffError(f"Freebuff queued session id missing: {data}", 502)

        deadline = time.monotonic() + self.settings.request_timeout
        attempts = 0
        while data.get("status") == "queued":
            logger.info(
                "freebuff session queued model=%s instance_id=%s position=%s estimated_wait_ms=%s",
                model,
                instance_id,
                data.get("position"),
                data.get("estimatedWaitMs"),
            )
            if time.monotonic() >= deadline:
                raise CodebuffError(
                    f"Freebuff session did not become active before timeout: {data}",
                    502,
                )
            if attempts:
                await asyncio.sleep(_queue_poll_delay(data.get("estimatedWaitMs")))
            data = await self.get_session(instance_id)
            attempts += 1

        return self._session_from_data(data, model, instance_id=instance_id)

    async def delete_session(self) -> None:
        await self._json(
            "DELETE",
            "/api/v1/freebuff/session",
            headers=self._headers(),
        )
        logger.info("deleted active freebuff session")

    async def get_streak(self) -> dict[str, Any]:
        data = await self._json(
            "GET",
            "/api/v1/freebuff/streak",
            headers=self._headers(),
        )
        logger.info(
            "freebuff streak streak=%s today_used=%s",
            data.get("streak"),
            data.get("todayUsed"),
        )
        return data

    async def request_ads(
        self,
        provider: str,
        messages: list[dict[str, Any]] | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "provider": provider,
            "messages": _ad_messages(messages),
            "sessionId": self.settings.session_id,
            "device": {
                "os": self.settings.os_name,
                "timezone": self.settings.timezone,
                "locale": self.settings.locale,
            },
            "userAgent": HAR_BROWSER_USER_AGENT,
        }
        if surface:
            body["surface"] = surface
        return await self._json(
            "POST",
            "/api/v1/ads",
            body=body,
            headers=self._headers(
                json_body=True,
                user_agent=FREEBUFF_CLI_USER_AGENT,
            ),
        )

    async def request_ad_chain(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        surface: str | None = None,
    ) -> None:
        for provider in self.settings.ad_providers:
            try:
                ads_data = await self.request_ads(
                    provider,
                    messages=messages,
                    surface=surface,
                )
                ads = ads_data.get("ads") or []
                ad = ads[0] if ads else None
                logger.info(
                    "ads provider=%s messages=%s count=%s selected=%s",
                    provider,
                    len(messages or []),
                    len(ads),
                    bool(ad),
                )
                if not ad:
                    continue
                await self.report_zeroclick_impressions(
                    list(ad.get("impressionIds") or [])
                )
                await self.report_codebuff_impression(ad.get("impUrl") or "")
                return
            except CodebuffError as error:
                logger.warning(
                    "ads provider=%s failed; continuing without blocking chat: %s",
                    provider,
                    error,
                    exc_info=self.settings.debug,
                )

    async def report_zeroclick_impressions(self, ids: list[str]) -> None:
        if not ids:
            return
        url = f"{self.settings.zeroclick_api_url}/api/v2/impressions"
        try:
            response = await self._client.post(
                url,
                json={"ids": ids},
                headers={
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "User-Agent": CODEBUFF_JSON_USER_AGENT,
                },
            )
        except httpx.RequestError as error:
            raise _network_error("POST", url, error) from error
        if self.settings.debug:
            logger.debug(
                "zeroclick impression ids=%s status=%s body=%s",
                ids,
                response.status_code,
                render_debug(response.text, self.settings.log_body_chars),
            )
        if response.status_code >= 400:
            raise CodebuffError(
                f"Zeroclick impression failed: {response.status_code} {response.text[:500]}",
                502,
            )

    async def report_codebuff_impression(self, imp_url: str) -> None:
        if not imp_url:
            return
        await self._json(
            "POST",
            "/api/v1/ads/impression",
            body={"impUrl": imp_url, "mode": "LITE"},
            headers=self._headers(
                json_body=True,
                user_agent=FREEBUFF_CLI_USER_AGENT,
            ),
        )

    async def start_run(
        self,
        agent_id: str,
        ancestor_run_ids: list[str] | None = None,
    ) -> str:
        data = await self._json(
            "POST",
            "/api/v1/agent-runs",
            body={
                "action": "START",
                "agentId": agent_id,
                "ancestorRunIds": ancestor_run_ids or [],
            },
        )
        run_id = data.get("runId")
        if not run_id:
            raise CodebuffError(f"Codebuff run id missing: {data}", 502)
        logger.info(
            "agent run started agent_id=%s run_id=%s ancestors=%s",
            agent_id,
            run_id,
            ancestor_run_ids or [],
        )
        return run_id

    async def record_run_step(
        self,
        run_id: str,
        *,
        step_number: int,
        message_id: str | None,
        start_time: str,
        child_run_ids: list[str] | None = None,
    ) -> None:
        await self._json(
            "POST",
            f"/api/v1/agent-runs/{run_id}/steps",
            body={
                "stepNumber": step_number,
                "credits": 0,
                "childRunIds": child_run_ids or [],
                "messageId": message_id,
                "status": "completed",
                "startTime": start_time,
            },
        )
        logger.info(
            "agent run step recorded run_id=%s step=%s message_id=%s children=%s",
            run_id,
            step_number,
            message_id,
            child_run_ids or [],
        )

    async def finish_run(self, run_id: str, *, total_steps: int) -> None:
        await self._json(
            "POST",
            "/api/v1/agent-runs",
            body={
                "action": "FINISH",
                "runId": run_id,
                "status": "completed",
                "totalSteps": total_steps,
                "directCredits": 0,
                "totalCredits": 0,
            },
        )
        logger.info("agent run finished run_id=%s total_steps=%s", run_id, total_steps)

    async def chat_events(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        url = f"{self.settings.codebuff_api_url}/api/v1/chat/completions"
        request_headers = self._headers(
            json_body=True,
            user_agent=CHAT_COMPLETIONS_USER_AGENT,
        )
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=request_headers,
            ) as response:
                if self.settings.debug:
                    logger.debug(
                        "chat stream request url=%s headers=%s payload=%s",
                        url,
                        redact_headers(request_headers),
                        render_debug(payload, self.settings.log_body_chars),
                    )
                    logger.debug(
                        "chat stream response status=%s headers=%s",
                        response.status_code,
                        redact_headers(dict(response.headers)),
                    )
                if response.status_code >= 400:
                    text = await response.aread()
                    raise _upstream_error(
                        response,
                        body=text,
                        prefix="Codebuff chat failed",
                    )
                async for line in response.aiter_lines():
                    if self.settings.debug:
                        logger.debug(
                            "chat stream line=%s",
                            render_debug(line, self.settings.log_body_chars),
                        )
                    yield line
        except httpx.RequestError as error:
            raise _network_error("POST", url, error) from error


class SessionManager:
    def __init__(self, client: CodebuffClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self._sessions: dict[str, FreebuffSession] = {}
        self._lock = asyncio.Lock()

    async def ensure_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSession:
        async with self._lock:
            return await self._ensure_session_locked(model, messages)

    async def acquire_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSessionLease:
        await self._lock.acquire()
        try:
            session = await self._ensure_session_locked(model, messages)
        except Exception:
            self._lock.release()
            raise
        return FreebuffSessionLease(session=session, _lock=self._lock)

    async def _ensure_session_locked(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSession:
        cached = self._sessions.get(model)
        if cached and cached.is_fresh:
            try:
                data = await self.client.get_session(cached.instance_id)
                if data.get("status") == "active" and data.get("model") in {
                    None,
                    model,
                }:
                    cached.remaining_ms = data.get("remainingMs")
                    logger.debug(
                        "reuse freebuff session model=%s instance_id=%s remaining_ms=%s",
                        model,
                        cached.instance_id,
                        cached.remaining_ms,
                    )
                    return cached
                if data.get("status") == "active":
                    logger.info(
                        "cached freebuff session model mismatch cached=%s upstream=%s",
                        model,
                        data.get("model"),
                    )
                    self._sessions.pop(model, None)
            except CodebuffError:
                logger.debug(
                    "cached freebuff session invalid model=%s instance_id=%s",
                    model,
                    cached.instance_id,
                    exc_info=self.settings.debug,
                )
                self._sessions.pop(model, None)

        active_session = await self._delete_locked_session(model)
        if active_session:
            return active_session
        await self._request_ads_and_streak(surface="waiting_room")

        try:
            session = await self.client.create_session(model)
        except CodebuffError as error:
            if "model_locked" not in str(error):
                raise
            logger.info(
                "freebuff session locked during create; delete and retry model=%s",
                model,
            )
            await self.client.delete_session()
            self._sessions.clear()
            await self._request_ads_and_streak(surface="waiting_room")
            session = await self.client.create_session(model)
        self._sessions[model] = session
        logger.debug(
            "created freebuff session model=%s instance_id=%s remaining_ms=%s",
            model,
            session.instance_id,
            session.remaining_ms,
        )
        return session

    async def _request_ads_and_streak(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        surface: str | None = None,
    ) -> None:
        for provider in self.settings.ad_providers:
            try:
                ads_data = await self.client.request_ads(
                    provider,
                    messages=messages,
                    surface=surface,
                )
                ads = ads_data.get("ads") or []
                ad = ads[0] if ads else None
                logger.info(
                    "ads provider=%s messages=%s count=%s selected=%s",
                    provider,
                    len(messages or []),
                    len(ads),
                    bool(ad),
                )
                if not ad:
                    continue
                await self.client.get_streak()
                await self.client.report_zeroclick_impressions(
                    list(ad.get("impressionIds") or [])
                )
                await self.client.report_codebuff_impression(ad.get("impUrl") or "")
                return
            except CodebuffError as error:
                logger.warning(
                    "ads provider=%s failed; continuing without blocking chat: %s",
                    provider,
                    error,
                    exc_info=self.settings.debug,
                )

    async def _delete_locked_session(
        self,
        requested_model: str,
    ) -> FreebuffSession | None:
        try:
            data = await self.client.get_session()
        except CodebuffError:
            logger.debug(
                "could not inspect active freebuff session before create",
                exc_info=self.settings.debug,
            )
            return None

        if data.get("status") != "active":
            return None

        current_model = data.get("model")
        instance_id = data.get("instanceId")
        if current_model == requested_model and instance_id:
            session = FreebuffSession(
                instance_id=instance_id,
                model=current_model,
                expires_at=data.get("expiresAt"),
                remaining_ms=data.get("remainingMs"),
            )
            self._sessions[requested_model] = session
            logger.info(
                "discovered active freebuff session model=%s instance_id=%s remaining_ms=%s",
                requested_model,
                session.instance_id,
                session.remaining_ms,
            )
            return session

        if not current_model or current_model == requested_model:
            return None

        logger.info(
            "switch freebuff session current_model=%s requested_model=%s instance_id=%s",
            current_model,
            requested_model,
            instance_id,
        )
        await self.client.delete_session()
        self._sessions.clear()
        return None


@dataclass
class CodebuffAccount:
    client: CodebuffClient
    sessions: SessionManager
    busy: bool = False


@dataclass
class CodebuffAccountLease:
    client: CodebuffClient
    session: FreebuffSession
    _session_lease: FreebuffSessionLease
    _pool: CodebuffAccountPool
    _account_index: int
    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._session_lease.aclose()
        await self._pool.release(self._account_index)

    @property
    def account_index(self) -> int:
        return self._account_index


class CodebuffAccountPool:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        tokens = settings.codebuff_tokens or (None,)
        self._accounts: list[CodebuffAccount] = []
        for index, token in enumerate(tokens, start=1):
            account_settings = replace(
                settings,
                codebuff_token=token,
                token_index=index,
            )
            client = CodebuffClient(account_settings)
            self._accounts.append(
                CodebuffAccount(
                    client=client,
                    sessions=SessionManager(client, account_settings),
                )
            )
        self._active_index: int | None = None
        self._condition = asyncio.Condition()
        self._scheduler_task: asyncio.Task[None] | None = None

    @property
    def account_count(self) -> int:
        return len(self._accounts)

    @property
    def default_client(self) -> CodebuffClient:
        return self._accounts[0].client

    @property
    def default_sessions(self) -> SessionManager:
        return self._accounts[0].sessions

    def start_scheduler(self) -> None:
        """Switch the active token window on time, even without traffic."""
        if len(self._accounts) <= 1 or self._scheduler_task is not None:
            return
        self._scheduler_task = asyncio.create_task(self._run_scheduler())

    async def aclose(self) -> None:
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        await asyncio.gather(
            *(account.client.aclose() for account in self._accounts),
            return_exceptions=True,
        )

    def _now(self) -> datetime:
        return datetime.now(self._settings.schedule_timezone)

    async def _run_scheduler(self) -> None:
        while True:
            await asyncio.sleep(self._seconds_until_next_window(self._now()))
            try:
                async with self._condition:
                    self._refresh_active_window()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("freebuff token window scheduler tick failed")

    def _seconds_until_next_window(self, now: datetime) -> float:
        window_seconds = 86_400 / len(self._accounts)
        seconds_into_day = (
            now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6
        )
        current_window = int(seconds_into_day / window_seconds)
        next_boundary = (current_window + 1) * window_seconds
        return max(next_boundary - seconds_into_day, 0.0) + 0.5

    async def acquire_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
        *,
        exclude_account_indices: set[int] | None = None,
    ) -> CodebuffAccountLease:
        account_index = await self._reserve_account(model, exclude_account_indices or set())
        account = self._accounts[account_index]
        try:
            session_lease = await account.sessions.acquire_session(model, messages)
        except CodebuffError as error:
            if error.token_index is None:
                error.token_index = account.client.settings.token_index
            if error.token_hint is None:
                error.token_hint = account.client.settings.token_hint
            if error.account_index is None:
                error.account_index = account_index
            logger.warning(
                "account session acquire failed token_index=%s token=%s model=%s: %s",
                account.client.settings.token_index,
                account.client.settings.token_hint,
                model,
                error,
                exc_info=account.client.settings.debug,
            )
            await self.release(account_index)
            raise
        except Exception:
            logger.exception(
                "account session acquire failed token_index=%s token=%s model=%s",
                account.client.settings.token_index,
                account.client.settings.token_hint,
                model,
            )
            await self.release(account_index)
            raise
        return CodebuffAccountLease(
            client=account.client,
            session=session_lease.session,
            _session_lease=session_lease,
            _pool=self,
            _account_index=account_index,
        )

    async def release(self, account_index: int) -> None:
        async with self._condition:
            self._accounts[account_index].busy = False
            self._condition.notify(1)

    async def _reserve_account(
        self,
        model: str,
        exclude_account_indices: set[int],
    ) -> int:
        if len(exclude_account_indices) >= len(self._accounts):
            raise CodebuffError("No remaining accounts available for retry", 503)
        allow_switch = model in self._settings.unlimited_models
        async with self._condition:
            self._refresh_active_window()
            active_index = self._active_index or 0
            if not allow_switch and active_index in exclude_account_indices:
                active_settings = self._accounts[active_index].client.settings
                raise CodebuffError(
                    "current window token_index="
                    f"{active_settings.token_index} unavailable and switching is "
                    f"disabled for non-unlimited model {model}",
                    503,
                )
            waiting_logged = False
            while True:
                account_index = self._next_available_index(
                    exclude_account_indices,
                    allow_switch=allow_switch,
                )
                if account_index is not None:
                    self._accounts[account_index].busy = True
                    self._bind_token_context(account_index, allow_switch)
                    self._log_account_selected(account_index, model)
                    return account_index
                if not waiting_logged:
                    self._log_account_busy(model, allow_switch)
                    waiting_logged = True
                await self._condition.wait()

    def _next_available_index(
        self,
        exclude_account_indices: set[int],
        *,
        allow_switch: bool = True,
    ) -> int | None:
        account_count = len(self._accounts)
        start = self._active_index or 0
        span = account_count if allow_switch else 1
        for offset in range(span):
            account_index = (start + offset) % account_count
            if account_index in exclude_account_indices:
                continue
            if not self._accounts[account_index].busy:
                return account_index
        return None

    def _bind_token_context(self, account_index: int, allow_switch: bool) -> None:
        token_index = self._accounts[account_index].client.settings.token_index
        set_token_context(
            f"t{token_index}/{len(self._accounts)}",
            "U" if allow_switch else "P",
        )

    def _log_account_selected(self, account_index: int, model: str) -> None:
        active_index = self._active_index or 0
        client_settings = self._accounts[account_index].client.settings
        if account_index == active_index:
            logger.info(
                "using freebuff token=%s model=%s",
                client_settings.token_hint,
                model,
            )
            return
        active_settings = self._accounts[active_index].client.settings
        logger.info(
            "using fallback freebuff token=%s (current window token_index=%s busy) model=%s",
            client_settings.token_hint,
            active_settings.token_index,
            model,
        )

    def _log_account_busy(self, model: str, allow_switch: bool) -> None:
        active_index = self._active_index or 0
        active_settings = self._accounts[active_index].client.settings
        if allow_switch:
            logger.info(
                "freebuff all tokens busy; waiting for a free token model=%s",
                model,
            )
            return
        logger.info(
            "freebuff current window token_index=%s/%s token=%s reached concurrency "
            "limit; switching disabled, waiting model=%s",
            active_settings.token_index,
            len(self._accounts),
            active_settings.token_hint,
            model,
        )

    def _refresh_active_window(self) -> None:
        account_count = len(self._accounts)
        new_index = _token_window_index(self._now(), account_count)
        if self._active_index is None:
            self._active_index = new_index
            return
        if new_index == self._active_index:
            return
        previous_index = self._active_index
        self._active_index = new_index
        logger.info(
            "freebuff token window switch previous_token_index=%s next_token_index=%s/%s",
            self._accounts[previous_index].client.settings.token_index,
            self._accounts[new_index].client.settings.token_index,
            account_count,
        )
        self._schedule_park(previous_index)

    def _schedule_park(self, account_index: int) -> None:
        model = self._settings.park_model
        if not model:
            return
        account = self._accounts[account_index]
        task = asyncio.create_task(self._park_account(account_index, model))

        def _log_park_error(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "parking token_index=%s token=%s on unlimited model=%s failed",
                    account.client.settings.token_index,
                    account.client.settings.token_hint,
                    model,
                    exc_info=account.client.settings.debug,
                )

        task.add_done_callback(_log_park_error)

    async def _park_account(self, account_index: int, model: str) -> None:
        account = self._accounts[account_index]
        set_request_id("park")
        self._bind_token_context(account_index, allow_switch=True)
        lease = await account.sessions.acquire_session(model)
        try:
            logger.info(
                "parked token_index=%s token=%s on unlimited model=%s instance_id=%s",
                account.client.settings.token_index,
                account.client.settings.token_hint,
                model,
                lease.session.instance_id,
            )
        finally:
            await lease.aclose()


def _token_window_index(now: datetime, account_count: int) -> int:
    """Map the current local time of day to a token index.

    The 24h day is split into ``account_count`` equal windows so each token is
    used for ``24 / account_count`` hours before switching to the next one.
    """
    if account_count <= 1:
        return 0
    seconds_into_day = now.hour * 3600 + now.minute * 60 + now.second
    window_seconds = 86_400 / account_count
    return min(int(seconds_into_day / window_seconds), account_count - 1)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _host_header(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or "www.codebuff.com"


def _queue_poll_delay(estimated_wait_ms: Any) -> float:
    if isinstance(estimated_wait_ms, int | float) and estimated_wait_ms > 0:
        return min(max(float(estimated_wait_ms) / 1000.0, 0.25), 2.0)
    return 0.25


def _network_error(method: str, url: str, error: httpx.RequestError) -> CodebuffError:
    detail = str(error).strip()
    suffix = f": {detail}" if detail else ""
    return CodebuffError(
        f"Codebuff request failed: {method} {url} network error "
        f"({type(error).__name__}){suffix}",
        502,
    )


def _upstream_error(
    response: httpx.Response,
    *,
    body: bytes | None = None,
    prefix: str = "Codebuff request failed",
) -> CodebuffError:
    raw_text = (
        body.decode("utf-8", errors="replace")
        if body is not None
        else response.text
    )
    text = raw_text[:500]
    if response.status_code == 409:
        try:
            data = (
                response.json()
                if body is None
                else httpx.Response(
                    response.status_code,
                    content=body,
                    headers=response.headers,
                ).json()
            )
        except ValueError:
            data = {}
        if data.get("error") == "session_model_mismatch":
            upstream_message = data.get("message") or text
            return CodebuffError(
                "Codebuff 409 session_model_mismatch: "
                f"{upstream_message} 当前 IP/区域受限；请换用 US 服务器或 US 出口 IP 后重试。",
                409,
            )

    return CodebuffError(
        f"{prefix}: {response.status_code} {text}",
        response.status_code,
    )


def _ad_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    return [
        {
            "role": _ad_message_role(message.get("role")),
            "content": _ad_message_content(message.get("content")),
        }
        for message in messages or []
    ]


def _ad_message_role(role: Any) -> str:
    if role == "developer":
        return "system"
    return str(role or "user")


def _ad_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = [
            str(part.get("text"))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content)
