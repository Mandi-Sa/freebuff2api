from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
from typing import Any, AsyncIterator
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .codebuff import (
    CodebuffAccountLease,
    CodebuffAccountPool,
    CodebuffClient,
    CodebuffError,
    FreebuffRun,
    SessionManager,
    utc_now_iso,
)
from .config import Settings, load_settings
from .logging_config import (
    configure_logging,
    redact_headers,
    render_debug,
    set_request_id,
    set_token_context,
)
from .openai_compat import (
    CompletionAccumulator,
    build_upstream_payload,
    normalize_chat_messages,
    sanitize_stream_chunk,
)
from .models import CONTEXT_PRUNER_AGENT_ID, FreebuffModel, models_response, resolve_model
from .sse import decode_sse_data, encode_sse


logger = logging.getLogger("freebuff2api.app")


@dataclass
class PreparedChatAttempt:
    lease: CodebuffAccountLease
    run: FreebuffRun
    payload: dict[str, Any]

    @property
    def client(self) -> CodebuffClient:
        return self.lease.client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(settings)
    accounts = CodebuffAccountPool(settings)
    app.state.settings = settings
    app.state.accounts = accounts
    app.state.codebuff = accounts.default_client
    app.state.sessions = accounts.default_sessions
    accounts.start_scheduler()
    logger.info("configured freebuff accounts count=%s", accounts.account_count)
    try:
        yield
    finally:
        await accounts.aclose()


app = FastAPI(title="freebuff2api", version="0.1.0", lifespan=lifespan)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _client(request: Request) -> CodebuffClient:
    return request.app.state.codebuff


def _sessions(request: Request) -> SessionManager:
    return request.app.state.sessions


def _accounts(request: Request) -> CodebuffAccountPool:
    return request.app.state.accounts


def _check_local_auth(request: Request) -> None:
    api_key = _settings(request).local_api_key
    if not api_key:
        return
    expected = f"Bearer {api_key}"
    if request.headers.get("authorization") != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _error_response(error: Exception) -> JSONResponse:
    if isinstance(error, CodebuffError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "message": str(error),
                    "type": "upstream_error",
                    "code": "codebuff_error",
                }
            },
        )
    raise error


def _token_index(client: CodebuffClient | None) -> int:
    settings = getattr(client, "settings", None)
    if settings is None:
        return 0
    return settings.token_index


def _token_hint(client: CodebuffClient | None) -> str:
    settings = getattr(client, "settings", None)
    if settings is None:
        return "unknown"
    return settings.token_hint


def _error_token_index(error: CodebuffError) -> int:
    return error.token_index or 0


def _error_token_hint(error: CodebuffError) -> str:
    return error.token_hint or "unknown"


def _max_account_attempts(request: Request) -> int:
    return max(1, _accounts(request).account_count)


def _bind_lease_log_context(
    request: Request,
    lease: CodebuffAccountLease,
    model_config: FreebuffModel,
) -> None:
    settings = _settings(request)
    count = _accounts(request).account_count
    mode = "U" if model_config.session_id in settings.unlimited_models else "P"
    set_token_context(f"t{lease.client.settings.token_index}/{count}", mode)


def _retry_log(
    stage: str,
    *,
    attempt: int,
    max_attempts: int,
    model: str,
    client: CodebuffClient | None = None,
) -> None:
    logger.info(
        "retrying chat completion stage=%s failed_token_index=%s failed_token=%s next_attempt=%s/%s model=%s",
        stage,
        _token_index(client),
        _token_hint(client),
        attempt + 1,
        max_attempts,
        model,
    )


async def _prepare_chat_attempt(
    request: Request,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    model_config: FreebuffModel,
    *,
    exclude_account_indices: set[int] | None = None,
) -> PreparedChatAttempt:
    settings = _settings(request)
    lease: CodebuffAccountLease | None = None
    prepared = False
    stage = "acquire_session"
    try:
        lease = await _accounts(request).acquire_session(
            model_config.session_id,
            messages=messages,
            exclude_account_indices=exclude_account_indices,
        )
        client = lease.client
        stage = "run_setup"
        await client.request_ad_chain(messages=messages)
        await client.validate_agents()
        run = await _start_freebuff_run_chain(client, model_config)
        trace_session_id = str(uuid.uuid4())
        payload = build_upstream_payload(
            {**body, "messages": messages},
            session=lease.session,
            run_id=run.payload_run_id,
            client_id=settings.client_id,
            trace_session_id=trace_session_id,
            upstream_model_id=model_config.upstream_id,
        )
        if settings.debug:
            logger.debug(
                "prepared upstream chat trace=%s run=%s payload=%s",
                trace_session_id,
                run,
                render_debug(payload, settings.log_body_chars),
            )
        prepared = True
        return PreparedChatAttempt(lease=lease, run=run, payload=payload)
    except CodebuffError as error:
        if lease is not None:
            if error.token_index is None:
                error.token_index = lease.client.settings.token_index
            if error.token_hint is None:
                error.token_hint = lease.client.settings.token_hint
            if error.account_index is None:
                error.account_index = lease.account_index
        logger.warning(
            "chat completion prepare failed stage=%s token_index=%s token=%s: %s",
            stage,
            (
                _token_index(lease.client)
                if lease is not None
                else _error_token_index(error)
            ),
            (
                _token_hint(lease.client)
                if lease is not None
                else _error_token_hint(error)
            ),
            error,
            exc_info=settings.debug,
        )
        raise
    except Exception:
        logger.exception(
            "chat completion prepare failed stage=%s token_index=%s token=%s",
            stage,
            _token_index(lease.client if lease is not None else None),
            _token_hint(lease.client if lease is not None else None),
        )
        raise
    finally:
        if lease is not None and not prepared:
            await lease.aclose()


def _stream_error_payload(error: CodebuffError) -> dict[str, dict[str, str]]:
    return {
        "error": {
            "message": str(error),
            "type": "upstream_error",
            "code": "codebuff_error",
        }
    }


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    _check_local_auth(request)
    return {"status": "ok"}


@app.get("/admin/quota")
async def admin_quota(request: Request) -> dict[str, Any]:
    _check_local_auth(request)
    return {"tokens": _accounts(request).quota.snapshot()}


@app.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    _check_local_auth(request)
    settings = _settings(request)
    data = models_response()
    data["data"] = [
        model
        for model in data["data"]
        if settings.is_allowed(resolve_model(model["id"]).session_id)
    ]
    return data


def _allowed_models_hint(settings: Settings) -> str:
    unlimited = ", ".join(settings.unlimited_models) or "(none)"
    return f"{settings.premium_model} (premium); {unlimited} (unlimited)"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _check_local_auth(request)
    body = await request.json()
    settings = _settings(request)
    request_id = uuid.uuid4().hex[:6]
    set_request_id(request_id)
    try:
        model_config = resolve_model(body.get("model"))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    model = model_config.id
    # gate on the underlying session model, so models that ride an allowed host
    # (e.g. gemini-3.1-pro -> kimi, gemini flash-lite -> deepseek-flash) pass.
    session_model = model_config.session_id
    if not settings.is_allowed(session_model):
        logger.info("chat completion rejected model=%s reason=not_allowed", model)
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not allowed. Allowed: {_allowed_models_hint(settings)}",
        )
    logger.info(
        "chat completion request model=%s stream=%s messages=%s",
        model,
        body.get("stream") is True,
        len(body.get("messages") or []),
    )
    if settings.debug:
        logger.debug(
            "incoming request headers=%s",
            redact_headers(dict(request.headers)),
        )
        logger.debug(
            "chat completion request body=%s",
            render_debug(body, settings.log_body_chars),
        )

    messages = normalize_chat_messages(body.get("messages"))
    max_attempts = _max_account_attempts(request)
    excluded_accounts: set[int] = set()

    for attempt in range(1, max_attempts + 1):
        try:
            prepared = await _prepare_chat_attempt(
                request,
                body,
                messages,
                model_config,
                exclude_account_indices=excluded_accounts,
            )
        except CodebuffError as error:
            if attempt < max_attempts:
                if error.account_index is not None:
                    excluded_accounts.add(error.account_index)
                _retry_log(
                    "prepare",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model=model,
                    client=None,
                )
                continue
            return _error_response(error)
        except Exception as error:
            return _error_response(error)

        if body.get("stream") is True:
            return StreamingResponse(
                _stream_openai_chunks(
                    request,
                    body,
                    messages,
                    model_config,
                    prepared_attempt=prepared,
                    attempt_number=attempt,
                    max_attempts=max_attempts,
                    request_id=request_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            response = await _collect_completion(
                request,
                prepared.payload,
                prepared.run,
                model,
                client=prepared.client,
            )
            return JSONResponse(response)
        except CodebuffError as error:
            logger.warning(
                "chat completion failed token_index=%s token=%s attempt=%s/%s run_id=%s: %s",
                _token_index(prepared.client),
                _token_hint(prepared.client),
                attempt,
                max_attempts,
                prepared.run.run_id,
                error,
                exc_info=settings.debug,
            )
            if attempt < max_attempts:
                excluded_accounts.add(prepared.lease.account_index)
                _retry_log(
                    "completion",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model=model,
                    client=prepared.client,
                )
                continue
            return _error_response(error)
        except Exception as error:
            logger.exception(
                "chat completion failed token_index=%s token=%s attempt=%s/%s run_id=%s",
                _token_index(prepared.client),
                _token_hint(prepared.client),
                attempt,
                max_attempts,
                prepared.run.run_id,
            )
            return _error_response(error)
        finally:
            await prepared.lease.aclose()

    raise RuntimeError("chat completion retries exhausted without response")


async def _stream_openai_chunks(
    request: Request,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    model_config: FreebuffModel,
    *,
    prepared_attempt: PreparedChatAttempt,
    attempt_number: int,
    max_attempts: int,
    request_id: str,
) -> AsyncIterator[bytes]:
    settings = _settings(request)
    attempt = attempt_number
    current = prepared_attempt
    set_request_id(request_id)
    _bind_lease_log_context(request, current.lease, model_config)
    excluded_accounts = {current.lease.account_index}

    while True:
        message_id: str | None = None
        emitted_chunk = False
        should_retry = False
        try:
            async for line in current.client.chat_events(current.payload):
                data = decode_sse_data(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    if settings.debug:
                        logger.debug(
                            "chat stream done run_id=%s message_id=%s",
                            current.run.run_id,
                            message_id,
                        )
                    yield encode_sse("[DONE]")
                    return

                message_id = data.get("id") or message_id
                chunk = sanitize_stream_chunk(data)
                if chunk is not None:
                    emitted_chunk = True
                    if settings.debug:
                        logger.debug(
                            "chat stream chunk=%s",
                            render_debug(chunk, settings.log_body_chars),
                        )
                    yield encode_sse(chunk)
                elif settings.debug:
                    logger.debug(
                        "chat stream ignored data=%s",
                        render_debug(data, settings.log_body_chars),
                    )
            return
        except CodebuffError as error:
            should_retry = not emitted_chunk and attempt < max_attempts
            logger.warning(
                "chat stream failed token_index=%s token=%s attempt=%s/%s run_id=%s emitted_chunk=%s: %s",
                _token_index(current.client),
                _token_hint(current.client),
                attempt,
                max_attempts,
                current.run.run_id,
                emitted_chunk,
                error,
                exc_info=settings.debug,
            )
            if not should_retry:
                yield encode_sse(_stream_error_payload(error))
                yield encode_sse("[DONE]")
                return
            _retry_log(
                "stream",
                attempt=attempt,
                max_attempts=max_attempts,
                model=model_config.id,
                client=current.client,
            )
        finally:
            _schedule_finalize_run(current.client, current.run, message_id)
            await current.lease.aclose()

        attempt += 1
        while True:
            try:
                current = await _prepare_chat_attempt(
                    request,
                    body,
                    messages,
                    model_config,
                    exclude_account_indices=excluded_accounts,
                )
                excluded_accounts.add(current.lease.account_index)
                break
            except CodebuffError as error:
                if error.account_index is not None:
                    excluded_accounts.add(error.account_index)
                if attempt >= max_attempts:
                    yield encode_sse(_stream_error_payload(error))
                    yield encode_sse("[DONE]")
                    return
                _retry_log(
                    "prepare",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model=model_config.id,
                )
                attempt += 1


async def _collect_completion(
    request: Request,
    payload: dict[str, Any],
    run: FreebuffRun,
    model: str,
    *,
    client: CodebuffClient | None = None,
) -> dict[str, Any]:
    message_id: str | None = None
    accumulator = CompletionAccumulator(model)
    client = client or _client(request)
    try:
        async for line in client.chat_events(payload):
            data = decode_sse_data(line)
            if data is None:
                continue
            if data == "[DONE]":
                break
            message_id = data.get("id") or message_id
            accumulator.add(data)
        response = accumulator.final_response()
        logger.info(
            "chat completion response run_id=%s message_id=%s content_chars=%s finish_reason=%s",
            run.run_id,
            message_id,
            len(response["choices"][0]["message"].get("content") or ""),
            response["choices"][0].get("finish_reason"),
        )
        if _settings(request).debug:
            logger.debug(
                "chat completion response body=%s",
                render_debug(response, _settings(request).log_body_chars),
            )
        return response
    finally:
        await _finalize_run(request, run, message_id, client=client)


async def _start_freebuff_run_chain(
    client: CodebuffClient,
    model: FreebuffModel | str,
) -> FreebuffRun:
    if isinstance(model, str):
        model = FreebuffModel(model, model)
    if model.parent_agent_id:
        return await _start_child_chat_run_chain(client, model)

    agent_id = model.agent_id
    started_at = utc_now_iso()
    run_id = await client.start_run(agent_id)
    child_started_at = utc_now_iso()
    child_run_id = await client.start_run(
        CONTEXT_PRUNER_AGENT_ID,
        ancestor_run_ids=[run_id],
    )
    await client.record_run_step(
        child_run_id,
        step_number=1,
        child_run_ids=[],
        message_id=None,
        start_time=child_started_at,
    )
    await client.finish_run(child_run_id, total_steps=2)
    await client.record_run_step(
        run_id,
        step_number=1,
        child_run_ids=[child_run_id],
        message_id=None,
        start_time=started_at,
    )
    return FreebuffRun(
        run_id=run_id,
        agent_id=agent_id,
        started_at=started_at,
        child_run_id=child_run_id,
    )


async def _start_child_chat_run_chain(
    client: CodebuffClient,
    model: FreebuffModel,
) -> FreebuffRun:
    assert model.parent_agent_id is not None

    started_at = utc_now_iso()
    parent_run_id = await client.start_run(model.parent_agent_id)
    chat_started_at = utc_now_iso()
    chat_run_id = await client.start_run(
        model.agent_id,
        ancestor_run_ids=[parent_run_id],
    )
    return FreebuffRun(
        run_id=parent_run_id,
        agent_id=model.parent_agent_id,
        started_at=started_at,
        child_run_id=chat_run_id,
        chat_run_id=chat_run_id,
        chat_started_at=chat_started_at,
    )


async def _finalize_run(
    request: Request,
    run: FreebuffRun,
    message_id: str | None,
    *,
    client: CodebuffClient | None = None,
) -> None:
    await _finalize_run_with_client(client or _client(request), run, message_id)


def _schedule_finalize_run(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    task = asyncio.create_task(_finalize_run_with_client(client, run, message_id))

    def _log_background_error(done: asyncio.Task[None]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            logger.debug("background finalize task cancelled run_id=%s", run.run_id)
        except Exception:
            logger.exception("background finalize task failed run_id=%s", run.run_id)

    task.add_done_callback(_log_background_error)


async def _finalize_run_with_client(
    client: CodebuffClient,
    run: FreebuffRun,
    message_id: str | None,
) -> None:
    try:
        logger.debug(
            "finalize run start run_id=%s message_id=%s started_at=%s",
            run.run_id,
            message_id,
            run.started_at,
        )
        if run.chat_run_id and run.chat_run_id != run.run_id:
            await client.record_run_step(
                run.chat_run_id,
                step_number=1,
                child_run_ids=[],
                message_id=message_id,
                start_time=run.chat_started_at or run.started_at,
            )
            await client.finish_run(run.chat_run_id, total_steps=2)
            await client.record_run_step(
                run.run_id,
                step_number=1,
                child_run_ids=[run.chat_run_id],
                message_id=None,
                start_time=run.started_at,
            )
            await client.finish_run(run.run_id, total_steps=2)
            logger.debug("finalize parent/child run done run_id=%s", run.run_id)
            return

        await client.record_run_step(
            run.run_id,
            step_number=2,
            child_run_ids=[],
            message_id=message_id,
            start_time=run.started_at,
        )
        await client.finish_run(run.run_id, total_steps=3)
        logger.debug("finalize run done run_id=%s", run.run_id)
    except CodebuffError as error:
        logger.warning(
            "finalize run failed token_index=%s token=%s run_id=%s: %s",
            _token_index(client),
            _token_hint(client),
            run.run_id,
            error,
            exc_info=client.settings.debug,
        )
    except Exception:
        logger.exception("finalize run failed run_id=%s", run.run_id)
