import asyncio
import json
import unittest
from types import SimpleNamespace

from freebuff2api.app import (
    PreparedChatAttempt,
    _finalize_run_with_client,
    _start_freebuff_run_chain,
    _stream_openai_chunks,
)
from freebuff2api.codebuff import CodebuffError, FreebuffRun
from freebuff2api.config import Settings
from freebuff2api.models import resolve_model


class FakeClient:
    def __init__(self) -> None:
        self.recorded = False
        self.finished = False
        self.calls = []
        self.settings = Settings(
            codebuff_token="token-1",
            local_api_key=None,
            token_index=1,
            client_id="client-1",
            debug=False,
        )

    async def chat_events(self, payload):
        yield (
            'data: {"id":"chunk-1","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek/deepseek-v4-flash",'
            '"choices":[{"index":0,"delta":{"content":null,'
            '"reasoning_content":"hello"},"finish_reason":null}]}'
        )
        yield "data: [DONE]"

    async def record_run_step(self, *args, **kwargs) -> None:
        self.recorded = True
        self.calls.append(("step", args, kwargs))
        await asyncio.sleep(0)

    async def finish_run(self, *args, **kwargs) -> None:
        self.finished = True
        self.calls.append(("finish", args, kwargs))
        await asyncio.sleep(0)

    async def start_run(self, agent_id, ancestor_run_ids=None):
        run_id = f"run-{len([call for call in self.calls if call[0] == 'start']) + 1}"
        self.calls.append(("start", agent_id, ancestor_run_ids or [], run_id))
        await asyncio.sleep(0)
        return run_id


class FailingStreamClient(FakeClient):
    async def chat_events(self, payload):
        raise CodebuffError(
            "Codebuff chat failed: 403 hierarchy",
            502,
            token_index=self.settings.token_index,
            token_hint=self.settings.token_hint,
        )
        yield


class RetryStreamClient(FakeClient):
    def __init__(self, token_index: int, *, fail_chat: bool) -> None:
        super().__init__()
        self.settings = Settings(
            codebuff_token=f"token-{token_index}",
            local_api_key=None,
            token_index=token_index,
            client_id=f"client-{token_index}",
            debug=False,
        )
        self.fail_chat = fail_chat

    async def validate_agents(self) -> None:
        return None

    async def request_ad_chain(self, messages=None, surface=None) -> None:
        return None

    def schedule_ad_chain(self, messages=None) -> None:
        return None

    async def start_run(self, agent_id, ancestor_run_ids=None):
        return f"run-{self.settings.token_index}-{agent_id}"

    async def chat_events(self, payload):
        if self.fail_chat:
            raise CodebuffError(
                "Codebuff chat failed: 429 rate_limited",
                429,
                token_index=self.settings.token_index,
                token_hint=self.settings.token_hint,
            )
        yield (
            'data: {"id":"chunk-ok","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek/deepseek-v4-flash",'
            '"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}'
        )
        yield "data: [DONE]"


class FakeLease:
    def __init__(self, client, account_index: int) -> None:
        self.client = client
        self.session = SimpleNamespace(instance_id=f"session-{account_index}")
        self._account_index = account_index
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    @property
    def account_index(self) -> int:
        return self._account_index


class RetryAccounts:
    def __init__(self, leases) -> None:
        self.leases = leases
        self.account_count = len(leases)
        self.calls = []

    async def acquire_session(self, model: str, messages=None, *, exclude_account_indices=None):
        excluded = set(exclude_account_indices or set())
        self.calls.append((model, tuple(sorted(excluded))))
        for lease in self.leases:
            if lease.account_index in excluded:
                continue
            if not lease.closed:
                return lease
        raise CodebuffError("No remaining accounts available for retry", 503)


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_forwards_content_before_finalize(self) -> None:
        client = FakeClient()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    accounts=SimpleNamespace(account_count=1),
                    codebuff=client,
                    settings=Settings(
                        codebuff_token="token",
                        local_api_key=None,
                        debug=False,
                    ),
                )
            )
        )

        prepared = PreparedChatAttempt(
            lease=FakeLease(client, 0),
            run=FreebuffRun(
                run_id="run-1",
                agent_id="base2-free-deepseek-flash",
                started_at="2026-05-23T00:00:00.000Z",
            ),
            payload={},
        )

        chunks = []
        async for chunk in _stream_openai_chunks(
            request,
            {"model": "deepseek/deepseek-v4-flash", "messages": []},
            [],
            resolve_model("deepseek/deepseek-v4-flash"),
            prepared_attempt=prepared,
            attempt_number=1,
            max_attempts=1,
            request_id="test01",
        ):
            chunks.append(chunk.decode("utf-8"))

        first_payload = json.loads(chunks[0].removeprefix("data: ").strip())

        delta = first_payload["choices"][0]["delta"]
        self.assertNotIn("content", delta)
        self.assertEqual(delta["reasoning_content"], "hello")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

        await asyncio.sleep(0.05)
        self.assertTrue(client.recorded)
        self.assertTrue(client.finished)

    async def test_run_chain_records_pruner_before_chat(self) -> None:
        client = FakeClient()

        run = await _start_freebuff_run_chain(client, "base2-free-kimi")

        # closest to the SDK: the pre-generation context-pruner sub-run is
        # fully recorded before the chat; the parent run's own steps are
        # deferred to finalize (SDK records addAgentStep after the stream).
        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.child_run_id, "run-2")
        self.assertEqual(client.calls[0], ("start", "base2-free-kimi", [], "run-1"))
        self.assertEqual(
            client.calls[1],
            ("start", "context-pruner", ["run-1"], "run-2"),
        )
        self.assertEqual(client.calls[2][0], "step")
        self.assertEqual(client.calls[2][1], ("run-2",))
        self.assertEqual(client.calls[2][2]["step_number"], 1)
        self.assertEqual(client.calls[2][2]["child_run_ids"], [])
        self.assertEqual(client.calls[3], ("finish", ("run-2",), {"total_steps": 2}))
        # no parent step recorded yet -- those land after the stream
        self.assertEqual(len(client.calls), 4)

    async def test_finalize_records_parent_steps_after_stream(self) -> None:
        client = FakeClient()

        run = await _start_freebuff_run_chain(client, "base2-free-kimi")
        await _finalize_run_with_client(client, run, "msg-1")

        # parent step1 references the pruner child; step2 carries the message id
        self.assertEqual(
            client.calls[4],
            (
                "step",
                ("run-1",),
                {
                    "step_number": 1,
                    "child_run_ids": ["run-2"],
                    "message_id": None,
                    "start_time": run.started_at,
                },
            ),
        )
        self.assertEqual(
            client.calls[5],
            (
                "step",
                ("run-1",),
                {
                    "step_number": 2,
                    "child_run_ids": [],
                    "message_id": "msg-1",
                    "start_time": run.started_at,
                },
            ),
        )
        self.assertEqual(client.calls[6], ("finish", ("run-1",), {"total_steps": 3}))

    async def test_gemini_thinker_run_chain_uses_child_as_payload_run(self) -> None:
        client = FakeClient()

        run = await _start_freebuff_run_chain(
            client,
            resolve_model("google/gemini-3.1-pro-preview"),
        )

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.chat_run_id, "run-2")
        self.assertEqual(run.payload_run_id, "run-2")
        self.assertEqual(client.calls[0], ("start", "base2-free-kimi", [], "run-1"))
        self.assertEqual(
            client.calls[1],
            ("start", "thinker-with-files-gemini", ["run-1"], "run-2"),
        )

    async def test_gemini_flash_lite_run_chain_uses_session_root_parent(self) -> None:
        client = FakeClient()

        run = await _start_freebuff_run_chain(
            client,
            resolve_model("google/gemini-2.5-flash-lite"),
        )

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.chat_run_id, "run-2")
        self.assertEqual(run.payload_run_id, "run-2")
        self.assertEqual(
            client.calls[0],
            ("start", "base2-free-deepseek-flash", [], "run-1"),
        )
        self.assertEqual(
            client.calls[1],
            ("start", "file-picker", ["run-1"], "run-2"),
        )

    async def test_streaming_codebuff_error_is_returned_as_sse_error(self) -> None:
        client = FailingStreamClient()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    accounts=SimpleNamespace(account_count=1),
                    codebuff=client,
                    settings=Settings(
                        codebuff_token="token",
                        local_api_key=None,
                        debug=False,
                    ),
                )
            )
        )

        prepared = PreparedChatAttempt(
            lease=FakeLease(client, 0),
            run=FreebuffRun(
                run_id="run-1",
                agent_id="base2-free-deepseek-flash",
                started_at="2026-05-23T00:00:00.000Z",
            ),
            payload={},
        )

        chunks = []
        with self.assertLogs("freebuff2api.app", level="WARNING"):
            async for chunk in _stream_openai_chunks(
                request,
                {"model": "deepseek/deepseek-v4-flash", "messages": []},
                [],
                resolve_model("deepseek/deepseek-v4-flash"),
                prepared_attempt=prepared,
                attempt_number=1,
                max_attempts=1,
                request_id="test02",
            ):
                chunks.append(chunk.decode("utf-8"))

        error_payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(error_payload["error"]["code"], "codebuff_error")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

    async def test_stream_retries_next_account_before_first_chunk(self) -> None:
        failing = RetryStreamClient(1, fail_chat=True)
        succeeding = RetryStreamClient(2, fail_chat=False)
        accounts = RetryAccounts(
            [
                FakeLease(failing, 0),
                FakeLease(succeeding, 1),
            ]
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    accounts=accounts,
                    codebuff=failing,
                    settings=Settings(
                        codebuff_token="token-1,token-2",
                        local_api_key=None,
                        client_id="local-client",
                        debug=False,
                    ),
                )
            )
        )

        prepared = PreparedChatAttempt(
            lease=accounts.leases[0],
            run=FreebuffRun(
                run_id="run-1",
                agent_id="base2-free-deepseek-flash",
                started_at="2026-05-23T00:00:00.000Z",
            ),
            payload={"messages": []},
        )

        chunks = []
        with self.assertLogs("freebuff2api.app", level="INFO") as logs:
            async for chunk in _stream_openai_chunks(
                request,
                {"model": "deepseek/deepseek-v4-flash", "messages": []},
                [],
                resolve_model("deepseek/deepseek-v4-flash"),
                prepared_attempt=prepared,
                attempt_number=1,
                max_attempts=2,
                request_id="test03",
            ):
                chunks.append(chunk.decode("utf-8"))

        first_payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(first_payload["choices"][0]["delta"]["content"], "hello")
        self.assertEqual(accounts.calls, [("deepseek/deepseek-v4-flash", (0,))])
        self.assertTrue(
            any("retrying chat completion stage=stream" in line for line in logs.output)
        )


if __name__ == "__main__":
    unittest.main()
