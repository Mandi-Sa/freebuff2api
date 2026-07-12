import json
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from freebuff2api.app import _error_response, _finalize_run_with_client, app
from freebuff2api.codebuff import CodebuffError, CodebuffClient, FreebuffRun
from freebuff2api.config import Settings
from freebuff2api.models import resolve_model


class FinalizeFailingClient:
    def __init__(self) -> None:
        self.settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            debug=False,
        )

    async def record_run_step(self, *args, **kwargs) -> None:
        raise CodebuffError("network error", 502)

    async def finish_run(self, *args, **kwargs) -> None:
        raise AssertionError("finish_run should not be called")


class FakeLease:
    def __init__(self, client, session, account_index: int) -> None:
        self.client = client
        self.session = session
        self._account_index = account_index
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    @property
    def account_index(self) -> int:
        return self._account_index


class FakeAccounts:
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


class RetryClient(CodebuffClient):
    def __init__(self, settings: Settings, *, fail_chat: bool = False) -> None:
        self.settings = settings
        self.fail_chat = fail_chat
        self.recorded = []

    async def validate_agents(self) -> None:
        return None

    async def request_ad_chain(self, messages=None, surface=None) -> None:
        return None

    async def start_run(self, agent_id, ancestor_run_ids=None) -> str:
        return f"run-{self.settings.token_index}-{agent_id}"

    async def record_run_step(self, run_id, **kwargs) -> None:
        self.recorded.append(("step", run_id, kwargs))

    async def finish_run(self, run_id, **kwargs) -> None:
        self.recorded.append(("finish", run_id, kwargs))

    async def chat_events(self, payload):
        if self.fail_chat:
            raise CodebuffError(
                "Codebuff chat failed: 429 rate_limited",
                429,
                token_index=self.settings.token_index,
                token_hint=self.settings.token_hint,
            )
        yield (
            'data: {"id":"chunk-1","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek/deepseek-v4-flash",'
            '"choices":[{"index":0,"delta":{"content":"hello"},'
            '"finish_reason":null}]}'
        )
        yield (
            'data: {"id":"chunk-1","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek/deepseek-v4-flash",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        )
        yield "data: [DONE]"


class AppErrorTests(unittest.TestCase):
    def test_chat_completion_rejects_disallowed_model(self) -> None:
        app.state.settings = Settings(
            codebuff_token="t1",
            local_api_key=None,
            unlimited_model="deepseek/deepseek-v4-flash",
            premium_model="moonshotai/kimi-k2.7-code",
        )
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "deepseek/deepseek-v4-pro",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not allowed", response.json()["detail"])

    def test_list_models_only_returns_allowed(self) -> None:
        app.state.settings = Settings(
            codebuff_token="t1",
            local_api_key=None,
            unlimited_model="deepseek/deepseek-v4-flash,minimax/minimax-m3",
            premium_model="moonshotai/kimi-k2.7-code",
        )
        response = TestClient(app).get("/v1/models")
        ids = {m["id"] for m in response.json()["data"]}
        self.assertEqual(
            ids,
            {
                "moonshotai/kimi-k2.7-code",
                "deepseek/deepseek-v4-flash",
                "minimax/minimax-m3",
                # gemini variants ride allowed hosts (kimi / deepseek-flash)
                "google/gemini-2.5-flash-lite",
                "google/gemini-3.1-flash-lite-preview",
                "google/gemini-3.1-pro-preview",
            },
        )

    def test_codebuff_error_returns_original_status_code(self) -> None:
        response = _error_response(CodebuffError("rate limited", 429))
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(body["error"]["message"], "rate limited")
        self.assertEqual(body["error"]["type"], "upstream_error")

    def test_finalize_codebuff_error_logs_warning_without_raising(self) -> None:
        client = FinalizeFailingClient()
        run = FreebuffRun(
            run_id="run-1",
            agent_id="agent-1",
            started_at="2026-05-24T00:00:00.000Z",
        )

        with self.assertLogs("freebuff2api.app", level="WARNING") as logs:
            self.asyncio_run(_finalize_run_with_client(client, run, None))

        self.assertIn("token_index=1", logs.output[0])
        self.assertIn("token=***oken", logs.output[0])
        self.assertIn("finalize run failed", logs.output[0])
        self.assertIn("run_id=run-1: network error", logs.output[0])

    def test_non_stream_request_retries_next_account_before_returning_error(self) -> None:
        failing = RetryClient(
            Settings(
                codebuff_token="first-token-1234",
                local_api_key=None,
                token_index=1,
                client_id="client-1",
                debug=False,
            ),
            fail_chat=True,
        )
        succeeding = RetryClient(
            Settings(
                codebuff_token="second-token-5678",
                local_api_key=None,
                token_index=2,
                client_id="client-2",
                debug=False,
            ),
            fail_chat=False,
        )
        accounts = FakeAccounts(
            [
                FakeLease(failing, SimpleNamespace(instance_id="session-1"), 0),
                FakeLease(succeeding, SimpleNamespace(instance_id="session-2"), 1),
            ]
        )

        app.state.settings = Settings(
            codebuff_token="first-token-1234,second-token-5678",
            local_api_key=None,
            client_id="local-client",
            debug=False,
        )
        app.state.accounts = accounts
        app.state.codebuff = failing
        app.state.sessions = None

        with self.assertLogs("freebuff2api.app", level="INFO") as logs:
            response = TestClient(app).post(
                "/v1/chat/completions",
                json={
                    "model": resolve_model("deepseek/deepseek-v4-flash").id,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "hello")
        self.assertEqual(
            accounts.calls,
            [
                ("deepseek/deepseek-v4-flash", ()),
                ("deepseek/deepseek-v4-flash", (0,)),
            ],
        )
        self.assertTrue(
            any("retrying chat completion stage=completion" in line for line in logs.output)
        )

    def asyncio_run(self, awaitable) -> None:
        import asyncio

        asyncio.run(awaitable)


if __name__ == "__main__":
    unittest.main()
