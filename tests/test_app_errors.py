import json
import unittest

from freebuff2api.app import _error_response, _finalize_run_with_client
from freebuff2api.codebuff import CodebuffError, FreebuffRun
from freebuff2api.config import Settings


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


class AppErrorTests(unittest.TestCase):
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

        self.assertIn("finalize run failed run_id=run-1: network error", logs.output[0])

    def asyncio_run(self, awaitable) -> None:
        import asyncio

        asyncio.run(awaitable)


if __name__ == "__main__":
    unittest.main()
