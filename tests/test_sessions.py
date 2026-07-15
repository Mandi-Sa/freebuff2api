import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from freebuff2api.codebuff import (
    CodebuffAccountPool,
    CodebuffError,
    FreebuffSession,
    SessionManager,
    _token_window_index,
)
from freebuff2api.config import Settings


class SwitchModelClient:
    def __init__(self) -> None:
        self.deleted = False
        self.calls = []

    async def get_session(self, instance_id=None):
        self.calls.append(("get_session", instance_id))
        if self.deleted:
            return {"status": "none"}
        return {
            "status": "active",
            "instanceId": "deepseek-instance",
            "model": "deepseek/deepseek-v4-pro",
            "expiresAt": "2026-05-23T15:27:34.581Z",
            "remainingMs": 3_000_000,
        }

    async def delete_session(self) -> None:
        self.calls.append(("delete_session",))
        self.deleted = True

    async def request_ad_chain(self, messages=None, *, surface=None) -> None:
        self.calls.append(("request_ad_chain", messages or [], surface))

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        self.calls.append(("request_ads", provider, messages or [], surface))
        return {"ads": []}

    async def get_streak(self) -> dict:
        self.calls.append(("get_streak",))
        return {"streak": 0}

    async def report_zeroclick_impressions(self, ids) -> None:
        self.calls.append(("report_zeroclick_impressions", ids))

    async def report_codebuff_impression(self, imp_url) -> None:
        self.calls.append(("report_codebuff_impression", imp_url))

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        if not self.deleted:
            raise CodebuffError(
                'Codebuff request failed: 409 {"status":"model_locked"}',
                502,
            )
        return FreebuffSession(
            instance_id="kimi-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class LeaseSwitchModelClient:
    def __init__(self) -> None:
        self.current_model = "deepseek/deepseek-v4-flash"
        self.calls = []

    async def get_session(self, instance_id=None):
        self.calls.append(("get_session", instance_id, self.current_model))
        return {
            "status": "active",
            "instanceId": f"{self.current_model}-instance",
            "model": self.current_model,
            "remainingMs": 3_000_000,
        }

    async def delete_session(self) -> None:
        self.calls.append(("delete_session", self.current_model))
        self.current_model = ""

    async def request_ad_chain(self, messages=None, *, surface=None) -> None:
        self.calls.append(("request_ad_chain", messages or [], surface))

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        self.calls.append(("request_ads", provider, messages or [], surface))
        return {"ads": []}

    async def get_streak(self) -> dict:
        self.calls.append(("get_streak",))
        return {"streak": 0}

    async def report_zeroclick_impressions(self, ids) -> None:
        self.calls.append(("report_zeroclick_impressions", ids))

    async def report_codebuff_impression(self, imp_url) -> None:
        self.calls.append(("report_codebuff_impression", imp_url))

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        self.current_model = model
        return FreebuffSession(
            instance_id=f"{model}-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class PoolClient:
    def __init__(self, settings, quota_store=None) -> None:
        self.settings = settings
        self.quota_store = quota_store
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def get_session(self, instance_id=None):
        token = self.settings.codebuff_token
        return {
            "status": "active",
            "instanceId": f"{token}-instance",
            "model": "deepseek/deepseek-v4-flash",
            "remainingMs": 3_000_000,
        }


class FailingPoolClient(PoolClient):
    async def get_session(self, instance_id=None):
        return {"status": "none"}

    async def request_ad_chain(self, messages=None, *, surface=None) -> None:
        return None

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        return {"ads": []}

    async def create_session(self, model):
        raise CodebuffError("Codebuff request failed: 429 rate_limited", 429)


class ParkingPoolClient(PoolClient):
    parked_models: list[str] = []
    deleted_tokens: list[str] = []

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        return {"ads": []}

    async def delete_session(self) -> None:
        ParkingPoolClient.deleted_tokens.append(self.settings.codebuff_token)

    async def create_session(self, model):
        ParkingPoolClient.parked_models.append(model)
        return FreebuffSession(
            instance_id=f"{model}-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class IdlePoolClient(PoolClient):
    deleted: list[str] = []

    async def get_session(self, instance_id=None):
        return {
            "status": "active",
            "instanceId": f"{self.settings.codebuff_token}-instance",
            "model": "deepseek/deepseek-v4-pro",
            "remainingMs": 3_000_000,
        }

    async def delete_session(self) -> None:
        IdlePoolClient.deleted.append(self.settings.codebuff_token)


class ReuseFastPathClient:
    """Records every upstream call so tests can assert the confirm GET is skipped."""

    def __init__(self) -> None:
        self.calls: list = []

    async def get_session(self, instance_id=None):
        self.calls.append(("get_session", instance_id))
        return {
            "status": "active",
            "instanceId": instance_id,
            "model": "deepseek/deepseek-v4-flash",
            "remainingMs": 3_600_000,
        }

    async def delete_session(self) -> None:
        self.calls.append(("delete_session",))

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        self.calls.append(("request_ads", provider))
        return {"ads": []}

    async def get_streak(self) -> dict:
        self.calls.append(("get_streak",))
        return {"streak": 0}

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        return FreebuffSession(instance_id="new", model=model, remaining_ms=3_600_000)


def _iso_in(**delta) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(**delta))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class ReuseFastPathTests(unittest.IsolatedAsyncioTestCase):
    MODEL = "deepseek/deepseek-v4-flash"

    def _seed(self, manager, expires_at) -> None:
        manager._sessions[self.MODEL] = FreebuffSession(
            instance_id="live",
            model=self.MODEL,
            expires_at=expires_at,
            remaining_ms=3_600_000,
        )

    async def test_reuse_skips_confirm_when_expiry_far(self):
        client = ReuseFastPathClient()
        manager = SessionManager(
            client, Settings(codebuff_token="t", local_api_key=None)
        )
        self._seed(manager, _iso_in(hours=1))

        session = await manager.ensure_session(self.MODEL)

        self.assertEqual(session.instance_id, "live")
        # far-future expiry -> reuse with zero upstream round-trips
        self.assertEqual(client.calls, [])

    async def test_reuse_confirms_when_expiry_near(self):
        client = ReuseFastPathClient()
        manager = SessionManager(
            client, Settings(codebuff_token="t", local_api_key=None)
        )
        self._seed(manager, _iso_in(seconds=60))  # inside the 300s margin

        session = await manager.ensure_session(self.MODEL)

        self.assertEqual(session.instance_id, "live")
        self.assertEqual(client.calls, [("get_session", "live")])

    async def test_margin_zero_always_confirms(self):
        client = ReuseFastPathClient()
        manager = SessionManager(
            client,
            Settings(
                codebuff_token="t",
                local_api_key=None,
                session_reuse_confirm_margin=0,
            ),
        )
        self._seed(manager, _iso_in(hours=1))

        await manager.ensure_session(self.MODEL)

        # margin 0 disables the fast path -> always confirm upstream
        self.assertEqual(client.calls, [("get_session", "live")])


class SessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_model_deletes_active_upstream_session_before_create(self):
        client = SwitchModelClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        session = await manager.ensure_session("moonshotai/kimi-k2.7-code")

        self.assertEqual(session.instance_id, "kimi-instance")
        self.assertEqual(session.model, "moonshotai/kimi-k2.7-code")
        self.assertEqual(
            client.calls,
            [
                ("get_session", None),
                ("delete_session",),
                ("request_ads", "gravity", [], "waiting_room"),
                ("request_ads", "carbon", [], "waiting_room"),
                ("create_session", "moonshotai/kimi-k2.7-code"),
            ],
        )

    async def test_session_lease_blocks_model_switch_until_chat_releases(self):
        client = LeaseSwitchModelClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        first = await manager.acquire_session("deepseek/deepseek-v4-flash")
        started = asyncio.Event()

        async def acquire_second():
            started.set()
            return await manager.acquire_session("moonshotai/kimi-k2.7-code")

        task = asyncio.create_task(acquire_second())
        await started.wait()
        await asyncio.sleep(0.05)

        self.assertFalse(task.done())
        self.assertNotIn(
            ("delete_session", "deepseek/deepseek-v4-flash"),
            client.calls,
        )

        await first.aclose()
        second = await asyncio.wait_for(task, timeout=1)
        try:
            self.assertEqual(second.session.model, "moonshotai/kimi-k2.7-code")
            self.assertIn(
                ("delete_session", "deepseek/deepseek-v4-flash"),
                client.calls,
            )
        finally:
            await second.aclose()

    async def test_account_pool_uses_next_free_token_for_concurrent_requests(self):
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
        )

        with patch("freebuff2api.codebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(settings)
            first = await pool.acquire_session("deepseek/deepseek-v4-flash")
            second = await pool.acquire_session("deepseek/deepseek-v4-flash")
            try:
                self.assertEqual(
                    {
                        first.client.settings.codebuff_token,
                        second.client.settings.codebuff_token,
                    },
                    {"token-a", "token-b"},
                )
                self.assertNotEqual(
                    first.session.instance_id,
                    second.session.instance_id,
                )
            finally:
                await second.aclose()
                await first.aclose()
                await pool.aclose()

    async def test_account_pool_logs_account_label_when_session_acquire_fails(self):
        settings = Settings(
            codebuff_token="first-token-1234",
            local_api_key=None,
        )

        with patch("freebuff2api.codebuff.CodebuffClient", FailingPoolClient):
            pool = CodebuffAccountPool(settings)
            try:
                with self.assertLogs("freebuff2api.codebuff", level="WARNING") as logs:
                    with self.assertRaises(CodebuffError):
                        await pool.acquire_session("deepseek/deepseek-v4-flash")
            finally:
                await pool.aclose()

        self.assertIn("token_index=1", logs.output[0])
        self.assertIn("token=***1234", logs.output[0])
        self.assertIn("429 rate_limited", logs.output[0])

    def test_token_window_index_splits_day_evenly(self):
        self.assertEqual(_token_window_index(datetime(2026, 6, 21, 0, 0, 0), 5), 0)
        self.assertEqual(_token_window_index(datetime(2026, 6, 21, 5, 0, 0), 5), 1)
        self.assertEqual(_token_window_index(datetime(2026, 6, 21, 23, 59, 59), 5), 4)
        self.assertEqual(_token_window_index(datetime(2026, 6, 21, 12, 0, 0), 1), 0)

    async def test_window_switch_deletes_previous_premium_session(self):
        IdlePoolClient.deleted = []
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="moonshotai/kimi-k2.7-code",
            premium_model="deepseek/deepseek-v4-pro",
            session_block_seconds=999,
            destroy_lead_seconds=0,
        )
        windows = iter([0, 1])

        with patch("freebuff2api.codebuff.CodebuffClient", IdlePoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index",
                lambda now, count: next(windows),
            ):
                pool = CodebuffAccountPool(settings)
                first = await pool.acquire_session("deepseek/deepseek-v4-pro")
                self.assertTrue(pool._accounts[0].holds_premium)
                await first.aclose()
                second = await pool.acquire_session("deepseek/deepseek-v4-pro")
                await second.aclose()
                await asyncio.sleep(0.05)
                await pool.aclose()

        self.assertEqual(second.client.settings.codebuff_token, "token-b")
        self.assertIn("token-a", IdlePoolClient.deleted)

    async def test_window_switch_keeps_previous_unlimited_session(self):
        IdlePoolClient.deleted = []
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="deepseek/deepseek-v4-pro",
            premium_model="moonshotai/kimi-k2.7-code",
        )
        windows = iter([0, 1])

        with patch("freebuff2api.codebuff.CodebuffClient", IdlePoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index",
                lambda now, count: next(windows),
            ):
                pool = CodebuffAccountPool(settings)
                first = await pool.acquire_session("deepseek/deepseek-v4-pro")
                self.assertFalse(pool._accounts[0].holds_premium)
                await first.aclose()
                second = await pool.acquire_session("deepseek/deepseek-v4-pro")
                await second.aclose()
                await asyncio.sleep(0.05)
                try:
                    # rotation must not delete the outgoing token's free session
                    self.assertEqual(IdlePoolClient.deleted, [])
                finally:
                    await pool.aclose()

    async def test_unlimited_request_keeps_session_no_idle_delete(self):
        IdlePoolClient.deleted = []
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="deepseek/deepseek-v4-pro",
            premium_model="moonshotai/kimi-k2.7-code",
        )

        with patch("freebuff2api.codebuff.CodebuffClient", IdlePoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index", lambda now, count: 0
            ):
                pool = CodebuffAccountPool(settings)
                lease = await pool.acquire_session("deepseek/deepseek-v4-pro")
                await lease.aclose()
                await asyncio.sleep(0.2)
                try:
                    # unlimited sessions are free; they are never idle-deleted
                    self.assertEqual(IdlePoolClient.deleted, [])
                finally:
                    await pool.aclose()

    async def test_premium_request_not_deleted_before_block_window(self):
        IdlePoolClient.deleted = []
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="moonshotai/kimi-k2.7-code",
            premium_model="deepseek/deepseek-v4-pro",
            session_block_seconds=999,
            destroy_lead_seconds=0,
        )

        with patch("freebuff2api.codebuff.CodebuffClient", IdlePoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index", lambda now, count: 0
            ):
                pool = CodebuffAccountPool(settings)
                lease = await pool.acquire_session("deepseek/deepseek-v4-pro")
                self.assertTrue(pool._accounts[0].holds_premium)
                await lease.aclose()
                await asyncio.sleep(0.2)  # block window is far away
                try:
                    self.assertEqual(IdlePoolClient.deleted, [])
                finally:
                    await pool.aclose()

    async def test_premium_block_watcher_destroys_session_at_window(self):
        IdlePoolClient.deleted = []
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="moonshotai/kimi-k2.7-code",
            premium_model="deepseek/deepseek-v4-pro",
            session_block_seconds=0.2,
            destroy_lead_seconds=0.05,
        )

        with patch("freebuff2api.codebuff.CodebuffClient", IdlePoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index", lambda now, count: 0
            ):
                pool = CodebuffAccountPool(settings)
                lease = await pool.acquire_session("deepseek/deepseek-v4-pro")
                await lease.aclose()
                await asyncio.sleep(0.45)  # past the destroy window (~0.15s)
                try:
                    self.assertEqual(IdlePoolClient.deleted, ["token-a"])
                    self.assertFalse(pool._accounts[0].holds_premium)
                finally:
                    await pool.aclose()

    async def test_non_unlimited_model_waits_for_current_token_without_switching(self):
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="moonshotai/kimi-k2.7-code",
        )

        with patch("freebuff2api.codebuff.CodebuffClient", PoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index", lambda now, count: 0
            ):
                pool = CodebuffAccountPool(settings)
                first = await pool.acquire_session("deepseek/deepseek-v4-flash")
                started = asyncio.Event()

                async def acquire_second():
                    started.set()
                    return await pool.acquire_session("deepseek/deepseek-v4-flash")

                task = asyncio.create_task(acquire_second())
                await started.wait()
                await asyncio.sleep(0.05)
                try:
                    self.assertFalse(task.done())
                    self.assertEqual(first.client.settings.codebuff_token, "token-a")
                    await first.aclose()
                    second = await asyncio.wait_for(task, timeout=1)
                    self.assertEqual(
                        second.client.settings.codebuff_token, "token-a"
                    )
                    await second.aclose()
                finally:
                    await pool.aclose()

    async def test_unlimited_model_switches_to_other_token_when_busy(self):
        ParkingPoolClient.parked_models = []
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="moonshotai/kimi-k2.7-code",
        )

        with patch("freebuff2api.codebuff.CodebuffClient", ParkingPoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index", lambda now, count: 0
            ):
                pool = CodebuffAccountPool(settings)
                first = await pool.acquire_session("moonshotai/kimi-k2.7-code")
                second = await pool.acquire_session("moonshotai/kimi-k2.7-code")
                try:
                    self.assertEqual(first.client.settings.codebuff_token, "token-a")
                    self.assertEqual(second.client.settings.codebuff_token, "token-b")
                finally:
                    await second.aclose()
                    await first.aclose()
                    await pool.aclose()

    async def test_unlimited_model_whitelist_allows_each_listed_model(self):
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            unlimited_model="moonshotai/kimi-k2.7-code, minimax/minimax-m3",
        )

        with patch("freebuff2api.codebuff.CodebuffClient", ParkingPoolClient):
            with patch(
                "freebuff2api.codebuff._token_window_index", lambda now, count: 0
            ):
                pool = CodebuffAccountPool(settings)
                first = await pool.acquire_session("minimax/minimax-m3")
                second = await pool.acquire_session("minimax/minimax-m3")
                try:
                    self.assertEqual(first.client.settings.codebuff_token, "token-a")
                    self.assertEqual(second.client.settings.codebuff_token, "token-b")
                finally:
                    await second.aclose()
                    await first.aclose()
                    await pool.aclose()

    def test_unlimited_skips_premium_holding_token(self):
        with patch("freebuff2api.codebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(
                Settings(codebuff_token="token-a,token-b", local_api_key=None)
            )
        pool._active_index = 0
        pool._accounts[0].premium_started = 1.0  # token-a is mid premium block
        # unlimited must skip the premium-holding active token
        self.assertEqual(pool._next_available_index(set(), allow_switch=True), 1)
        # premium still targets the active token
        self.assertEqual(pool._next_available_index(set(), allow_switch=False), 0)

    def test_seconds_until_next_window_counts_down_to_boundary(self):
        with patch("freebuff2api.codebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(
                Settings(codebuff_token="a,b,c,d,e", local_api_key=None)
            )
        # 5 tokens -> 4.8h windows; 04:48 is the boundary into token 2.
        self.assertAlmostEqual(
            pool._seconds_until_next_window(datetime(2026, 6, 21, 4, 47, 30)),
            30.0 + 0.5,
            places=3,
        )
        # Just past midnight: next boundary is the end of token 1's window.
        self.assertAlmostEqual(
            pool._seconds_until_next_window(datetime(2026, 6, 21, 0, 0, 0)),
            17280.0 + 0.5,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
