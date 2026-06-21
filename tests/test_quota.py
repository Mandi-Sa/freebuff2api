import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from freebuff2api.quota import QuotaStore


def _iso(delta_hours: float) -> str:
    moment = datetime.now(timezone.utc) + timedelta(hours=delta_hours)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class QuotaStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "sub", "quota.json")

    def test_record_persists_and_reloads(self) -> None:
        store = QuotaStore(self.path)
        store.record(
            token_index=2,
            token_hint="***2acb",
            used=2.3,
            limit=5,
            reset_at=_iso(5),
        )
        self.assertTrue(os.path.exists(self.path))
        with open(self.path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["tokens"][0]["used"], 2.3)

        reloaded = QuotaStore(self.path)
        snap = reloaded.snapshot()
        self.assertEqual(len(snap), 1)
        self.assertEqual(snap[0]["token_index"], 2)
        self.assertEqual(snap[0]["used"], 2.3)

    def test_snapshot_zeroes_used_after_reset(self) -> None:
        store = QuotaStore(self.path)
        store.record(token_index=1, token_hint="***1", used=4.0, limit=5, reset_at=_iso(-1))
        store.record(token_index=2, token_hint="***2", used=4.0, limit=5, reset_at=_iso(3))
        snap = {row["token_index"]: row for row in store.snapshot()}
        self.assertEqual(snap[1]["effective_used"], 0.0)  # reset already passed
        self.assertEqual(snap[2]["effective_used"], 4.0)  # not yet reset


if __name__ == "__main__":
    unittest.main()
