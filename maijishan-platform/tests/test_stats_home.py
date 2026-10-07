import unittest
from unittest.mock import patch

from app.api.routes import stats


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return _FakeQuery(self._rows)


class _FakeSettings:
    homepage_stats_cache_ttl_seconds = 60


class StatsRouteTests(unittest.TestCase):
    def setUp(self):
        stats._cache_data = None
        stats._cache_expires_at = 0.0

    def test_compute_home_stats_payload_shape_and_ordering(self):
        rows = [
            ("石窟,雕塑", "张三", "2012-04-25", "中国敦煌研究院", "知网", "期刊", "https://example.cn/a"),
            ("石窟,壁画", "李四", "2011", "University of Oxford", "Scopus", "journal", "https://example.uk/b"),
            ("壁画", "张三", "2012", "北京大学", "知网", "期刊", "https://example.cn/c"),
        ]
        payload = stats._compute_home_stats_payload(_FakeDB(rows), top_n=5)

        self.assertIn("keywords", payload)
        self.assertIn("year_series", payload)
        self.assertIn("top_authors", payload)
        self.assertIn("geo_hotspots", payload)
        self.assertIn("partial_data", payload)

        self.assertEqual(payload["keywords"][0]["keyword"], "壁画")
        self.assertEqual(payload["keywords"][0]["count"], 2)
        self.assertEqual(payload["top_authors"][0]["author"], "张三")
        self.assertEqual(payload["top_authors"][0]["count"], 2)

        years = [item["year"] for item in payload["year_series"]]
        self.assertEqual(years, sorted(years))

        self.assertTrue(any(item["region"] == "中国" for item in payload["geo_hotspots"]))

    def test_cache_hit_bypasses_recompute(self):
        calls = {"count": 0}

        def _fake_compute(_db, top_n=10):
            calls["count"] += 1
            return {
                "keywords": [],
                "year_series": [],
                "top_authors": [],
                "geo_hotspots": [],
                "partial_data": {
                    "keywords": {"available": False, "reason": "no_usable_data"},
                    "year_series": {"available": False, "reason": "no_usable_data"},
                    "top_authors": {"available": False, "reason": "no_usable_data"},
                    "geo_hotspots": {"available": False, "reason": "no_usable_data"},
                },
                "updated_at": "2026-01-01T00:00:00Z",
            }

        with patch("app.api.routes.stats.get_settings", return_value=_FakeSettings()), patch(
            "app.api.routes.stats._compute_home_stats_payload", side_effect=_fake_compute
        ):
            data1, hit1 = stats._get_home_stats_cached(_FakeDB([]))
            data2, hit2 = stats._get_home_stats_cached(_FakeDB([]))

        self.assertFalse(hit1)
        self.assertTrue(hit2)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(data1["updated_at"], data2["updated_at"])


if __name__ == "__main__":
    unittest.main()
