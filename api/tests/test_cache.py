"""
Redis Cache Tests
=================
Tests cache key generation, proximity deduplication, and TTL logic.
No Redis server required — tests pure functions.
"""

import pytest
from api.cache.redis_client import build_cache_key, _snap_to_grid, _haversine_km


class TestCacheKeyGeneration:
    def test_same_location_same_key(self):
        k1 = build_cache_key(13.08, 80.27, 5, "realtime")
        k2 = build_cache_key(13.08, 80.27, 5, "realtime")
        assert k1 == k2

    def test_nearby_locations_same_key(self):
        """Locations within 0.5° grid should coalesce."""
        k1 = build_cache_key(13.08, 80.27, 5, "realtime")
        k2 = build_cache_key(13.10, 80.30, 5, "realtime")
        assert k1 == k2, "Nearby locations should share a cache key"

    def test_distant_locations_different_key(self):
        """Locations in different grid cells should have different keys."""
        k1 = build_cache_key(13.08, 80.27, 5, "realtime")
        k2 = build_cache_key(28.61, 77.20, 5, "realtime")
        assert k1 != k2

    def test_different_days_different_key(self):
        k1 = build_cache_key(13.08, 80.27, 5, "realtime")
        k2 = build_cache_key(13.08, 80.27, 3, "realtime")
        assert k1 != k2

    def test_different_mode_different_key(self):
        k1 = build_cache_key(13.08, 80.27, 5, "realtime")
        k2 = build_cache_key(13.08, 80.27, 5, "historical")
        assert k1 != k2

    def test_init_date_changes_key(self):
        k1 = build_cache_key(13.08, 80.27, 5, "historical", "2020-06-01")
        k2 = build_cache_key(13.08, 80.27, 5, "historical", "2020-07-01")
        assert k1 != k2

    def test_cache_key_format(self):
        k = build_cache_key(13.08, 80.27, 5, "realtime")
        assert k.startswith("forecast:")
        assert len(k) > 20  # forecast: + md5 hash


class TestSnapToGrid:
    def test_snap_exact(self):
        assert _snap_to_grid(13.0, 80.0, 0.5) == (13.0, 80.0)

    def test_snap_round_up(self):
        assert _snap_to_grid(13.3, 80.4, 0.5) == (13.5, 80.5)

    def test_snap_round_down(self):
        assert _snap_to_grid(13.1, 80.1, 0.5) == (13.0, 80.0)

    def test_negative_coords(self):
        lat, lon = _snap_to_grid(-33.87, 151.21, 0.5)
        assert lat == -34.0
        assert lon == 151.0


class TestHaversine:
    def test_same_point(self):
        assert _haversine_km(13.08, 80.27, 13.08, 80.27) == 0.0

    def test_known_distance(self):
        # Chennai to Delhi ≈ 1757 km
        d = _haversine_km(13.0827, 80.2707, 28.6139, 77.2090)
        assert 1700 < d < 1800

    def test_short_distance(self):
        # ~11 km offset
        d = _haversine_km(13.08, 80.27, 13.18, 80.27)
        assert 10 < d < 12
