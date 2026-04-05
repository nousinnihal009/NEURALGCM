"""
Redis Cache Layer
=================
Caches NeuralGCM forecast results for 6 hours.
Deduplicates requests within 50km — nearby locations share a cached
forecast, reducing NeuralGCM inference load by ~80%.

In standalone dev mode, uses a simple in-memory dict instead of Redis.
"""

import json
import math
import hashlib
import time
from typing import Optional
from loguru import logger
from api.settings import get_settings

settings = get_settings()

# ── In-memory cache for standalone mode ───────────────────────
_memory_cache: dict[str, tuple[str, float]] = {}  # key → (json_str, expires_at)


def _snap_to_grid(lat: float, lon: float,
                  grid_deg: float = 1.0) -> tuple[float, float]:
    snapped_lat = round(lat / grid_deg) * grid_deg
    snapped_lon = round(lon / grid_deg) * grid_deg
    return snapped_lat, snapped_lon


def build_cache_key(lat: float, lon: float,
                    days: int, mode: str,
                    init_date: Optional[str] = None) -> str:
    snap_lat, snap_lon = _snap_to_grid(lat, lon, 0.5)
    raw = f"{snap_lat:.1f}_{snap_lon:.1f}_{days}_{mode}"
    if init_date:
        raw += f"_{init_date}"
    return f"forecast:{hashlib.md5(raw.encode()).hexdigest()}"


if settings.standalone:
    # ── In-memory implementations ─────────────────────────────

    async def get_redis():
        """No-op in standalone mode."""
        return None

    async def close_redis():
        """No-op in standalone mode."""
        pass

    async def get_cached_forecast(cache_key: str) -> Optional[dict]:
        entry = _memory_cache.get(cache_key)
        if entry:
            data_str, expires_at = entry
            if time.time() < expires_at:
                logger.info(f"Memory cache HIT: {cache_key}")
                return json.loads(data_str)
            else:
                del _memory_cache[cache_key]
        logger.debug(f"Memory cache MISS: {cache_key}")
        return None

    async def set_cached_forecast(
        cache_key: str,
        forecast_data: dict,
        ttl: int = None,
    ) -> bool:
        ttl = ttl or settings.cache_ttl_seconds
        _memory_cache[cache_key] = (
            json.dumps(forecast_data, default=str),
            time.time() + ttl
        )
        logger.info(f"Memory cache SET: {cache_key} (TTL={ttl}s)")
        return True

    async def invalidate_forecast(cache_key: str) -> bool:
        return _memory_cache.pop(cache_key, None) is not None

    async def get_cache_stats() -> dict:
        return {
            "connected": True,
            "backend": "in-memory",
            "total_keys": len(_memory_cache),
            "used_memory_mb": 0,
        }

else:
    # ── Redis implementations ─────────────────────────────────
    import redis.asyncio as aioredis

    _redis: Optional[aioredis.Redis] = None

    async def get_redis() -> aioredis.Redis:
        global _redis
        if _redis is None:
            _redis = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return _redis

    async def close_redis():
        global _redis
        if _redis:
            await _redis.aclose()
            _redis = None

    async def get_cached_forecast(cache_key: str) -> Optional[dict]:
        r = await get_redis()
        try:
            data = await r.get(cache_key)
            if data:
                logger.info(f"Cache HIT: {cache_key}")
                return json.loads(data)
            logger.debug(f"Cache MISS: {cache_key}")
            return None
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None

    async def set_cached_forecast(
        cache_key: str,
        forecast_data: dict,
        ttl: int = None,
    ) -> bool:
        r = await get_redis()
        ttl = ttl or settings.cache_ttl_seconds
        try:
            await r.setex(cache_key, ttl, json.dumps(forecast_data, default=str))
            logger.info(f"Cache SET: {cache_key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    async def invalidate_forecast(cache_key: str) -> bool:
        r = await get_redis()
        try:
            deleted = await r.delete(cache_key)
            return deleted > 0
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    async def get_cache_stats() -> dict:
        r = await get_redis()
        try:
            info = await r.info("memory")
            keys = await r.dbsize()
            return {
                "connected": True,
                "total_keys": keys,
                "used_memory_mb": round(
                    info.get("used_memory", 0) / 1024 / 1024, 2),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
