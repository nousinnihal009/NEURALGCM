from fastapi import APIRouter
from datetime import datetime
from api.cache.redis_client import get_cache_stats
from api.settings import get_settings
import sqlalchemy

settings = get_settings()
router   = APIRouter(tags=["Health"])


@router.get("/health", summary="Basic health check")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/ready", summary="Readiness probe — checks all dependencies")
async def ready():
    checks = {}

    # Redis
    cache_stats = await get_cache_stats()
    checks["redis"] = "ok" if cache_stats.get("connected") else "error"

    # DB (quick connection test)
    try:
        from api.models.database import engine
        async with engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # NeuralGCM checkpoint (check local cache exists)
    import os
    checkpoint_cached = os.path.exists(
        f"{settings.neuralgcm_cache_dir}/checkpoints/"
        f"{settings.neuralgcm_model.replace('/', '_')}")
    checks["checkpoint_cached"] = checkpoint_cached

    # Celery (ping worker)
    try:
        from api.worker.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=2)
        active  = inspect.active()
        checks["celery"] = "ok" if active is not None else "no workers"
    except Exception as e:
        checks["celery"] = f"error: {e}"

    all_ok = all(v == "ok" or isinstance(v, bool)
                 for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/metrics/cache", summary="Redis cache statistics")
async def cache_metrics():
    return await get_cache_stats()
