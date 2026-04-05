from fastapi import APIRouter
from datetime import datetime
from api.cache.redis_client import get_cache_stats
from api.settings import get_settings

settings = get_settings()
router   = APIRouter(tags=["Health"])


@router.get("/health", summary="Basic health check")
async def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "mode": "standalone" if settings.standalone else "production",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/ready", summary="Readiness probe — checks all dependencies")
async def ready():
    checks = {}

    # Cache check
    cache_stats = await get_cache_stats()
    checks["cache"] = "ok" if cache_stats.get("connected") else "error"
    checks["cache_backend"] = cache_stats.get("backend", "redis")

    # DB (quick connection test)
    try:
        from api.models.database import engine
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # NeuralGCM checkpoint
    import os
    checkpoint_cached = os.path.exists(
        f"{settings.neuralgcm_cache_dir}/checkpoints/"
        f"{settings.neuralgcm_model.replace('/', '_')}")
    checks["checkpoint_cached"] = checkpoint_cached

    # Celery (only in production mode)
    if not settings.standalone:
        try:
            from api.worker.celery_app import celery_app
            inspect = celery_app.control.inspect(timeout=2)
            active  = inspect.active()
            checks["celery"] = "ok" if active is not None else "no workers"
        except Exception as e:
            checks["celery"] = f"error: {e}"
    else:
        checks["celery"] = "skipped (standalone mode)"

    all_ok = all(
        v == "ok" or isinstance(v, bool) or v.startswith("skipped")
        for v in checks.values()
        if isinstance(v, str)
    )
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/metrics/cache", summary="Cache statistics")
async def cache_metrics():
    return await get_cache_stats()
