"""
NeuralGCM Weather API — FastAPI Application
============================================
Supports two modes:
  - Production:  PostgreSQL + Redis + Celery (via Docker)
  - Standalone:  SQLite + in-memory cache, no external deps
                 Set STANDALONE=true in .env
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from loguru import logger

from api.settings import get_settings
from api.models.database import engine, Base
from api.rate_limit import limiter, HAS_SLOWAPI
from api.middleware.logging import RequestLoggingMiddleware
from api.routers import forecast, health, auth, locations

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Mode: {'STANDALONE (SQLite)' if settings.standalone else 'PRODUCTION (PostgreSQL)'}")

    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created")

    # Redis — only in production mode
    if not settings.standalone:
        try:
            from api.cache.redis_client import get_redis
            r = await get_redis()
            await r.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    else:
        logger.info("Standalone mode — Redis skipped (using in-memory cache)")

    # NeuralGCM checkpoint pre-load (optional)
    try:
        from neuralgcm_weather.model.checkpoint import load_checkpoint
        load_checkpoint(
            model_name=settings.neuralgcm_model,
            local_cache_dir=settings.neuralgcm_cache_dir)
        logger.info("NeuralGCM checkpoint pre-loaded")
    except Exception as e:
        logger.warning(f"Checkpoint pre-load skipped: {e}")

    logger.info(f"API ready at {settings.api_prefix}")
    yield

    # Cleanup
    if not settings.standalone:
        try:
            from api.cache.redis_client import close_redis
            await close_redis()
        except Exception:
            pass
    await engine.dispose()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## NeuralGCM Weather Intelligence API

Production REST API serving **NeuralGCM** weather forecasts —
the state-of-the-art hybrid deep learning + physics atmospheric model
by Google Research (Kochkov et al., 2024).

### Key Features
- **Any location**: Submit any latitude/longitude worldwide
- **5-day forecasts**: 10+ atmospheric variables per day
- **Real-time**: Initialised from today's ECMWF operational analysis
- **Historical**: Initialised from ERA5 reanalysis (1979-2020)
- **Async**: Submit → get job_id → poll for results
- **Cached**: Requests within 50km share a cached forecast

### Model
- Checkpoint: `v1/deterministic_2_8_deg.pkl`
- Paper: [arXiv:2311.07222](https://arxiv.org/abs/2311.07222)
    """,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    lifespan=lifespan,
)

# ── Rate limiter (only when slowapi is available) ─────────
if HAS_SLOWAPI:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Structured request logging (modular middleware) ───────────
app.add_middleware(RequestLoggingMiddleware)

# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(forecast.router, prefix=settings.api_prefix)
app.include_router(locations.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=settings.api_prefix)

# ── Static files ──────────────────────────────────────────────
import pathlib
pathlib.Path("./forecasts").mkdir(exist_ok=True)
app.mount(
    "/static/forecasts",
    StaticFiles(directory="./forecasts"),
    name="forecasts",
)

# ── Root redirect ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "name":    settings.app_name,
        "version": settings.app_version,
        "docs":    f"{settings.api_prefix}/docs",
        "health":  "/health",
        "ready":   "/ready",
        "mode":    "standalone" if settings.standalone else "production",
    })
