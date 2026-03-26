"""
NeuralGCM Weather API — FastAPI Application
============================================
Serves NeuralGCM weather forecasts via REST API.
Async architecture: FastAPI + asyncpg + aioredis.
Inference runs in Celery workers (30-60s per forecast).
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
import time
import pathlib

from api.settings import get_settings
from api.models.database import engine, Base
from api.cache.redis_client import get_redis, close_redis
from api.routers import forecast, health, auth, locations

settings = get_settings()
limiter  = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Create all DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created")

    # Test Redis connection
    r = await get_redis()
    await r.ping()
    logger.info("Redis connection established")

    # Pre-warm NeuralGCM checkpoint (optional, speeds first request)
    try:
        from neuralgcm_weather.model.checkpoint import load_checkpoint
        load_checkpoint(model_name=settings.neuralgcm_model,
                        local_cache_dir=settings.neuralgcm_cache_dir)
        logger.info("NeuralGCM checkpoint pre-loaded")
    except Exception as e:
        logger.warning(f"Checkpoint pre-load skipped: {e}")

    logger.info(f"API ready at {settings.api_prefix}")
    yield

    # ── Shutdown ──────────────────────────────────────────────
    await close_redis()
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

### Variables
Temperature (850/500 hPa), Relative Humidity (850/500 hPa),
Wind Speed & Direction (850/500/250 hPa), Total Precipitable Water,
Geopotential Height Z500, Surface Pressure, Lapse Rate Stability,
Cloud Water Content, Vorticity

### Model
- Checkpoint: `v1/deterministic_2_8_deg.pkl` (default)
- Resolution: 2.8° Gaussian grid (64×128)
- Paper: [arXiv:2311.07222](https://arxiv.org/abs/2311.07222)
    """,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins    = settings.allowed_origins,
    allow_credentials= True,
    allow_methods    = ["*"],
    allow_headers    = ["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed  = (time.time() - t0) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed:.0f}ms)")
    return response


# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(
    forecast.router,
    prefix=settings.api_prefix,
    dependencies=[],
)
app.include_router(
    auth.router,
    prefix=settings.api_prefix,
)
app.include_router(
    locations.router,
    prefix=settings.api_prefix,
)

# ── Static files (serve PNG/CSV forecast outputs) ─────────────
pathlib.Path("./forecasts").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="."), name="static")


# ── Root redirect ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "name":    settings.app_name,
        "version": settings.app_version,
        "docs":    f"{settings.api_prefix}/docs",
        "health":  "/health",
        "ready":   "/ready",
    })
