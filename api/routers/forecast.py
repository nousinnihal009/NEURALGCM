"""
Forecast Router
===============
POST /api/v1/forecast       → submit forecast job, get job_id immediately
GET  /api/v1/forecast/{id}  → poll for result
GET  /api/v1/forecasts      → list past forecasts with pagination
DELETE /api/v1/forecast/{id} → cancel pending job
"""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from loguru import logger
from geoalchemy2.elements import WKTElement

from api.models.database import get_db
from api.models.forecast_run import ForecastRun, ForecastStatus, ForecastMode
from api.schemas.forecast import (
    ForecastRequest, ForecastJobResponse,
    ForecastResultResponse, ForecastListResponse, DailyForecast)
from api.cache.redis_client import (
    build_cache_key, get_cached_forecast, set_cached_forecast)
from api.dependencies import get_current_api_key
from api.settings import get_settings
from api.rate_limit import limiter


settings = get_settings()
router   = APIRouter(prefix="/forecast", tags=["Forecasts"])

import math as _math

def _point_geom(lon: float, lat: float):
    """
    Build a PostGIS-compatible geometry value for a lat/lon point.

    Returns WKTElement when connected to Postgres (production).
    Returns a plain WKT string when connected to SQLite (tests) so
    SQLAlchemy can bind it to a String column without mangling it.

    The caller (ForecastRun constructor) does not need to change —
    both return types are valid for their respective column types.
    """
    if _math.isnan(lon) or _math.isnan(lat):
        raise ValueError(
            f"Cannot build geometry for NaN coordinates: "
            f"lon={lon}, lat={lat}")

    wkt = f"POINT({lon} {lat})"

    # Detect the active dialect. The column type in the ORM model
    # is Geometry on Postgres and String on SQLite (test override).
    # Import lazily to avoid a hard dependency at module load time.
    try:
        from geoalchemy2.elements import WKTElement
        from sqlalchemy import inspect as sa_inspect
        from api.models.database import engine

        # Use sync_engine directly since it exposes the dialect synchronously
        dialect = getattr(engine, "sync_engine", engine).dialect.name
        if dialect == "postgresql":
            return WKTElement(wkt, srid=4326)
        # SQLite path — return plain WKT string stored as TEXT
        return wkt
    except Exception:
        # Fallback: return plain string if engine is not yet
        # initialised (e.g. during import-time module scanning).
        return wkt


@router.post(
    "",
    response_model=ForecastJobResponse,
    summary="Submit a weather forecast request",
    description="""
Submit a NeuralGCM weather forecast for any location on Earth.

**Returns immediately** with a `job_id`. The actual forecast runs
asynchronously in a background worker (30-60 seconds on CPU).

Poll `GET /forecast/{job_id}` until `status == "complete"`.

**Modes:**
- `realtime`: Initialises from today's ECMWF operational analysis
- `historical`: Initialises from ERA5 reanalysis (1979-2020)

**Variables returned** (when complete):
Temperature, relative humidity, wind speed/direction at 850/500/250 hPa,
total precipitable water, geopotential height Z500, surface pressure,
atmospheric stability lapse rate, cloud water content, vorticity.
    """,
    status_code=202,
)
@limiter.limit("30/minute")
async def submit_forecast(
    request: Request,
    body: ForecastRequest,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    # ── Check cache first ─────────────────────────────────────
    cache_key = build_cache_key(
        body.lat, body.lon,
        body.days, body.mode.value,
        body.init_date,
    )
    cached = await get_cached_forecast(cache_key)
    if cached:
        # Return cached result wrapped as completed job
        job_id = str(uuid.uuid4())
        cached["job_id"]    = job_id
        cached["is_cached"] = True
        cached["status"]    = "complete"
        logger.info(
            f"Cache hit for {body.location_name} "
            f"({body.lat},{body.lon}) → returning cached")

        # Store cache hit in DB
        run = ForecastRun(
            id=uuid.UUID(job_id),
            location_name=body.location_name,
            lat=body.lat, lon=body.lon,
            geom=_point_geom(body.lon, body.lat),
            forecast_days=body.days,
            init_date=body.init_date,
            mode=body.mode.value,
            status="cached",
            is_cached=True,
            cache_key=cache_key,
            result=cached,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            api_key_id=api_key.id if api_key else None,
        )
        db.add(run)
        await db.commit()

        return ForecastJobResponse(
            job_id=job_id,
            status="cached",
            message="Returning cached forecast (within 50km, last 6h)",
            poll_url=f"{settings.api_prefix}/forecast/{job_id}",
            estimated_seconds=0,
        )

    # ── Create job record in DB ───────────────────────────────
    job_id = str(uuid.uuid4())
    run = ForecastRun(
        id=uuid.UUID(job_id),
        location_name=body.location_name,
        lat=body.lat, lon=body.lon,
        geom=_point_geom(body.lon, body.lat),
        forecast_days=body.days,
        init_date=body.init_date,
        mode=body.mode.value,
        status="pending",
        cache_key=cache_key,
        created_at=datetime.utcnow(),
        api_key_id=api_key.id if api_key else None,
    )
    db.add(run)
    await db.commit()

    # ── Run forecast in background thread (bypasses Celery) ──
    import threading

    def _run_in_thread():
        """Run forecast pipeline directly in a background thread."""
        import sys, os
        # Ensure project root is on sys.path
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from api.worker.sync_runner import (
            update_job_status, run_forecast_sync)
        try:
            update_job_status(job_id, "running")
            result = run_forecast_sync(
                job_id=job_id,
                location_name=body.location_name,
                lat=body.lat,
                lon=body.lon,
                days=body.days,
                mode=body.mode.value,
                init_date=body.init_date,
            )
            update_job_status(job_id, "complete", result)
            logger.success(f"Forecast complete | job={job_id}")
        except Exception as exc:
            logger.error(f"Forecast failed | job={job_id} | {exc}")
            update_job_status(job_id, "failed", error=str(exc))

    t = threading.Thread(target=_run_in_thread, daemon=True)
    t.start()

    await db.commit()

    logger.info(
        f"Forecast queued | job={job_id} | "
        f"loc={body.location_name} | mode={body.mode.value}")

    return ForecastJobResponse(
        job_id=job_id,
        status="pending",
        message=(f"Forecast queued for {body.location_name}. "
                 f"Poll the poll_url for results."),
        poll_url=f"{settings.api_prefix}/forecast/{job_id}",
        estimated_seconds=45,
    )


@router.get(
    "/{job_id}",
    response_model=ForecastResultResponse,
    summary="Poll forecast job status and retrieve result",
)
async def get_forecast(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    # Look up in DB
    result = await db.execute(
        select(ForecastRun).where(
            ForecastRun.id == uuid.UUID(job_id)))
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Build response
    base = {
        "job_id":           str(run.id),
        "status":           run.status,
        "location_name":    run.location_name,
        "lat":              run.lat,
        "lon":              run.lon,
        "forecast_days":    run.forecast_days,
        "is_cached":        run.is_cached or False,
        "created_at":       run.created_at.isoformat() + "Z",
        "model_checkpoint": "v1/deterministic_2_8_deg.pkl",
        "paper_reference":  "Kochkov et al. 2024 (arXiv:2311.07222v3)",
    }

    if run.status in ("pending", "running"):
        return ForecastResultResponse(**base, daily=[])

    if run.status == "failed":
        return ForecastResultResponse(**base, daily=[], error=run.error_msg)

    if run.result:
        r = run.result
        daily = [DailyForecast(**d) for d in r.get("daily", [])]

        # Store to cache for subsequent identical requests
        cache_key = run.cache_key
        if cache_key and not run.is_cached:
            await set_cached_forecast(cache_key, r)

        import os as _os

        def _static_url(raw_path: str | None) -> str | None:
            if not raw_path:
                return None
            # Strip to filename only — files live in ./forecasts/
            fname = _os.path.basename(raw_path)
            return f"/static/forecasts/{fname}"

        return ForecastResultResponse(
            **base,
            model_lat       = r.get("model_lat"),
            model_lon       = r.get("model_lon"),
            init_time_utc   = r.get("init_time_utc"),
            mode_used       = r.get("mode_used"),
            elapsed_seconds = r.get("elapsed_seconds"),
            daily           = daily,
            sanity_ok       = r.get("sanity_ok"),
            sanity_violations = r.get("sanity_violations"),
            png_url         = _static_url(r.get("png_path")),
            csv_url         = _static_url(r.get("csv_path")),
            json_url        = _static_url(r.get("json_path")),
        )

    return ForecastResultResponse(**base, daily=[])


@router.delete(
    "/{job_id}",
    status_code=204,
    summary="Cancel a pending forecast job or delete a completed record",
    description="""
Cancel a **pending** or **running** job (revokes the Celery task).
Delete the DB record for **complete** or **failed** jobs.
Returns 204 No Content on success.
Returns 409 Conflict if the job is already running and cannot be safely
cancelled (Celery task has already begun NeuralGCM inference).
    """,
)
async def delete_forecast(
    job_id: str,
    request: Request,
    force: bool = Query(
        default=False,
        description="Force-delete even if status is running"),
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    try:
        uid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    result = await db.execute(
        select(ForecastRun).where(ForecastRun.id == uid))
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if run.status == "running" and not force:
        raise HTTPException(
            status_code=409,
            detail=(
                "Job is currently running. NeuralGCM inference is in "
                "progress and cannot be cleanly cancelled. Use "
                "?force=true to delete the DB record anyway (the "
                "Celery task will complete or fail independently)."))

    # Revoke Celery task if still pending
    if run.status == "pending" and run.celery_id:
        try:
            from api.worker.celery_app import celery_app
            celery_app.control.revoke(run.celery_id, terminate=False)
            logger.info(f"Celery task {run.celery_id} revoked for job {job_id}")
        except Exception as e:
            logger.warning(f"Could not revoke Celery task: {e}")

    await db.delete(run)
    await db.commit()
    logger.info(f"Forecast job {job_id} deleted (status was {run.status})")
    # 204 No Content — FastAPI returns empty body automatically


@router.get(
    "s",
    response_model=ForecastListResponse,
    summary="List past forecast runs with pagination",
)
@limiter.limit("120/minute")
async def list_forecasts(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    query = select(ForecastRun).order_by(desc(ForecastRun.created_at))

    if status:
        query = query.where(ForecastRun.status == status)

    # Count total
    count_query = select(func.count(ForecastRun.id))
    if status:
        count_query = count_query.where(
            ForecastRun.status == status)
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    runs   = result.scalars().all()

    items = []
    for run in runs:
        items.append(ForecastResultResponse(
            job_id=str(run.id), status=run.status,
            location_name=run.location_name, lat=run.lat, lon=run.lon,
            forecast_days=run.forecast_days, is_cached=run.is_cached or False,
            created_at=run.created_at.isoformat() + "Z",
            elapsed_seconds=run.elapsed_sec, daily=[],
            model_checkpoint="v1/deterministic_2_8_deg.pkl",
            paper_reference="Kochkov et al. 2024 (arXiv:2311.07222v3)",
        ))

    return ForecastListResponse(
        total=total, page=page, page_size=page_size, items=items)
