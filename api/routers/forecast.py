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
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from loguru import logger

from api.models.database import get_db
from api.models.forecast_run import ForecastRun, ForecastStatus, ForecastMode
from api.schemas.forecast import (
    ForecastRequest, ForecastJobResponse,
    ForecastResultResponse, ForecastListResponse, DailyForecast)
from api.cache.redis_client import (
    build_cache_key, get_cached_forecast, set_cached_forecast)
from api.dependencies import get_current_api_key
from api.settings import get_settings

settings = get_settings()
router   = APIRouter(prefix="/forecast", tags=["Forecasts"])


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
async def submit_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    # ── Check cache first ─────────────────────────────────────
    cache_key = build_cache_key(
        request.lat, request.lon,
        request.days, request.mode.value,
        request.init_date,
    )
    cached = await get_cached_forecast(cache_key)
    if cached:
        # Return cached result wrapped as completed job
        job_id = str(uuid.uuid4())
        cached["job_id"]    = job_id
        cached["is_cached"] = True
        cached["status"]    = "complete"
        logger.info(
            f"Cache hit for {request.location_name} "
            f"({request.lat},{request.lon}) → returning cached")

        # Store cache hit in DB
        run = ForecastRun(
            id=uuid.UUID(job_id),
            location_name=request.location_name,
            lat=request.lat, lon=request.lon,
            forecast_days=request.days,
            init_date=request.init_date,
            mode=ForecastMode(request.mode.value),
            status=ForecastStatus.CACHED,
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
        location_name=request.location_name,
        lat=request.lat, lon=request.lon,
        forecast_days=request.days,
        init_date=request.init_date,
        mode=ForecastMode(request.mode.value),
        status=ForecastStatus.PENDING,
        cache_key=cache_key,
        created_at=datetime.utcnow(),
        api_key_id=api_key.id if api_key else None,
    )
    db.add(run)
    await db.commit()

    # ── Submit Celery task ────────────────────────────────────
    from api.worker.tasks import run_forecast_task
    task = run_forecast_task.apply_async(
        kwargs={
            "job_id":        job_id,
            "location_name": request.location_name,
            "lat":           request.lat,
            "lon":           request.lon,
            "days":          request.days,
            "mode":          request.mode.value,
            "init_date":     request.init_date,
        },
        task_id=job_id,
    )

    # Save celery task ID
    run.celery_id = task.id
    await db.commit()

    logger.info(
        f"Forecast queued | job={job_id} | "
        f"loc={request.location_name} | mode={request.mode.value}")

    return ForecastJobResponse(
        job_id=job_id,
        status="pending",
        message=(f"Forecast queued for {request.location_name}. "
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
        "status":           run.status.value,
        "location_name":    run.location_name,
        "lat":              run.lat,
        "lon":              run.lon,
        "forecast_days":    run.forecast_days,
        "is_cached":        run.is_cached or False,
        "created_at":       run.created_at.isoformat() + "Z",
        "model_checkpoint": "v1/deterministic_2_8_deg.pkl",
        "paper_reference":  "Kochkov et al. 2024 (arXiv:2311.07222v3)",
    }

    if run.status in (ForecastStatus.PENDING, ForecastStatus.RUNNING):
        return ForecastResultResponse(**base, daily=[])

    if run.status == ForecastStatus.FAILED:
        return ForecastResultResponse(**base, daily=[], error=run.error_msg)

    if run.result:
        r = run.result
        daily = [DailyForecast(**d) for d in r.get("daily", [])]

        # Store to cache for subsequent identical requests
        cache_key = run.cache_key
        if cache_key and not run.is_cached:
            await set_cached_forecast(cache_key, r)

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
            png_url         = (f"/static/{r['png_path']}"
                               if r.get("png_path") else None),
            csv_url         = (f"/static/{r['csv_path']}"
                               if r.get("csv_path") else None),
        )

    return ForecastResultResponse(**base, daily=[])


@router.get(
    "s",
    response_model=ForecastListResponse,
    summary="List past forecast runs with pagination",
)
async def list_forecasts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    query = select(ForecastRun).order_by(desc(ForecastRun.created_at))

    if status:
        query = query.where(ForecastRun.status == ForecastStatus(status))

    # Count total
    count_query = select(func.count(ForecastRun.id))
    if status:
        count_query = count_query.where(
            ForecastRun.status == ForecastStatus(status))
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    runs   = result.scalars().all()

    items = []
    for run in runs:
        items.append(ForecastResultResponse(
            job_id=str(run.id), status=run.status.value,
            location_name=run.location_name, lat=run.lat, lon=run.lon,
            forecast_days=run.forecast_days, is_cached=run.is_cached or False,
            created_at=run.created_at.isoformat() + "Z",
            elapsed_seconds=run.elapsed_sec, daily=[],
            model_checkpoint="v1/deterministic_2_8_deg.pkl",
            paper_reference="Kochkov et al. 2024 (arXiv:2311.07222v3)",
        ))

    return ForecastListResponse(
        total=total, page=page, page_size=page_size, items=items)
