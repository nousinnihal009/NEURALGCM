"""
Forecast Router
===============
POST /api/v1/forecast       → submit forecast job, get job_id immediately
GET  /api/v1/forecast/{id}  → poll for result
GET  /api/v1/forecasts      → list past forecasts with pagination
DELETE /api/v1/forecast/{id} → cancel pending job

In standalone mode, forecasts run synchronously (no Celery).
"""

import uuid
import math
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
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
from api.rate_limit import limiter

settings = get_settings()
router   = APIRouter(prefix="/forecast", tags=["Forecasts"])


def _point_geom(lon: float, lat: float):
    """Build geometry value — plain WKT string in standalone, WKTElement in prod."""
    if math.isnan(lon) or math.isnan(lat):
        raise ValueError(f"Cannot build geometry for NaN: lon={lon}, lat={lat}")

    wkt = f"POINT({lon} {lat})"

    if settings.standalone:
        return wkt

    try:
        from geoalchemy2.elements import WKTElement
        return WKTElement(wkt, srid=4326)
    except Exception:
        return wkt


def _make_run_id() -> str:
    """Return a string or UUID depending on mode."""
    return str(uuid.uuid4())


@router.post(
    "",
    response_model=ForecastJobResponse,
    summary="Submit a weather forecast request",
    status_code=202,
)
@limiter.limit("30/minute")
async def submit_forecast(
    request: Request,
    body: ForecastRequest,
    background_tasks: BackgroundTasks,
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
        job_id = _make_run_id()
        cached["job_id"]    = job_id
        cached["is_cached"] = True
        cached["status"]    = "complete"
        logger.info(
            f"Cache hit for {body.location_name} "
            f"({body.lat},{body.lon}) → returning cached")

        run = ForecastRun(
            id=job_id if settings.standalone else uuid.UUID(job_id),
            location_name=body.location_name,
            lat=body.lat, lon=body.lon,
            geom=_point_geom(body.lon, body.lat),
            forecast_days=body.days,
            init_date=body.init_date,
            mode=body.mode.value if settings.standalone else ForecastMode(body.mode.value),
            status="cached" if settings.standalone else ForecastStatus.CACHED,
            is_cached=True,
            cache_key=cache_key,
            result=cached,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            api_key_id=None,
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
    job_id = _make_run_id()
    run = ForecastRun(
        id=job_id if settings.standalone else uuid.UUID(job_id),
        location_name=body.location_name,
        lat=body.lat, lon=body.lon,
        geom=_point_geom(body.lon, body.lat),
        forecast_days=body.days,
        init_date=body.init_date,
        mode=body.mode.value if settings.standalone else ForecastMode(body.mode.value),
        status="pending" if settings.standalone else ForecastStatus.PENDING,
        cache_key=cache_key,
        created_at=datetime.utcnow(),
        api_key_id=None,
    )
    db.add(run)
    await db.commit()

    if settings.standalone:
        # ── Run synchronously in background task ──────────────
        background_tasks.add_task(
            _run_forecast_standalone,
            job_id, body.location_name, body.lat, body.lon,
            body.days, body.mode.value, body.init_date,
        )
        logger.info(f"Forecast queued (standalone) | job={job_id}")
        return ForecastJobResponse(
            job_id=job_id,
            status="pending",
            message=f"Forecast queued for {body.location_name} (standalone mode).",
            poll_url=f"{settings.api_prefix}/forecast/{job_id}",
            estimated_seconds=45,
        )
    else:
        # ── Submit Celery task ────────────────────────────────
        from api.worker.tasks import run_forecast_task
        task = run_forecast_task.apply_async(
            kwargs={
                "job_id":        job_id,
                "location_name": body.location_name,
                "lat":           body.lat,
                "lon":           body.lon,
                "days":          body.days,
                "mode":          body.mode.value,
                "init_date":     body.init_date,
            },
            task_id=job_id,
        )
        # Save celery task ID
        result = await db.execute(
            select(ForecastRun).where(ForecastRun.id == uuid.UUID(job_id)))
        run_obj = result.scalar_one_or_none()
        if run_obj:
            run_obj.celery_id = task.id
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


async def _run_forecast_standalone(
    job_id: str, location_name: str,
    lat: float, lon: float,
    days: int, mode: str, init_date: str = None,
):
    """Run forecast synchronously in standalone mode (no Celery)."""
    from api.models.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        try:
            # Mark as running
            result = await db.execute(
                select(ForecastRun).where(ForecastRun.id == job_id))
            run = result.scalar_one_or_none()
            if run:
                run.status = "running"
                run.started_at = datetime.utcnow()
                await db.commit()

            # Run NeuralGCM inference
            from neuralgcm_weather.pipeline.forecast import run_forecast_pipeline
            import numpy as np

            forecast_result = run_forecast_pipeline(
                location_name=location_name,
                lat=lat, lon=lon,
                forecast_days=days,
                init_date=f"{init_date}T00:00" if init_date else None,
                mode=mode,
                save=True,
            )

            fp      = forecast_result["forecast_point"]
            elapsed = forecast_result["elapsed_seconds"]

            def _compass(deg):
                if deg is None:
                    return None
                dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                        "S","SSW","SW","WSW","W","WNW","NW","NNW"]
                return dirs[int(round(float(deg) + 11.25) // 22) % 16]

            def _stability(lr):
                if lr is None or np.isnan(lr):
                    return None
                if lr > 9.8:
                    return "UNSTABLE"
                if lr > 7.0:
                    return "Conditionally unstable"
                return "Stable"

            daily = []
            for i, dt in enumerate(fp.dates):
                def _v(arr, i=i):
                    if arr is None:
                        return None
                    v = arr[i]
                    return None if np.isnan(v) else round(float(v), 4)

                wdir = _v(fp.wind_dir_850)
                lr   = _v(fp.lapse_rate)
                daily.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "temperature_c_850":    _v(fp.temperature_c_850),
                    "temperature_c_500":    _v(fp.temperature_c_500),
                    "rh_850":               _v(fp.rh_850),
                    "rh_500":               _v(fp.rh_500),
                    "specific_humidity_850": _v(fp.specific_humidity_850),
                    "tpw_mm":               _v(fp.tpw_mm),
                    "wind_speed_850":       _v(fp.wind_speed_850),
                    "wind_speed_500":       _v(fp.wind_speed_500),
                    "wind_speed_250":       _v(fp.wind_speed_250),
                    "wind_dir_850":         wdir,
                    "wind_dir_compass":     _compass(wdir),
                    "u_850":                _v(fp.u_850),
                    "v_850":                _v(fp.v_850),
                    "z500_m":               _v(fp.z500_m),
                    "mslp_hpa":             _v(fp.mslp_hpa),
                    "lapse_rate":           lr,
                    "stability":            _stability(lr),
                    "clwc_gkg_850":         _v(fp.clwc_gkg_850),
                    "ciwc_gkg_850":         _v(fp.ciwc_gkg_850),
                    "vorticity_850":        _v(fp.vorticity_850),
                })

            output = {
                "job_id":           job_id,
                "status":           "complete",
                "location_name":    fp.location_name,
                "lat":              fp.lat,
                "lon":              fp.lon,
                "model_lat":        fp.model_lat,
                "model_lon":        fp.model_lon,
                "init_time_utc":    forecast_result["init_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode_used":        forecast_result["mode_used"],
                "forecast_days":    fp.days,
                "elapsed_seconds":  round(elapsed, 2),
                "is_cached":        False,
                "created_at":       datetime.utcnow().isoformat() + "Z",
                "daily":            daily,
                "sanity_ok":        len(forecast_result["violations"]) == 0,
                "sanity_violations": forecast_result["violations"],
                "json_path":        forecast_result["saved_files"].get("json"),
                "csv_path":         forecast_result["saved_files"].get("csv"),
                "png_path":         forecast_result["saved_files"].get("png"),
            }

            # Update DB
            result = await db.execute(
                select(ForecastRun).where(ForecastRun.id == job_id))
            run = result.scalar_one_or_none()
            if run:
                run.status = "complete"
                run.completed_at = datetime.utcnow()
                run.elapsed_sec = elapsed
                run.result = output
                run.sanity_ok = output["sanity_ok"]
                await db.commit()

            # Cache the result
            cache_key = build_cache_key(lat, lon, days, mode, init_date)
            await set_cached_forecast(cache_key, output)

            logger.success(f"Forecast complete (standalone) | job={job_id} | elapsed={elapsed:.1f}s")

        except Exception as exc:
            logger.error(f"Forecast failed (standalone) | job={job_id} | error={exc}")
            result = await db.execute(
                select(ForecastRun).where(ForecastRun.id == job_id))
            run = result.scalar_one_or_none()
            if run:
                run.status = "failed"
                run.error_msg = str(exc)
                await db.commit()


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
    if settings.standalone:
        result = await db.execute(
            select(ForecastRun).where(ForecastRun.id == job_id))
    else:
        result = await db.execute(
            select(ForecastRun).where(
                ForecastRun.id == uuid.UUID(job_id)))
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    base = {
        "job_id":           str(run.id),
        "status":           run.status if isinstance(run.status, str) else run.status.value,
        "location_name":    run.location_name,
        "lat":              run.lat,
        "lon":              run.lon,
        "forecast_days":    run.forecast_days,
        "is_cached":        run.is_cached or False,
        "created_at":       run.created_at.isoformat() + "Z",
        "model_checkpoint": "v1/deterministic_2_8_deg.pkl",
        "paper_reference":  "Kochkov et al. 2024 (arXiv:2311.07222v3)",
    }

    status_val = run.status if isinstance(run.status, str) else run.status.value

    if status_val in ("pending", "running"):
        return ForecastResultResponse(**base, daily=[])

    if status_val == "failed":
        return ForecastResultResponse(**base, daily=[], error=run.error_msg)

    if run.result:
        r = run.result
        daily = [DailyForecast(**d) for d in r.get("daily", [])]

        if run.cache_key and not run.is_cached:
            await set_cached_forecast(run.cache_key, r)

        import os as _os

        def _static_url(raw_path: str | None) -> str | None:
            if not raw_path:
                return None
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
)
async def delete_forecast(
    job_id: str,
    request: Request,
    force: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    try:
        if settings.standalone:
            uid = job_id
        else:
            uid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    result = await db.execute(
        select(ForecastRun).where(ForecastRun.id == uid))
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status_val = run.status if isinstance(run.status, str) else run.status.value

    if status_val == "running" and not force:
        raise HTTPException(
            status_code=409,
            detail="Job is currently running. Use ?force=true to delete.")

    if not settings.standalone and status_val == "pending" and run.celery_id:
        try:
            from api.worker.celery_app import celery_app
            celery_app.control.revoke(run.celery_id, terminate=False)
        except Exception as e:
            logger.warning(f"Could not revoke Celery task: {e}")

    await db.delete(run)
    await db.commit()
    logger.info(f"Forecast job {job_id} deleted (status was {status_val})")


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
        if settings.standalone:
            query = query.where(ForecastRun.status == status)
        else:
            query = query.where(ForecastRun.status == ForecastStatus(status))

    count_query = select(func.count(ForecastRun.id))
    if status:
        if settings.standalone:
            count_query = count_query.where(ForecastRun.status == status)
        else:
            count_query = count_query.where(
                ForecastRun.status == ForecastStatus(status))
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    runs   = result.scalars().all()

    items = []
    for run in runs:
        status_val = run.status if isinstance(run.status, str) else run.status.value
        items.append(ForecastResultResponse(
            job_id=str(run.id), status=status_val,
            location_name=run.location_name, lat=run.lat, lon=run.lon,
            forecast_days=run.forecast_days, is_cached=run.is_cached or False,
            created_at=run.created_at.isoformat() + "Z",
            elapsed_seconds=run.elapsed_sec, daily=[],
            model_checkpoint="v1/deterministic_2_8_deg.pkl",
            paper_reference="Kochkov et al. 2024 (arXiv:2311.07222v3)",
        ))

    return ForecastListResponse(
        total=total, page=page, page_size=page_size, items=items)
