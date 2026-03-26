"""
Locations Router
================
CRUD operations for saved locations.
Locations are auto-created on forecast submission and can be queried,
updated, or deleted independently.

Endpoints:
  GET    /api/v1/locations           → paginated list
  GET    /api/v1/locations/{id}      → single location with forecast history
  POST   /api/v1/locations           → create location manually
  PATCH  /api/v1/locations/{id}      → update name / timezone / country
  DELETE /api/v1/locations/{id}      → remove location record
"""

import uuid
import math
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update, desc
from loguru import logger

from api.models.database import get_db
from api.models.location import Location
from api.schemas.location import LocationCreate, LocationUpdate, LocationResponse
from api.schemas.common import PaginatedResponse
from api.dependencies import get_current_api_key

router = APIRouter(prefix="/locations", tags=["Locations"])


@router.get(
    "",
    response_model=PaginatedResponse[LocationResponse],
    summary="List all saved locations",
)
async def list_locations(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    search: Optional[str] = Query(
        default=None,
        description="Filter by name substring (case-insensitive)"),
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    query = select(Location).order_by(desc(Location.last_forecast))

    if search:
        query = query.where(
            Location.name.ilike(f"%{search}%"))

    # Count
    count_q = select(func.count()).select_from(
        query.subquery())
    total = (await db.execute(count_q)).scalar_one()

    # Page
    query = query.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(query)).scalars().all()

    items = [LocationResponse.from_orm(r) for r in rows]
    return PaginatedResponse.build(
        items=items, total=total, page=page, page_size=page_size)


@router.get(
    "/{location_id}",
    response_model=LocationResponse,
    summary="Get a single location by ID",
)
async def get_location(
    location_id: str,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    try:
        uid = uuid.UUID(location_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid location ID format")

    row = (await db.execute(
        select(Location).where(Location.id == uid)
    )).scalar_one_or_none()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Location {location_id} not found")

    return LocationResponse.from_orm(row)


@router.post(
    "",
    response_model=LocationResponse,
    status_code=201,
    summary="Create a location record",
    description="Manually register a location. Forecasts submitted via "
                "POST /forecast also auto-create location records.",
)
async def create_location(
    body: LocationCreate,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    from geoalchemy2.elements import WKTElement

    loc = Location(
        id=uuid.uuid4(),
        name=body.name,
        lat=body.lat,
        lon=body.lon,
        geom=WKTElement(f"POINT({body.lon} {body.lat})", srid=4326),
        country_code=body.country_code,
        timezone=body.timezone,
        forecast_count=0,
        created_at=datetime.utcnow(),
    )
    db.add(loc)
    await db.commit()
    await db.refresh(loc)

    logger.info(f"Location created: {loc.name} ({loc.lat},{loc.lon})")
    return LocationResponse.from_orm(loc)


@router.patch(
    "/{location_id}",
    response_model=LocationResponse,
    summary="Update location metadata",
)
async def update_location(
    location_id: str,
    body: LocationUpdate,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    try:
        uid = uuid.UUID(location_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid location ID format")

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")

    await db.execute(
        update(Location).where(Location.id == uid).values(**updates))
    await db.commit()

    row = (await db.execute(
        select(Location).where(Location.id == uid)
    )).scalar_one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Location not found")

    return LocationResponse.from_orm(row)


@router.delete(
    "/{location_id}",
    status_code=204,
    summary="Delete a location record",
)
async def delete_location(
    location_id: str,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    try:
        uid = uuid.UUID(location_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid location ID format")

    result = await db.execute(
        delete(Location).where(Location.id == uid))
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Location not found")
