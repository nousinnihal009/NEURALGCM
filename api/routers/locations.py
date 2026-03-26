"""
Locations Router
================
Manage saved forecast locations.
"""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from api.models.database import get_db
from api.models.location import Location
from api.schemas.location import (
    LocationCreate, LocationResponse, LocationListResponse)
from api.dependencies import get_current_api_key

router = APIRouter(prefix="/locations", tags=["Locations"])


@router.get(
    "",
    response_model=LocationListResponse,
    summary="List all saved locations",
)
async def list_locations(
    search: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    query = select(Location)
    if search:
        query = query.where(Location.name.ilike(f"%{search}%"))

    count_result = await db.execute(select(func.count(Location.id)))
    total = count_result.scalar() or 0

    result = await db.execute(query)
    locs = result.scalars().all()

    items = [
        LocationResponse(
            id=str(loc.id),
            name=loc.name,
            lat=loc.lat,
            lon=loc.lon,
            country_code=loc.country_code,
            timezone=loc.timezone,
            forecast_count=loc.forecast_count or 0,
            created_at=loc.created_at.isoformat() + "Z",
            last_forecast=(loc.last_forecast.isoformat() + "Z"
                          if loc.last_forecast else None),
        )
        for loc in locs
    ]

    return LocationListResponse(total=total, items=items)


@router.post(
    "",
    response_model=LocationResponse,
    summary="Save a new location",
    status_code=201,
)
async def create_location(
    request: LocationCreate,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    loc = Location(
        id=uuid.uuid4(),
        name=request.name,
        lat=request.lat,
        lon=request.lon,
        country_code=request.country_code,
        timezone=request.timezone,
    )
    db.add(loc)
    await db.commit()

    return LocationResponse(
        id=str(loc.id),
        name=loc.name,
        lat=loc.lat,
        lon=loc.lon,
        country_code=loc.country_code,
        timezone=loc.timezone,
        forecast_count=0,
        created_at=loc.created_at.isoformat() + "Z",
    )


@router.get(
    "/{location_id}",
    response_model=LocationResponse,
    summary="Get a location by ID",
)
async def get_location(
    location_id: str,
    db: AsyncSession = Depends(get_db),
    api_key=Depends(get_current_api_key),
):
    result = await db.execute(
        select(Location).where(Location.id == uuid.UUID(location_id)))
    loc = result.scalar_one_or_none()

    if not loc:
        raise HTTPException(status_code=404, detail="Location not found")

    return LocationResponse(
        id=str(loc.id),
        name=loc.name,
        lat=loc.lat,
        lon=loc.lon,
        country_code=loc.country_code,
        timezone=loc.timezone,
        forecast_count=loc.forecast_count or 0,
        created_at=loc.created_at.isoformat() + "Z",
        last_forecast=(loc.last_forecast.isoformat() + "Z"
                      if loc.last_forecast else None),
    )
