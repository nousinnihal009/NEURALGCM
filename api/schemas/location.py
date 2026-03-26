from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class LocationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    country_code: Optional[str] = Field(None, max_length=3)
    timezone: Optional[str] = None


class LocationResponse(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    country_code: Optional[str] = None
    timezone: Optional[str] = None
    forecast_count: int = 0
    created_at: str
    last_forecast: Optional[str] = None


class LocationListResponse(BaseModel):
    total: int
    items: List[LocationResponse]
