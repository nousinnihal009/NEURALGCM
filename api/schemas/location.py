from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import uuid


class LocationCreate(BaseModel):
    name: str = Field(
        ..., min_length=1, max_length=255,
        description="Human-readable location name",
        example="Chennai, India")
    lat: float = Field(..., ge=-90.0, le=90.0, example=13.0827)
    lon: float = Field(..., ge=-180.0, le=180.0, example=80.2707)
    country_code: Optional[str] = Field(
        None, min_length=2, max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
        example="IN")
    timezone: Optional[str] = Field(
        None, max_length=50,
        description="IANA timezone string",
        example="Asia/Kolkata")


class LocationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    country_code: Optional[str] = Field(None, min_length=2, max_length=3)
    timezone: Optional[str] = Field(None, max_length=50)


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

    @classmethod
    def from_orm(cls, loc) -> "LocationResponse":
        return cls(
            id=str(loc.id),
            name=loc.name,
            lat=loc.lat,
            lon=loc.lon,
            country_code=loc.country_code,
            timezone=loc.timezone,
            forecast_count=loc.forecast_count or 0,
            created_at=loc.created_at.isoformat() + "Z",
            last_forecast=(
                loc.last_forecast.isoformat() + "Z"
                if loc.last_forecast else None),
        )
