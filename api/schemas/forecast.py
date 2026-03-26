from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import uuid


class ForecastMode(str, Enum):
    REALTIME   = "realtime"
    HISTORICAL = "historical"


class ForecastStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"
    CACHED   = "cached"


# ── REQUEST ───────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    location_name: str = Field(
        ..., min_length=1, max_length=255,
        description="Human-readable location name",
        json_schema_extra={"example": "Chennai, India"})
    lat: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees",
        json_schema_extra={"example": 13.0827})
    lon: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees",
        json_schema_extra={"example": 80.2707})
    days: int = Field(
        default=5, ge=1, le=10,
        description="Forecast horizon in days (1-10)",
        json_schema_extra={"example": 5})
    mode: ForecastMode = Field(
        default=ForecastMode.REALTIME,
        description="realtime=ECMWF today, historical=ERA5 analog")
    init_date: Optional[str] = Field(
        default=None,
        description="Historical init date YYYY-MM-DD (historical mode only)",
        json_schema_extra={"example": "2020-06-01"})

    @field_validator("init_date")
    @classmethod
    def validate_init_date(cls, v, info):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("init_date must be YYYY-MM-DD format")
        return v


# ── VARIABLE VALUE ────────────────────────────────────────────
class DailyForecast(BaseModel):
    date: str
    temperature_c_850: Optional[float] = Field(
        None, description="Temperature at 850 hPa in Celsius (near-surface proxy)")
    temperature_c_500: Optional[float] = Field(
        None, description="Temperature at 500 hPa in Celsius (mid-troposphere)")
    rh_850: Optional[float] = Field(
        None, description="Relative humidity at 850 hPa in percent")
    rh_500: Optional[float] = Field(
        None, description="Relative humidity at 500 hPa in percent")
    specific_humidity_850: Optional[float] = Field(
        None, description="Specific humidity at 850 hPa in kg/kg")
    tpw_mm: Optional[float] = Field(
        None, description="Total Precipitable Water in mm (rain potential proxy)")
    wind_speed_850: Optional[float] = Field(
        None, description="Wind speed at 850 hPa in m/s (surface winds)")
    wind_speed_500: Optional[float] = Field(
        None, description="Wind speed at 500 hPa in m/s (steering-level winds)")
    wind_speed_250: Optional[float] = Field(
        None, description="Wind speed at 250 hPa in m/s (jet stream level)")
    wind_dir_850: Optional[float] = Field(
        None, description="Wind direction at 850 hPa in degrees (0=N 90=E 180=S 270=W)")
    wind_dir_compass: Optional[str] = Field(
        None, description="Wind direction as compass bearing (N, NNE, NE, ...)")
    u_850: Optional[float] = Field(
        None, description="U (eastward) wind component at 850 hPa in m/s")
    v_850: Optional[float] = Field(
        None, description="V (northward) wind component at 850 hPa in m/s")
    z500_m: Optional[float] = Field(
        None, description="Geopotential height at 500 hPa in metres (primary synoptic indicator)")
    mslp_hpa: Optional[float] = Field(
        None, description="Mean sea-level pressure in hPa")
    lapse_rate: Optional[float] = Field(
        None, description="Atmospheric lapse rate 850→500 hPa in deg C/km (stability indicator)")
    stability: Optional[str] = Field(
        None, description="Stability classification: Stable / Conditionally unstable / UNSTABLE")
    clwc_gkg_850: Optional[float] = Field(
        None, description="Cloud liquid water content at 850 hPa in g/kg")
    ciwc_gkg_850: Optional[float] = Field(
        None, description="Cloud ice water content at 850 hPa in g/kg")
    vorticity_850: Optional[float] = Field(
        None, description="Relative vorticity at 850 hPa in s-1 (positive=cyclonic NH)")


# ── RESPONSE ──────────────────────────────────────────────────
class ForecastJobResponse(BaseModel):
    job_id: str
    status: ForecastStatus
    message: str
    poll_url: str
    estimated_seconds: int = 45


class ForecastResultResponse(BaseModel):
    job_id: str
    status: ForecastStatus

    # Metadata
    location_name: str
    lat: float
    lon: float
    model_lat: Optional[float] = None
    model_lon: Optional[float] = None
    init_time_utc: Optional[str] = None
    mode_used: Optional[str] = None
    forecast_days: int
    elapsed_seconds: Optional[float] = None
    is_cached: bool = False
    created_at: str

    # Daily forecasts
    daily: List[DailyForecast] = []

    # Quality
    sanity_ok: Optional[bool] = None
    sanity_violations: Optional[List[str]] = None

    # File download links (populated when saved to disk)
    png_url: Optional[str] = None
    csv_url: Optional[str] = None
    json_url: Optional[str] = None

    # Error
    error: Optional[str] = None

    # Attribution
    model_checkpoint: str = "v1/deterministic_2_8_deg.pkl"
    paper_reference: str = "Kochkov et al. 2024 (arXiv:2311.07222v3)"


class ForecastListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ForecastResultResponse]
