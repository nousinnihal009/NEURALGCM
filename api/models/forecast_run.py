import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime,
    JSON, Enum, ForeignKey, Text, Boolean)
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
from api.models.database import Base
import enum


class ForecastStatus(str, enum.Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"
    CACHED    = "cached"


class ForecastMode(str, enum.Enum):
    REALTIME   = "realtime"
    HISTORICAL = "historical"


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    # Identity
    id         = Column(UUID(as_uuid=True), primary_key=True,
                        default=uuid.uuid4, index=True)
    celery_id  = Column(String(255), nullable=True, index=True)
    api_key_id = Column(UUID(as_uuid=True),
                        ForeignKey("api_keys.id"), nullable=True)

    # Request parameters
    location_name = Column(String(255), nullable=False)
    lat           = Column(Float, nullable=False)
    lon           = Column(Float, nullable=False)
    geom          = Column(Geometry("POINT", srid=4326), nullable=True)
    forecast_days = Column(Integer, nullable=False, default=5)
    init_date     = Column(String(50), nullable=True)
    mode          = Column(
        String(20),
        nullable=False, default="realtime")

    # Status
    status     = Column(
        String(20),
        nullable=False, default="pending", index=True)
    error_msg  = Column(Text, nullable=True)
    is_cached  = Column(Boolean, default=False)
    cache_key  = Column(String(255), nullable=True, index=True)

    # Timing
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)
    started_at    = Column(DateTime, nullable=True)
    completed_at  = Column(DateTime, nullable=True)
    elapsed_sec   = Column(Float, nullable=True)

    # Results — store full forecast as JSON
    result        = Column(JSON, nullable=True)
    model_lat     = Column(Float, nullable=True)
    model_lon     = Column(Float, nullable=True)
    init_time_utc = Column(String(50), nullable=True)
    mode_used     = Column(String(20), nullable=True)

    # Output file paths
    json_path  = Column(String(512), nullable=True)
    csv_path   = Column(String(512), nullable=True)
    png_path   = Column(String(512), nullable=True)

    # Sanity
    sanity_ok         = Column(Boolean, nullable=True)
    sanity_violations = Column(JSON, nullable=True)
