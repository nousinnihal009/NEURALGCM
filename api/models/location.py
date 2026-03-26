import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
from api.models.database import Base


class Location(Base):
    __tablename__ = "locations"

    id           = Column(UUID(as_uuid=True), primary_key=True,
                          default=uuid.uuid4)
    name         = Column(String(255), nullable=False, index=True)
    lat          = Column(Float, nullable=False)
    lon          = Column(Float, nullable=False)
    geom         = Column(Geometry("POINT", srid=4326), nullable=True)
    country_code = Column(String(3), nullable=True)
    timezone     = Column(String(50), nullable=True)
    forecast_count = Column(Integer, default=0)
    created_at   = Column(DateTime, default=datetime.utcnow)
    last_forecast = Column(DateTime, nullable=True)
