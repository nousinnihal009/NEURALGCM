import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Integer
from api.models.database import Base
from api.settings import get_settings

_standalone = get_settings().standalone

if _standalone:
    class Location(Base):
        __tablename__ = "locations"

        id           = Column(String(36), primary_key=True,
                              default=lambda: str(uuid.uuid4()))
        name         = Column(String(255), nullable=False, index=True)
        lat          = Column(Float, nullable=False)
        lon          = Column(Float, nullable=False)
        geom         = Column(String(255), nullable=True)
        country_code = Column(String(3), nullable=True)
        timezone     = Column(String(50), nullable=True)
        forecast_count = Column(Integer, default=0)
        created_at   = Column(DateTime, default=datetime.utcnow)
        last_forecast = Column(DateTime, nullable=True)
else:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
    from geoalchemy2 import Geometry

    class Location(Base):
        __tablename__ = "locations"

        id           = Column(PG_UUID(as_uuid=True), primary_key=True,
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
