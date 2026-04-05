import uuid
import secrets
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Integer
from api.models.database import Base
from api.settings import get_settings

_standalone = get_settings().standalone

if _standalone:
    class APIKey(Base):
        __tablename__ = "api_keys"

        id          = Column(String(36), primary_key=True,
                             default=lambda: str(uuid.uuid4()))
        key_hash    = Column(String(255), nullable=False,
                             unique=True, index=True)
        name        = Column(String(255), nullable=False)
        is_active   = Column(Boolean, default=True)
        rate_limit  = Column(String(50), default="60/minute")
        total_calls = Column(Integer, default=0)
        created_at  = Column(DateTime, default=datetime.utcnow)
        expires_at  = Column(DateTime, nullable=True)
        last_used   = Column(DateTime, nullable=True)

        @staticmethod
        def generate() -> str:
            return f"ngcm_{secrets.token_urlsafe(32)}"
else:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID

    class APIKey(Base):
        __tablename__ = "api_keys"

        id          = Column(PG_UUID(as_uuid=True), primary_key=True,
                             default=uuid.uuid4)
        key_hash    = Column(String(255), nullable=False,
                             unique=True, index=True)
        name        = Column(String(255), nullable=False)
        is_active   = Column(Boolean, default=True)
        rate_limit  = Column(String(50), default="60/minute")
        total_calls = Column(Integer, default=0)
        created_at  = Column(DateTime, default=datetime.utcnow)
        expires_at  = Column(DateTime, nullable=True)
        last_used   = Column(DateTime, nullable=True)

        @staticmethod
        def generate() -> str:
            return f"ngcm_{secrets.token_urlsafe(32)}"
