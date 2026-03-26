import hashlib
from typing import Optional
from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime
from loguru import logger

from api.models.database import get_db
from api.models.api_key import APIKey
from api.settings import get_settings

settings  = get_settings()
api_key_header = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False,
)


async def get_current_api_key(
    key: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> Optional[APIKey]:
    """
    Validate API key from X-API-Key header.
    Returns None in development mode (no key required).
    In production, raises 401 if key is missing or invalid.
    """
    if settings.environment == "development" and not key:
        return None   # open access in dev

    if not key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header.")

    key_hash = hashlib.sha256(key.encode()).hexdigest()
    result   = await db.execute(
        select(APIKey).where(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,
        ))
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key.")

    # Update last_used and total_calls
    await db.execute(
        update(APIKey)
        .where(APIKey.id == api_key.id)
        .values(
            last_used=datetime.utcnow(),
            total_calls=APIKey.total_calls + 1,
        ))
    await db.commit()
    return api_key
