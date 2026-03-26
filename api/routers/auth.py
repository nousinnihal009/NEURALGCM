import hashlib
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from api.models.database import get_db
from api.models.api_key import APIKey
from api.main import limiter

router = APIRouter(prefix="/auth", tags=["Authentication"])


class CreateKeyRequest(BaseModel):
    name: str
    rate_limit: str = "60/minute"
    expires_days: Optional[int] = None


class KeyResponse(BaseModel):
    key: str          # shown ONCE — not stored
    key_id: str
    name: str
    rate_limit: str
    created_at: str
    message: str


@router.post("/keys", response_model=KeyResponse,
             summary="Create a new API key")
@limiter.limit("10/minute")
async def create_api_key(
    request: Request,
    body: CreateKeyRequest,
    db: AsyncSession = Depends(get_db),
):
    raw_key  = APIKey.generate()
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    expires_at = None
    if body.expires_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=body.expires_days)

    api_key = APIKey(
        id=uuid.uuid4(),
        key_hash=key_hash,
        name=body.name,
        rate_limit=body.rate_limit,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()

    return KeyResponse(
        key=raw_key,
        key_id=str(api_key.id),
        name=api_key.name,
        rate_limit=api_key.rate_limit,
        created_at=api_key.created_at.isoformat() + "Z",
        message="Store this key securely — it will not be shown again.",
    )


@router.get("/keys/{key_id}/stats",
            summary="Get API key usage statistics")
async def key_stats(key_id: str, db: AsyncSession = Depends(get_db)):
    result  = await db.execute(
        select(APIKey).where(APIKey.id == uuid.UUID(key_id)))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="Key not found")
    return {
        "key_id":     str(api_key.id),
        "name":       api_key.name,
        "is_active":  api_key.is_active,
        "total_calls": api_key.total_calls,
        "created_at": api_key.created_at.isoformat() + "Z",
        "last_used":  api_key.last_used.isoformat() + "Z"
                      if api_key.last_used else None,
        "expires_at": api_key.expires_at.isoformat() + "Z"
                      if api_key.expires_at else "never",
    }
