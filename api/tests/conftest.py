"""
pytest fixtures for API tests.
Uses in-memory SQLite for fast testing without PostgreSQL.
"""

import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession)
from unittest.mock import AsyncMock, patch

from api.models.database import Base, get_db
from api.settings import Settings


# Override settings for testing
def get_test_settings():
    return Settings(
        environment="development",
        debug=True,
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="neuralgcm_weather_test",
        redis_host="localhost",
        redis_port=6379,
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="module")
async def client():
    """Test client with mocked Redis to avoid requiring external services."""
    # Mock Redis before importing app
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.info = AsyncMock(return_value={"used_memory": 1024})
    mock_redis.dbsize = AsyncMock(return_value=0)

    with patch("api.cache.redis_client.get_redis",
               return_value=mock_redis):
        with patch("api.cache.redis_client.get_cache_stats",
                   return_value={"connected": True, "total_keys": 0,
                                 "used_memory_mb": 0.01}):
            from api.main import app
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test"
            ) as c:
                yield c
