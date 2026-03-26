"""
CI-safe pytest fixtures for API testing.
Uses aiosqlite in-memory database instead of PostgreSQL.
Mocks Redis completely to remove dependencies.
"""

import pytest
import pytest_asyncio
import asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from unittest.mock import AsyncMock, patch

from api.models.database import Base, get_db
from api.settings import Settings

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
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest_asyncio.fixture(scope="module")
async def client():
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock(return_value=True)

    with patch("api.cache.redis_client.get_redis", return_value=mock_redis):
        # Setup in-memory SQLite instead of asyncpg for testing
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        TestingSessionLocal = async_sessionmaker(
            engine, expire_on_commit=False, autoflush=False)

        from api.main import app

        async def override_get_db():
            async with TestingSessionLocal() as session:
                yield session

        app.dependency_overrides[get_db] = override_get_db

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as c:
            yield c

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
