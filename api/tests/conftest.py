"""
Pytest fixtures — CI-safe, zero external dependencies.
=======================================================
· SQLite in-memory via aiosqlite (no Postgres needed)
· Geometry columns replaced with String for SQLite compatibility
  using a COPY of metadata — Base.metadata is never mutated.
· Redis mocked via direct module-level patch of redis_client._redis
  BEFORE the FastAPI app is imported, so the global is already set
  when lifespan runs.
· All overrides are scoped to the module and torn down after.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock

from sqlalchemy import Column, String, Table, MetaData
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession)
from sqlalchemy.pool import StaticPool


# ── Patch Redis BEFORE importing the FastAPI app ──────────────
# redis_client stores the connection in a module-level _redis global.
# If the app is imported first, lifespan calls get_redis() which
# populates _redis with a live connection that ignores overrides.
# Patching the global directly before any import is the only safe fix.

import api.cache.redis_client as _rc

# _mock_redis is built fresh per-module via the fixture below.
# We still need to set _rc._redis to SOMETHING before the app
# imports (to prevent lifespan from connecting to real Redis).
# Inject a sentinel AsyncMock at import time; the per-module
# fixture will replace it with a properly wired instance.
_sentinel_redis = AsyncMock()
_sentinel_redis.ping   = AsyncMock(return_value=True)
_sentinel_redis.aclose = AsyncMock(return_value=None)
_rc._redis = _sentinel_redis


# ── NOW safe to import the app ────────────────────────────────
from api.models.database import Base, get_db   # noqa: E402
from api.cache.redis_client import get_redis    # noqa: E402


# ── Build a COPY of Base.metadata with Geometry → String ─────
# Never touch Base.metadata directly — it is a module-level singleton
# and mutations persist for the entire Python process.

def _build_test_metadata() -> MetaData:
    """
    Return a new MetaData object that mirrors Base.metadata but with
    all GeoAlchemy2 Geometry columns replaced by String(255).
    This copy is used only for in-memory SQLite table creation.
    """
    from geoalchemy2 import Geometry

    src = Base.metadata
    dst = MetaData()

    for src_table in src.tables.values():
        cols = []
        for col in src_table.columns:
            if isinstance(col.type, Geometry):
                new_col = Column(
                    col.name,
                    String(255),
                    nullable=col.nullable,
                    index=col.index,
                )
            else:
                new_col = col.copy()
            cols.append(new_col)

        Table(src_table.name, dst, *cols, extend_existing=True)

    return dst


_TEST_METADATA = _build_test_metadata()


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="module", autouse=True)
def fresh_redis_mock():
    """
    Build a fresh in-memory store and Redis mock for each test module.
    Injected into _rc._redis so all get_redis() calls return it.
    The store is scoped to this module — no cross-module leakage.
    autouse=True means every test module gets this automatically.
    """
    store: dict = {}
    mock = AsyncMock()
    mock.ping   = AsyncMock(return_value=True)
    mock.aclose = AsyncMock(return_value=None)
    mock.get    = AsyncMock(side_effect=lambda k: store.get(k))
    mock.setex  = AsyncMock(
        side_effect=lambda k, ttl, v: store.update({k: v}) or True)
    mock.delete = AsyncMock(
        side_effect=lambda k: bool(store.pop(k, None)))
    mock.dbsize = AsyncMock(side_effect=lambda: len(store))
    mock.info   = AsyncMock(return_value={"used_memory": 0})

    _rc._redis = mock      # inject before any test in this module runs
    yield mock
    store.clear()          # clean up after the module finishes
    _rc._redis = _sentinel_redis  # restore sentinel for next module


@pytest_asyncio.fixture(scope="module")
async def db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(_TEST_METADATA.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    Session = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="module")
async def client(db_engine):
    """
    Async test client with DB and Redis overrides.
    Redis is already patched at module level above.
    Only DB needs dependency_overrides.
    """
    from api.main import app

    Session = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with Session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c

    app.dependency_overrides.clear()
