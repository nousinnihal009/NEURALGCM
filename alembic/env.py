import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Import all ORM models so autogenerate can see them ────────
# Must be done before Base.metadata is referenced.
from api.models.database import Base          # noqa: F401
from api.models.forecast_run import ForecastRun  # noqa: F401
from api.models.location import Location         # noqa: F401
from api.models.api_key import APIKey            # noqa: F401

# ── Alembic config object ──────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """
    Prefer DATABASE_URL env var (Docker Compose sets it).
    Fall back to alembic.ini sqlalchemy.url.
    Always return a *sync* psycopg2 URL — Alembic's internal engine
    is synchronous even when using async_engine_from_config.
    """
    from api.settings import get_settings
    s = get_settings()
    return s.database_url_sync


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no DB connection required)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using the async engine (online mode)."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
