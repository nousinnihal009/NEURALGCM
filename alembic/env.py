import asyncio
from logging.config import fileConfig

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from api.models.database import Base          # noqa: F401
from api.models.forecast_run import ForecastRun  # noqa: F401
from api.models.location import Location         # noqa: F401
from api.models.api_key import APIKey            # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _async_url() -> str:
    """
    Returns the asyncpg database URL from Settings.
    Always uses postgresql+asyncpg:// driver.
    """
    from api.settings import get_settings
    return get_settings().database_url          # asyncpg URL


def _sync_url() -> str:
    """
    Returns a synchronous psycopg2 URL derived from the async URL.
    Used only for offline mode (no live DB connection).
    Replaces +asyncpg with +psycopg2 in the driver string.
    """
    return _async_url().replace(
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
        1,
    )


def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection.
    Alembic emits SQL to stdout / a file instead of executing it.
    Uses psycopg2 URL — requires psycopg2-binary in environment.
    """
    context.configure(
        url=_sync_url(),
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
    """
    Run migrations against a live DB using the async engine.
    asyncpg driver — no psycopg2 dependency at runtime.
    """
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _async_url()

    connectable = async_engine_from_config(
        cfg,
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
