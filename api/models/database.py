from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker)
from sqlalchemy.orm import DeclarativeBase
from api.settings import get_settings

settings = get_settings()

_engine_kwargs: dict = {}

if settings.standalone:
    # SQLite — no pooling options, need check_same_thread for async
    _engine_kwargs = {
        "echo": settings.debug,
        "connect_args": {"check_same_thread": False},
    }
else:
    # PostgreSQL — full connection pool
    _engine_kwargs = {
        "echo": settings.debug,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
    }

engine = create_async_engine(settings.database_url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
