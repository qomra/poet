"""
Async database engine and session factory.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.models import Base

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://alshaer:alshaer@localhost:5433/alshaer",
)

engine = create_async_engine(
    _DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

AsyncSessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency-injectable session — use in FastAPI or agent tool handlers."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create all tables. Used in tests and first-time setup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables() -> None:
    """Drop all tables. Tests only."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
