"""
Poet enrichment utilities — used interactively with Claude.
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine, text

_DB = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")


def fetch_batch(offset: int = 0, limit: int = 50) -> list[dict]:
    """Fetch poets that have biography text but haven't been enriched yet."""
    engine = create_engine(_DB)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, name_ar, era, region, tribe, biography, diwan_url
            FROM poets
            WHERE birth_year IS NULL
              AND biography IS NOT NULL
              AND biography NOT LIKE 'لاتتوفر%%'
              AND length(biography) > 40
            ORDER BY length(biography) DESC
            LIMIT :limit OFFSET :offset
        """), {"limit": limit, "offset": offset}).fetchall()
    return [dict(r._mapping) for r in rows]


def fetch_stats() -> dict:
    """How many poets still need enrichment."""
    engine = create_engine(_DB)
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM poets")).scalar()
        enriched = conn.execute(
            text("SELECT count(*) FROM poets WHERE birth_year IS NOT NULL")
        ).scalar()
        with_bio = conn.execute(
            text("""
                SELECT count(*) FROM poets
                WHERE birth_year IS NULL
                  AND biography IS NOT NULL
                  AND length(biography) > 40
            """)
        ).scalar()
    return {"total": total, "enriched": enriched, "with_bio_remaining": with_bio}


def apply_updates(updates: list[dict]) -> int:
    """
    updates: list of {id, birth_year, death_year, era, region, tribe, biography}
    All fields nullable — COALESCE preserves existing values.
    """
    engine = create_engine(_DB)
    count = 0
    with engine.begin() as conn:
        for u in updates:
            conn.execute(text("""
                UPDATE poets SET
                    birth_year = COALESCE(:birth_year, birth_year),
                    death_year = COALESCE(:death_year, death_year),
                    era        = CASE WHEN CAST(:era AS VARCHAR) IS NOT NULL AND era = 'unknown'
                                      THEN :era ELSE era END,
                    region     = COALESCE(:region, region),
                    tribe      = COALESCE(:tribe, tribe),
                    biography  = COALESCE(:biography, biography)
                WHERE id = :id
            """), u)
            count += 1
    return count
