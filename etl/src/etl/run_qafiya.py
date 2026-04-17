"""Run qafiya classifier over all poems in the DB."""
from __future__ import annotations

import os
from uuid import UUID

from rich.console import Console
from rich.progress import track
from sqlalchemy import create_engine, text

from etl.qafiya_classifier import classify

console = Console()
_DB = os.getenv("DATABASE_URL", "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer").replace("+asyncpg", "+psycopg")
BATCH = 500


def run(min_verses: int = 6, reset: bool = False) -> None:
    engine = create_engine(_DB)

    with engine.connect() as conn:
        if reset:
            conn.execute(text("UPDATE poems SET qafiya_confidence = NULL, qafiya_rawiy = NULL"))
            conn.connection.commit()

        total = conn.execute(text("SELECT count(*) FROM poems WHERE verse_count >= :n"), {"n": min_verses}).scalar()
        console.print(f"[cyan]Processing {total:,} poems with {min_verses}+ verses[/]")

    stats = {"high": 0, "medium": 0, "low": 0, "none": 0}

    with engine.connect() as conn:
        offset = 0
        processed = 0

        while True:
            rows = conn.execute(text("""
                SELECT p.id, array_agg(v.text ORDER BY v.position) as verses
                FROM poems p
                JOIN verses v ON v.poem_id = p.id
                WHERE p.verse_count >= :min_v
                  AND p.qafiya_confidence IS NULL
                GROUP BY p.id
                LIMIT :lim OFFSET :off
            """), {"min_v": min_verses, "lim": BATCH, "off": offset}).fetchall()

            if not rows:
                break

            updates = []
            for row in rows:
                poem_id, verse_texts = row
                result = classify(verse_texts, min_verses=min_verses)
                stats[result.confidence] += 1
                updates.append({
                    "id": str(poem_id),
                    "rawiy": result.rawiy,
                    "radf": result.radf,
                    "wasl": result.wasl,
                    "harakah": result.harakah,
                    "type_": result.type_,
                    "taassis": result.taassis if result.taassis else None,
                    "pattern": result.pattern,
                    "confidence": result.confidence,
                })

            with engine.begin() as wconn:
                wconn.execute(text("""
                    UPDATE poems SET
                        qafiya_rawiy      = :rawiy,
                        qafiya_radf       = :radf,
                        qafiya_wasl       = :wasl,
                        qafiya_harakah    = :harakah,
                        qafiya_type       = :type_,
                        qafiya_taassis    = :taassis,
                        qafiya_pattern    = :pattern,
                        qafiya_confidence = :confidence
                    WHERE id = :id
                """), updates)

            processed += len(rows)
            offset += BATCH
            console.print(f"  {processed:,} / {total:,} | high={stats['high']:,} medium={stats['medium']:,} low={stats['low']:,}")

    console.print(f"\n[bold green]Done[/]")
    console.print(f"  high:   {stats['high']:,}")
    console.print(f"  medium: {stats['medium']:,}")
    console.print(f"  low:    {stats['low']:,}")
    console.print(f"  none:   {stats['none']:,}")


if __name__ == "__main__":
    run()
