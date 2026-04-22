"""
Load deduplicated ProcessedPoem records into Postgres.

Uses SQLAlchemy core bulk insert with ON CONFLICT DO NOTHING for idempotency
— re-running the pipeline is safe.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections.abc import Iterable
from uuid import UUID, uuid4, uuid5

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from etl.normalize import (
    extract_rhyme_letter,
    normalize_era,
    normalize_language_type,
    normalize_meter,
    normalize_theme,
)
from schema import ProcessedPoem

console = Console()

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")  # load uses sync driver for bulk ops

BATCH_SIZE = 500

# Fixed namespace for deterministic UUIDs. Any machine that runs the ETL on
# the same source (e.g. arbml/ashaar) gets IDENTICAL UUIDs — so the tashkeel
# parquet on HuggingFace, keyed by verse_id, is portable.
_NAMESPACE = UUID("6b1a0000-0000-4000-8000-000000000001")


def _poet_id(source: str, poet_url: str | None, poet_name: str | None) -> str:
    key = poet_url or f"{source}:{poet_name or 'unknown'}"
    return str(uuid5(_NAMESPACE, f"poet:{key}"))


def _poem_id(source: str, source_id: str | None) -> str:
    # source_id is the poem URL for ashaar; always unique per record in upstream
    key = source_id or ""
    return str(uuid5(_NAMESPACE, f"poem:{source}:{key}"))


def _verse_id(source: str, source_id: str | None, position: int) -> str:
    key = source_id or ""
    return str(uuid5(_NAMESPACE, f"verse:{source}:{key}:{position}"))


def _clean_for_search(text_: str) -> str:
    """Strip diacritics and normalize for the search-facing `text` column."""
    s = unicodedata.normalize("NFC", text_)
    s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", s)
    s = re.sub(r"[أإآٱ]", "ا", s)
    s = s.replace("\u0640", "")  # tatweel
    return s.strip()


def load(records: Iterable[ProcessedPoem], *, dry_run: bool = False) -> dict[str, int]:
    """
    Insert records into poets, poems, verses tables.

    Returns counts: {poets_inserted, poems_inserted, verses_inserted}
    """
    engine = create_engine(_DB_URL, pool_pre_ping=True)

    # Collect all records (need two passes: poet resolution + poem insert)
    all_poems = list(records)
    console.print(f"[cyan]Loading {len(all_poems):,} poems into Postgres…[/]")

    if dry_run:
        console.print("[yellow]DRY RUN — no data written[/]")
        return {"poems": len(all_poems), "dry_run": True}

    poets_inserted = 0
    poems_inserted = 0
    verses_inserted = 0

    with engine.begin() as conn:
        # ── Step 1: upsert poets ─────────────────────────────────────────────
        poet_name_to_id: dict[str, str] = {}

        unique_poets = {
            p.poet_name.strip()
            for p in all_poems
            if p.poet_name and p.poet_name.strip()
        }
        console.print(f"  Upserting {len(unique_poets):,} poets…")

        for poet_name in unique_poets:
            # Check if exists
            row = conn.execute(
                text("SELECT id FROM poets WHERE name_ar = :n LIMIT 1"),
                {"n": poet_name},
            ).fetchone()

            if row:
                poet_name_to_id[poet_name] = str(row[0])
            else:
                # Find the first record for this poet to get description/url/era
                sample = next(
                    (p for p in all_poems if p.poet_name and p.poet_name.strip() == poet_name),
                    None,
                )
                new_id = _poet_id(
                    source=sample.source if sample else "unknown",
                    poet_url=sample.poet_url if sample else None,
                    poet_name=poet_name,
                )
                conn.execute(
                    text("""
                        INSERT INTO poets
                            (id, name_ar, era, tradition, region, biography,
                             diwan_url, signature_meters, signature_themes)
                        VALUES
                            (:id, :name, :era, 'fusha', :region, :biography,
                             :diwan_url, '{}', '{}')
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "id": new_id,
                        "name": poet_name,
                        "era": normalize_era(sample.poet_era_raw).value if sample else "unknown",
                        "region": sample.poet_location_raw if sample else None,
                        "biography": sample.poet_description if sample else None,
                        "diwan_url": sample.poet_url if sample else None,
                    },
                )
                poet_name_to_id[poet_name] = new_id
                poets_inserted += 1

        # ── Step 2: bulk insert poems + verses ───────────────────────────────
        poem_rows = []
        verse_rows = []

        for p in all_poems:
            poem_id = _poem_id(p.source, p.source_id)
            poet_id = poet_name_to_id.get(p.poet_name.strip()) if p.poet_name else None
            meter = normalize_meter(p.meter_raw)
            rhyme_letter = extract_rhyme_letter(p.rhyme_raw)
            theme = normalize_theme(p.theme_raw)
            era = normalize_era(p.poet_era_raw)

            poem_rows.append({
                "id": poem_id,
                "title": p.title,
                "poet_id": poet_id,
                "poet_name": p.poet_name,
                "tradition": p.tradition.value,
                "language_type": normalize_language_type(p.language_type_raw).value,
                "meter": meter.value,
                "meter_raw": p.meter_raw,
                "rhyme_letter": rhyme_letter,
                "theme": theme.value,
                "theme_raw": p.theme_raw,
                "era": era.value,
                "form": p.form,
                "tariq": p.tariq,
                "occasion": p.occasion,
                "verse_count": len(p.verses),
                "sources": [p.source],
                "canonical_source": p.source,
            })

            for pos, verse_text in enumerate(p.verses):
                verse_rows.append({
                    "id": _verse_id(p.source, p.source_id, pos),
                    "poem_id": poem_id,
                    "position": pos,
                    "text": _clean_for_search(verse_text),
                    "text_diacritized": verse_text if p.has_diacritics else None,
                })

        # Batch insert poems
        console.print(f"  Inserting {len(poem_rows):,} poems in batches of {BATCH_SIZE}…")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("poems", total=len(poem_rows))
            for i in range(0, len(poem_rows), BATCH_SIZE):
                batch = poem_rows[i : i + BATCH_SIZE]
                conn.execute(
                    text("""
                        INSERT INTO poems
                            (id, title, poet_id, poet_name, tradition, language_type,
                             meter, meter_raw, rhyme_letter, theme, theme_raw, era,
                             form, tariq, occasion, verse_count, sources, canonical_source)
                        VALUES
                            (:id, :title, :poet_id, :poet_name, :tradition, :language_type,
                             :meter, :meter_raw, :rhyme_letter, :theme, :theme_raw, :era,
                             :form, :tariq, :occasion, :verse_count, :sources, :canonical_source)
                        ON CONFLICT DO NOTHING
                    """),
                    batch,
                )
                poems_inserted += len(batch)
                progress.advance(task, len(batch))

        # Batch insert verses
        console.print(f"  Inserting {len(verse_rows):,} verses in batches of {BATCH_SIZE}…")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("verses", total=len(verse_rows))
            for i in range(0, len(verse_rows), BATCH_SIZE):
                batch = verse_rows[i : i + BATCH_SIZE]
                conn.execute(
                    text("""
                        INSERT INTO verses (id, poem_id, position, text, text_diacritized)
                        VALUES (:id, :poem_id, :position, :text, :text_diacritized)
                        ON CONFLICT DO NOTHING
                    """),
                    batch,
                )
                verses_inserted += len(batch)
                progress.advance(task, len(batch))

    return {
        "poets_inserted": poets_inserted,
        "poems_inserted": poems_inserted,
        "verses_inserted": verses_inserted,
    }
