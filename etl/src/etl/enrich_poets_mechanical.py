"""
Mechanical poet enrichment — no LLM.

Extracts from ashaar dataset fields and biography text:
  - era        → from poet_era field (after fixing normalize_era)
  - region     → from poet_location field (already a clean country name)
  - birth_year → regex on biography text
  - death_year → regex on biography text
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.progress import track
from sqlalchemy import create_engine, text

from etl.normalize import normalize_era

console = Console()

_DB = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")

_ROOT = Path(os.getenv("ALSHAER_ROOT", Path(__file__).parents[5]))
_ARROW_DIR = _ROOT / "dataset" / "ashaar"


# ---------------------------------------------------------------------------
# Regex patterns for year extraction from biography text
# ---------------------------------------------------------------------------

def extract_years(bio: str) -> tuple[int | None, int | None]:
    """
    Extract birth and death year (Gregorian) from biography text.
    Handles patterns like:
      (303هـ-354هـ/915م-965م)   → birth=915, death=965
      (1342 - 1419 هـ / 1923 - 1998 م)  → birth=1923, death=1998
      ولد ... عام 1955          → birth=1955
      (26 يوليو 1903 – 27 يوليو 1997)   → birth=1903, death=1997
      توفي سنة 1250هـ           → death (converted)
    """
    birth: int | None = None
    death: int | None = None

    # Pattern 1: explicit CE pair  (NNNهـ-NNNهـ/NNNNم-NNNNم)
    m = re.search(r"\d+هـ[-–]\d+هـ/(\d{3,4})م[-–](\d{3,4})م", bio)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Pattern 2: gregorian range with م marker  / NNNNم - NNNNم
    m = re.search(r"/\s*(\d{3,4})\s*م\s*[-–]\s*(\d{3,4})\s*م", bio)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Pattern 3: two 4-digit years with – or - between them (gregorian range)
    m = re.search(r"\((\d{4})\s*[-–]\s*(\d{4})\)", bio)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if 600 <= y1 <= 2024 and 600 <= y2 <= 2024 and y2 > y1:
            return y1, y2

    # Pattern 4: ولد ... عام/سنة NNNN
    m = re.search(r"ولد[^.،\n]{0,30}(?:عام|سنة)\s+(\d{4})", bio)
    if m:
        birth = int(m.group(1))

    # Pattern 5: مولده / ولادته ... NNNN
    m = re.search(r"(?:مولده|ولادته)[^.،\n]{0,30}(\d{4})", bio)
    if m and not birth:
        birth = int(m.group(1))

    # Pattern 6: توفي/وفاته ... NNNN م
    m = re.search(r"(?:توفي|وفاته|مات)[^.،\n]{0,30}(\d{4})\s*م", bio)
    if m:
        death = int(m.group(1))

    # Pattern 7: plain 4-digit year with م after it anywhere
    if not birth:
        years_ce = [int(x) for x in re.findall(r"\b(\d{4})\s*م", bio)
                    if 600 <= int(x) <= 2024]
        if len(years_ce) >= 2:
            birth, death = min(years_ce), max(years_ce)
        elif len(years_ce) == 1:
            birth = years_ce[0]

    return birth, death


# ---------------------------------------------------------------------------
# Main enrichment run
# ---------------------------------------------------------------------------

def run() -> None:
    from datasets import load_from_disk

    console.print(f"[cyan]Loading {_ARROW_DIR}…[/]")
    ds = load_from_disk(str(_ARROW_DIR))

    # Build per-poet aggregated data
    # Key: poet_name → {era_raw, location, biography, birth_year, death_year}
    poet_data: dict[str, dict] = {}

    for rec in track(ds, description="Scanning poets…"):
        name = (rec.get("poet name") or "").strip()
        if not name:
            continue
        if name not in poet_data:
            poet_data[name] = {
                "era_raw": rec.get("poet era") or "",
                "location": rec.get("poet location") or "",
                "biography": rec.get("poet description") or "",
            }

    console.print(f"[cyan]{len(poet_data):,} unique poets found in dataset[/]")

    # Enrich each poet mechanically
    engine = create_engine(_DB)
    updated = skipped = 0

    with engine.begin() as conn:
        for name, data in track(poet_data.items(), description="Enriching…"):
            era = normalize_era(data["era_raw"])
            region = data["location"].strip() or None
            bio = data["biography"]
            birth_year, death_year = extract_years(bio) if bio else (None, None)

            # Only update if we have something to contribute
            if era.value == "unknown" and not region and not birth_year and not death_year:
                skipped += 1
                continue

            conn.execute(text("""
                UPDATE poets SET
                    era        = CASE WHEN :era != 'unknown' THEN :era ELSE era END,
                    region     = COALESCE(:region, region),
                    birth_year = COALESCE(:birth_year, birth_year),
                    death_year = COALESCE(:death_year, death_year)
                WHERE name_ar = :name
            """), {
                "name": name,
                "era": era.value,
                "region": region,
                "birth_year": birth_year,
                "death_year": death_year,
            })
            updated += 1

    console.print(f"\n[green]✓[/] Updated {updated:,} poets | Skipped {skipped:,} (no data)")

    # Stats
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                count(*) as total,
                count(birth_year) as with_birth,
                count(death_year) as with_death,
                count(region) as with_region,
                sum(CASE WHEN era != 'unknown' THEN 1 ELSE 0 END) as with_era
            FROM poets
        """)).fetchone()
        console.print(f"\n[bold]Poets table after enrichment:[/]")
        console.print(f"  total:       {stats[0]:,}")
        console.print(f"  with era:    {stats[4]:,}")
        console.print(f"  with region: {stats[3]:,}")
        console.print(f"  with birth:  {stats[1]:,}")
        console.print(f"  with death:  {stats[2]:,}")
