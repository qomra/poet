"""
One-off bridge: apply a (title, poet_name, position, text_tashkeel) parquet to
the current verses table.

Strategy:
    1. Load poems (id, title, poet_name) and the bridge parquet into Polars.
    2. Join in memory → (poem_id, position, text_tashkeel).
    3. Write a parquet keyed by verse_id (looked up from verses via poem_id+pos).
    4. Call the existing tashkeel-import batched UPDATE by verse_id.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from rich.console import Console
from sqlalchemy import create_engine, text

console = Console()

_DB = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")


def apply_bridge(parquet_path: Path) -> dict:
    engine = create_engine(_DB, pool_pre_ping=True)

    console.print("[cyan]reading bridge + poems + verses…[/]")
    bridge = pl.read_parquet(parquet_path)
    console.print(f"  bridge: {bridge.height:,} rows")

    with engine.connect() as c:
        poems_df = pl.from_pandas(
            __import__("pandas").read_sql(
                "SELECT id::text AS poem_id, title, poet_name FROM poems",
                c.connection,
            )
        )
    console.print(f"  poems:  {poems_df.height:,}")

    joined = bridge.join(poems_df, on=["title", "poet_name"], how="inner")
    console.print(f"  after poems join: {joined.height:,}")

    with engine.connect() as c:
        verses_df = pl.from_pandas(
            __import__("pandas").read_sql(
                "SELECT id::text AS verse_id, poem_id::text AS poem_id, position FROM verses",
                c.connection,
            )
        )
    console.print(f"  verses: {verses_df.height:,}")

    final = joined.join(verses_df, on=["poem_id", "position"], how="inner") \
                  .select(pl.col("verse_id").alias("id"),
                          pl.col("text_tashkeel"))
    console.print(f"  after verses join: {final.height:,}")

    out = Path("/tmp/tashkeel_merged.parquet")
    final.write_parquet(out, compression="zstd")
    console.print(f"[green]wrote[/] {out} ({final.height:,} rows)")

    # Reuse the existing bulk import
    from etl.tashkeel_bulk import import_
    import_(out)

    return {"merged": final.height}
