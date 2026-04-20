"""
Bulk tashkeel fill via parquet round-trip (not HTTP).

Workflow:
    1. export: dump verses needing tashkeel → local parquet
    2. rsync parquet → GPU host
    3. run bulk_tashkeel.py in container on GPU host (model runs in-process,
       no HTTP overhead)
    4. rsync result parquet back
    5. import: bulk UPDATE verses.text_tashkeel from result parquet

The export + import sides live here; the orchestration shell is provided
by the `etl tashkeel-bulk` CLI command (see run.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sqlalchemy import create_engine, text

console = Console()

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")

_COPY_CHUNK = 50_000


def export(out_path: Path, limit: int | None = None) -> int:
    """Dump verses WHERE text_tashkeel IS NULL to a parquet with (id, text)."""
    engine = create_engine(_DB_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM verses WHERE text_tashkeel IS NULL")
        ).scalar_one()
    target = min(total, limit) if limit else total
    console.print(f"[cyan]exporting[/] {target:,} verses to {out_path}")

    writer: pq.ParquetWriter | None = None
    written = 0
    with engine.connect().execution_options(yield_per=_COPY_CHUNK) as conn, \
         Progress(
             SpinnerColumn(), TextColumn("{task.description}"),
             BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
             console=console,
         ) as progress:
        task = progress.add_task("export", total=target)
        q = text("""
            SELECT CAST(id AS text) AS id, text
            FROM verses
            WHERE text_tashkeel IS NULL
            ORDER BY id
        """ + (" LIMIT :lim" if limit else ""))
        result = conn.execute(q, {"lim": limit} if limit else {})
        buf_ids: list[str] = []
        buf_texts: list[str] = []
        for row in result:
            buf_ids.append(row[0])
            buf_texts.append(row[1])
            if len(buf_ids) >= _COPY_CHUNK:
                tbl = pa.table({"id": buf_ids, "text": buf_texts})
                if writer is None:
                    writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
                writer.write_table(tbl)
                written += len(buf_ids)
                progress.advance(task, len(buf_ids))
                buf_ids.clear(); buf_texts.clear()
                if limit and written >= limit:
                    break
        if buf_ids:
            tbl = pa.table({"id": buf_ids, "text": buf_texts})
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            written += len(buf_ids)
            progress.advance(task, len(buf_ids))
        if writer is not None:
            writer.close()

    console.print(f"[green]done:[/] {written:,} rows → {out_path}")
    return written


def pull(
    repo_id: str = "mysamai/ashaar-tashkeel",
    path_in_repo: str = "data/ashaar-tashkeel.parquet",
    cache_dir: str | None = None,
) -> Path:
    """Download the tashkeel parquet from HuggingFace and return its local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub not installed") from e

    console.print(f"[cyan]pulling[/] {repo_id}:{path_in_repo}")
    p = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    console.print(f"[green]cached at:[/] {p}")
    return Path(p)


def import_(in_path: Path) -> int:
    """Apply a parquet of (id, text_tashkeel) to verses rows."""
    engine = create_engine(_DB_URL, pool_pre_ping=True)

    pf = pq.ParquetFile(in_path)
    total = pf.metadata.num_rows
    console.print(f"[cyan]importing[/] {total:,} rows from {in_path}")

    written = 0
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("import", total=total)
        for batch in pf.iter_batches(batch_size=_COPY_CHUNK, columns=["id", "text_tashkeel"]):
            ids = batch.column("id").to_pylist()
            diacs = batch.column("text_tashkeel").to_pylist()
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE verses v
                           SET text_tashkeel = data.tashkeel
                          FROM (
                            SELECT UNNEST(CAST(:ids AS uuid[])) AS id,
                                   UNNEST(CAST(:diacs AS text[])) AS tashkeel
                          ) AS data
                         WHERE v.id = data.id
                    """),
                    {"ids": ids, "diacs": diacs},
                )
            written += len(ids)
            progress.advance(task, len(ids))

    console.print(f"[green]done:[/] wrote {written:,} rows to verses.text_tashkeel")
    return written
