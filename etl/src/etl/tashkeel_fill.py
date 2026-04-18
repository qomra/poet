"""
Populate verses.text_tashkeel by calling the Fine-Tashkeel service.

Streams verses WHERE text_tashkeel IS NULL, batches them, posts to the
/diacritize endpoint, and writes the returned strings back to the row.

Idempotent — safe to re-run. Use --limit during testing.

Usage:
    uv run etl tashkeel-fill
    uv run etl tashkeel-fill --limit 1000 --batch-size 128
    uv run etl tashkeel-fill --service http://100.76.65.1:8502
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import create_engine, text

console = Console()

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")

_DEFAULT_SERVICE = os.getenv("TASHKEEL_SERVICE", "http://100.76.65.1:8502")


def _iter_batches(conn, batch_size: int, limit: int | None) -> Iterator[list[tuple[str, str]]]:
    """Yield batches of (verse_id, text) pairs, in id-keyset pagination order."""
    last_id: str | None = None
    returned = 0
    while True:
        remaining = (limit - returned) if limit else None
        fetch_n = min(batch_size, remaining) if remaining is not None else batch_size
        if fetch_n <= 0:
            return
        rows = conn.execute(
            text("""
                SELECT id, text FROM verses
                WHERE text_tashkeel IS NULL
                  AND (CAST(:last_id AS uuid) IS NULL OR id > CAST(:last_id AS uuid))
                ORDER BY id
                LIMIT :lim
            """),
            {"last_id": last_id, "lim": fetch_n},
        ).all()
        if not rows:
            return
        yield [(str(r[0]), r[1]) for r in rows]
        last_id = str(rows[-1][0])
        returned += len(rows)


def _diacritize_batch(client: httpx.Client, service: str, texts: list[str]) -> list[str]:
    r = client.post(f"{service}/diacritize", json={"texts": texts}, timeout=600)
    r.raise_for_status()
    data = r.json()
    out = data.get("diacritized") or []
    if len(out) != len(texts):
        raise RuntimeError(f"service returned {len(out)} items for a batch of {len(texts)}")
    return out


def fill(
    *,
    batch_size: int = 64,
    limit: int | None = None,
    service: str = _DEFAULT_SERVICE,
    retry: int = 3,
) -> dict:
    engine = create_engine(_DB_URL, pool_pre_ping=True)

    # Total remaining (for progress bar)
    with engine.connect() as conn:
        total_remaining = conn.execute(
            text("SELECT COUNT(*) FROM verses WHERE text_tashkeel IS NULL")
        ).scalar_one()
    target = min(total_remaining, limit) if limit else total_remaining
    if target == 0:
        console.print("[green]nothing to do — all verses already diacritized[/]")
        return {"processed": 0}

    console.print(f"[cyan]service:[/] {service}")
    console.print(f"[cyan]to process:[/] {target:,} verses (of {total_remaining:,} missing)")

    # Warm ping
    with httpx.Client(timeout=30) as client:
        try:
            health = client.get(f"{service}/health").json()
            console.print(f"[cyan]service health:[/] {health}")
        except Exception as e:
            console.print(f"[red]service unreachable: {e}[/]")
            raise

    processed = 0
    failed = 0
    started = time.time()

    with httpx.Client(timeout=600) as client, Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("tashkeel", total=target)

        # Use a fresh connection per batch write to keep transactions short
        with engine.connect() as read_conn:
            for batch in _iter_batches(read_conn, batch_size, limit):
                ids = [b[0] for b in batch]
                texts = [b[1] for b in batch]

                diacritized: list[str] | None = None
                for attempt in range(retry):
                    try:
                        diacritized = _diacritize_batch(client, service, texts)
                        break
                    except Exception as e:
                        if attempt == retry - 1:
                            console.print(f"[red]batch failed permanently: {e}[/]")
                            failed += len(batch)
                            break
                        console.print(f"[yellow]batch error (attempt {attempt+1}/{retry}): {e}[/]")
                        time.sleep(2 ** attempt)

                if diacritized is None:
                    progress.advance(task, len(batch))
                    continue

                # UPDATE in a single statement — bulk update via UNNEST
                with engine.begin() as write_conn:
                    write_conn.execute(
                        text("""
                            UPDATE verses v
                               SET text_tashkeel = data.tashkeel
                              FROM (
                                SELECT UNNEST(CAST(:ids AS uuid[])) AS id,
                                       UNNEST(CAST(:diacs AS text[])) AS tashkeel
                              ) AS data
                             WHERE v.id = data.id
                        """),
                        {"ids": ids, "diacs": diacritized},
                    )

                processed += len(batch)
                progress.advance(task, len(batch))

    elapsed = time.time() - started
    rate = processed / elapsed if elapsed > 0 else 0
    console.print(f"[green]done:[/] processed {processed:,} verses in {elapsed:.1f}s "
                  f"({rate:.1f} verses/s)")
    if failed:
        console.print(f"[red]failed:[/] {failed:,} verses — rerun to retry")
    return {"processed": processed, "failed": failed, "elapsed_s": elapsed}
