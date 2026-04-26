"""
Runner for qafiya augmentation rules.

Streams classified poems (qafiya_rule_id IS NOT NULL) and applies each
augment rule. COALESCE semantics: only NULL qafiya_* columns get written.
"""
from __future__ import annotations

import os
import unicodedata

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn,
)
from sqlalchemy import create_engine, text

console = Console()

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")

_BATCH = 2000

_AUGMENTABLE_FIELDS = ["rawiy", "radf", "wasl", "harakah", "type", "taassis", "pattern"]


def _load_batch(conn, last_id: str | None, only_missing_type: bool) -> list[dict]:
    extra = "AND p.qafiya_type IS NULL" if only_missing_type else ""
    rows = conn.execute(text(f"""
        SELECT p.id, p.title, p.poet_name,
               p.qafiya_rawiy, p.qafiya_radf, p.qafiya_wasl, p.qafiya_harakah,
               p.qafiya_type, p.qafiya_taassis, p.qafiya_pattern
        FROM poems p
        WHERE p.qafiya_rule_id IS NOT NULL
          {extra}
          AND (CAST(:last_id AS uuid) IS NULL OR p.id > CAST(:last_id AS uuid))
        ORDER BY p.id
        LIMIT :lim
    """), {"last_id": last_id, "lim": _BATCH}).mappings().all()
    if not rows:
        return []
    poem_ids = [str(r["id"]) for r in rows]
    verses = conn.execute(text("""
        SELECT poem_id, position, text, text_diacritized, text_tashkeel
        FROM verses
        WHERE poem_id = ANY(CAST(:ids AS uuid[]))
        ORDER BY poem_id, position
    """), {"ids": poem_ids}).mappings().all()
    by_poem: dict[str, list] = {pid: [] for pid in poem_ids}
    for v in verses:
        by_poem[str(v["poem_id"])].append(v)

    def _nfkc(s):
        return unicodedata.normalize("NFKC", s) if s else s

    from etl.qafiya_rules import is_real_verse

    def _normalize(verses, verses_tk):
        if not verses:
            return verses, verses_tk
        odd_real = sum(1 for i, v in enumerate(verses) if i % 2 == 1 and is_real_verse(v))
        even_real = sum(1 for i, v in enumerate(verses) if i % 2 == 0 and is_real_verse(v))
        if even_real > odd_real:
            return [""] + list(verses), [None] + list(verses_tk)
        return verses, verses_tk

    out = []
    for r in rows:
        vs = by_poem[str(r["id"])]
        verses_p = [_nfkc(v["text"]) for v in vs]
        verses_t = [_nfkc(v["text_tashkeel"]) for v in vs]
        verses_p, verses_t = _normalize(verses_p, verses_t)
        out.append({
            "id": str(r["id"]),
            "title": r["title"],
            "poet_name": r["poet_name"],
            "verses": verses_p,
            "verses_tashkeel": verses_t,
            "qafiya_rawiy":      r["qafiya_rawiy"],
            "qafiya_radf":       r["qafiya_radf"],
            "qafiya_wasl":       r["qafiya_wasl"],
            "qafiya_harakah":    r["qafiya_harakah"],
            "qafiya_type":       r["qafiya_type"],
            "qafiya_taassis":    r["qafiya_taassis"],
            "qafiya_pattern":    r["qafiya_pattern"],
        })
    return out


def _apply_suggestions(conn, poem_id: str, sugg: dict, current: dict) -> int:
    """Write suggested values to NULL columns only. Returns # fields written."""
    updates: dict = {}
    for field in _AUGMENTABLE_FIELDS:
        new_val = sugg.get(field)
        if new_val is None:
            continue
        if current.get(f"qafiya_{field}") is not None:
            continue
        updates[field] = new_val
    if not updates:
        return 0
    set_sql = ", ".join(f"qafiya_{k} = :v_{k}" for k in updates)
    params = {f"v_{k}": v for k, v in updates.items()}
    params["pid"] = poem_id
    conn.execute(text(f"""
        UPDATE poems SET {set_sql}
        WHERE id = CAST(:pid AS uuid)
    """), params)
    return len(updates)


def run(code: str | None = None, only_missing_type: bool = True, dry_run: bool = False) -> dict:
    from etl.qafiya_augment import all_augments, get_augment

    if code:
        a = get_augment(code)
        if a is None:
            raise RuntimeError(f"Augment {code!r} not found")
        augments = [a]
    else:
        augments = all_augments()
    if not augments:
        console.print("[yellow]no augments registered[/]")
        return {"augments": 0}

    engine = create_engine(_DB_URL, pool_pre_ping=True)
    with engine.begin() as conn:
        total = conn.execute(text(
            "SELECT count(*) FROM poems WHERE qafiya_rule_id IS NOT NULL"
            + (" AND qafiya_type IS NULL" if only_missing_type else "")
        )).scalar_one()
    console.print(f"[cyan]augmenting[/] {total:,} poems with {len(augments)} augment(s)")
    if dry_run:
        console.print("[yellow]DRY RUN — no writes[/]")

    fields_written = {a.code: 0 for a in augments}
    poems_touched = {a.code: 0 for a in augments}
    last_id: str | None = None
    scanned = 0

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("augment", total=total)
        while True:
            with engine.begin() as conn:
                batch = _load_batch(conn, last_id, only_missing_type)
                if not batch:
                    break
                for poem in batch:
                    scanned += 1
                    for a in augments:
                        try:
                            sugg = a.fn(poem) or {}
                        except Exception as e:
                            console.print(f"[red]{a.code} raised on {poem['id']}: {e}[/]")
                            sugg = {}
                        if not sugg:
                            continue
                        if dry_run:
                            applicable = sum(
                                1 for f, v in sugg.items()
                                if v is not None and poem.get(f"qafiya_{f}") is None
                            )
                            if applicable:
                                fields_written[a.code] += applicable
                                poems_touched[a.code] += 1
                            continue
                        n = _apply_suggestions(conn, poem["id"], sugg, poem)
                        if n:
                            fields_written[a.code] += n
                            poems_touched[a.code] += 1
                            # Reflect locally so later augments in same loop see the update
                            for k, v in sugg.items():
                                if v is not None and poem.get(f"qafiya_{k}") is None:
                                    poem[f"qafiya_{k}"] = v
                    last_id = poem["id"]
                progress.advance(task, len(batch))

    console.print(f"[green]done:[/] scanned {scanned:,}")
    for a in augments:
        console.print(
            f"  {a.code}: touched {poems_touched[a.code]:,} poems, "
            f"wrote {fields_written[a.code]:,} field(s)"
        )
    return {"scanned": scanned, "fields_written": fields_written}
