"""
Apply a registered qafiya rule against the unclassified pool.

Flow:
    1. Look up the rule by code in the DB (must be status='coded' with a
       function_name pointing to a module in etl.qafiya_rules).
    2. Stream poems WHERE qafiya_rule_id IS NULL, one batch at a time.
    3. For each poem, call rule.match(poem_dict). If it returns a dict,
       update poems.qafiya_* fields + qafiya_rule_id in a single UPDATE.
    4. Report how many poems matched; update rules.poems_matched and
       rules.applied_at + rules.status='applied'.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from sqlalchemy import create_engine, text

from etl.qafiya_rules import Rule, get_rule

console = Console()

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")

_BATCH = 2000


def _load_poem_batch(conn, last_id: str | None) -> list[dict]:
    """Fetch one batch of unclassified poems, keyset-paginated by id."""
    q = """
        SELECT p.id, p.title, p.poet_name, p.meter, p.rhyme_letter
        FROM poems p
        WHERE p.qafiya_rule_id IS NULL
          AND (CAST(:last_id AS uuid) IS NULL OR p.id > CAST(:last_id AS uuid))
        ORDER BY p.id
        LIMIT :lim
    """
    rows = conn.execute(text(q), {"last_id": last_id, "lim": _BATCH}).mappings().all()
    if not rows:
        return []
    poem_ids = [str(r["id"]) for r in rows]
    verses = conn.execute(
        text("""
            SELECT poem_id, position, text, text_diacritized, text_tashkeel
            FROM verses
            WHERE poem_id = ANY(CAST(:ids AS uuid[]))
            ORDER BY poem_id, position
        """),
        {"ids": poem_ids},
    ).mappings().all()
    by_poem: dict[str, list] = {pid: [] for pid in poem_ids}
    for v in verses:
        by_poem[str(v["poem_id"])].append(v)

    from etl.qafiya_rules import is_real_verse

    def _normalize(verses: list, verses_tk: list) -> tuple[list, list]:
        """If real ajuzes are at even indices (placeholders at odd), shift the
        arrays so ajuzes end up at odd indices — keeps the universal
        `i % 2 == 1` filter working across all rules.
        """
        if not verses:
            return verses, verses_tk
        odd_real  = sum(1 for i, v in enumerate(verses) if i % 2 == 1 and is_real_verse(v))
        even_real = sum(1 for i, v in enumerate(verses) if i % 2 == 0 and is_real_verse(v))
        if even_real > odd_real:
            # Shift by 1: prepend an empty sadr; ajuzes now at odd positions.
            return ([""] + list(verses), [None] + list(verses_tk))
        return verses, verses_tk

    result = []
    for r in rows:
        vs = by_poem[str(r["id"])]
        verses    = [v["text"] for v in vs]
        verses_tk = [v["text_tashkeel"] for v in vs]
        verses, verses_tk = _normalize(verses, verses_tk)
        result.append({
            "id": str(r["id"]),
            "title": r["title"],
            "poet_name": r["poet_name"],
            "meter": r["meter"],
            "rhyme_letter": r["rhyme_letter"],
            "verses": verses,
            "verses_diacritized": [v["text_diacritized"] for v in vs],
            "verses_tashkeel": verses_tk,
        })
    return result


def _apply_to_poem(conn, poem_id: str, rule_id: str, qafiya: dict) -> None:
    """Apply rule's qafiya dict to a single poem. Missing keys stay NULL."""
    conn.execute(
        text("""
            UPDATE poems SET
                qafiya_rawiy      = COALESCE(:rawiy, qafiya_rawiy),
                qafiya_radf       = COALESCE(:radf, qafiya_radf),
                qafiya_wasl       = COALESCE(:wasl, qafiya_wasl),
                qafiya_harakah    = COALESCE(:harakah, qafiya_harakah),
                qafiya_type       = COALESCE(:type_, qafiya_type),
                qafiya_taassis    = COALESCE(:taassis, qafiya_taassis),
                qafiya_pattern    = COALESCE(:pattern, qafiya_pattern),
                qafiya_confidence = COALESCE(:confidence, qafiya_confidence),
                qafiya_rule_id    = CAST(:rule_id AS uuid)
            WHERE id = CAST(:pid AS uuid)
        """),
        {
            "pid": poem_id,
            "rule_id": rule_id,
            "rawiy": qafiya.get("rawiy"),
            "radf": qafiya.get("radf"),
            "wasl": qafiya.get("wasl"),
            "harakah": qafiya.get("harakah"),
            "type_": qafiya.get("type"),
            "taassis": qafiya.get("taassis"),
            "pattern": qafiya.get("pattern"),
            "confidence": qafiya.get("confidence"),
        },
    )


def apply_rule(code: str, *, dry_run: bool = False) -> dict:
    rule: Rule | None = get_rule(code)
    if rule is None:
        raise RuntimeError(f"Rule {code!r} not found in etl.qafiya_rules")

    engine = create_engine(_DB_URL, pool_pre_ping=True)

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM rules WHERE code = :c"),
            {"c": code},
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Rule {code!r} is not in the rules table yet")
        rule_db_id = str(row[0])

    console.rule(f"[bold cyan]Applying rule {code} — {rule.title_ar}")
    if dry_run:
        console.print("[yellow]DRY RUN — no data will be written[/]")

    scanned = 0
    matched = 0
    last_id: str | None = None

    with engine.begin() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM poems WHERE qafiya_rule_id IS NULL")
        ).scalar_one()
    console.print(f"[cyan]{total:,} poems in the unclassified pool[/]")

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"rule {code}", total=total)

        while True:
            with engine.begin() as conn:
                batch = _load_poem_batch(conn, last_id)
                if not batch:
                    break

                for poem in batch:
                    scanned += 1
                    try:
                        result = rule.match(poem)
                    except Exception as e:
                        console.print(f"[red]rule raised on poem {poem['id']}: {e}[/]")
                        result = None
                    if result:
                        matched += 1
                        if not dry_run:
                            _apply_to_poem(conn, poem["id"], rule_db_id, result)
                    last_id = poem["id"]
                progress.advance(task, len(batch))

    if not dry_run:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE rules
                    SET status = 'applied',
                        poems_matched = poems_matched + :n,
                        applied_at = :ts
                    WHERE code = :c
                """),
                {"c": code, "n": matched, "ts": datetime.now(timezone.utc)},
            )

    console.rule(f"[bold green]rule {code} done")
    console.print(f"  scanned: {scanned:,}")
    console.print(f"  matched: {matched:,}")
    return {"code": code, "scanned": scanned, "matched": matched, "dry_run": dry_run}


def reset_rule(code: str) -> int:
    """Clear all poems matched by a single rule, and reset the rule's counter.
    Returns the number of poems reset.
    """
    engine = create_engine(_DB_URL, pool_pre_ping=True)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM rules WHERE code = :c"), {"c": code}
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Rule {code!r} not found in rules table")
        rid = row[0]
        res = conn.execute(text("""
            UPDATE poems SET
                qafiya_rawiy = NULL,
                qafiya_radf = NULL,
                qafiya_wasl = NULL,
                qafiya_harakah = NULL,
                qafiya_type = NULL,
                qafiya_taassis = NULL,
                qafiya_pattern = NULL,
                qafiya_confidence = NULL,
                qafiya_rule_id = NULL
            WHERE qafiya_rule_id = :rid
        """), {"rid": rid})
        conn.execute(text("""
            UPDATE rules
               SET poems_matched = 0,
                   applied_at = NULL,
                   status = 'coded'
             WHERE id = :rid
        """), {"rid": rid})
    affected = res.rowcount or 0
    console.print(f"[yellow]reset {affected:,} poems previously matched by {code}[/]")
    return affected


def reset_all() -> dict:
    """Null all qafiya_* fields + qafiya_rule_id. Leaves the rules table alone."""
    engine = create_engine(_DB_URL, pool_pre_ping=True)
    with engine.begin() as conn:
        res = conn.execute(text("""
            UPDATE poems SET
                qafiya_rawiy = NULL,
                qafiya_radf = NULL,
                qafiya_wasl = NULL,
                qafiya_harakah = NULL,
                qafiya_type = NULL,
                qafiya_taassis = NULL,
                qafiya_pattern = NULL,
                qafiya_confidence = NULL,
                qafiya_rule_id = NULL
            WHERE qafiya_rule_id IS NOT NULL OR qafiya_rawiy IS NOT NULL
        """))
        conn.execute(text("UPDATE rules SET poems_matched = 0, applied_at = NULL, status = CASE WHEN function_name IS NULL THEN 'proposed' ELSE 'coded' END"))
    affected = res.rowcount or 0
    console.print(f"[yellow]reset {affected:,} poems[/]")
    return {"reset": affected}
