"""
ETL CLI

Usage:
    uv run etl ashaar
    uv run etl all
    uv run etl all --dry-run
    uv run etl qafiya-apply r001
    uv run etl qafiya-apply --all
    uv run etl qafiya-reset
"""
from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="الشاعر ETL pipelines")
console = Console()


def _run_sources(source: str, dry_run: bool) -> None:
    from etl.dedup import dedup
    from etl.load import load

    streams = []

    if source in ("ashaar", "all"):
        from etl.pipelines.ashaar import extract
        console.rule("[bold cyan]ashaar")
        streams.append(extract())

    if not streams:
        console.print(f"[red]Unknown source: {source!r}[/]")
        raise typer.Exit(1)

    console.rule("[bold]Deduplicating")
    deduplicated = dedup(streams)

    console.rule("[bold]Loading → Postgres")
    counts = load(deduplicated, dry_run=dry_run)

    console.rule("[bold green]Done")
    for k, v in counts.items():
        console.print(f"  {k}: {v:,}")


@app.command("ashaar-download")
def ashaar_download(
    repo: str = typer.Option("arbml/ashaar", "--repo"),
    split: str = typer.Option("train", "--split"),
    out: str | None = typer.Option(None, "--out", help="Defaults to $ALSHAER_ROOT/dataset/ashaar"),
) -> None:
    """Download the arbml/ashaar dataset from HuggingFace and save in Arrow format."""
    import os
    from pathlib import Path
    from datasets import load_dataset
    root = Path(os.getenv("ALSHAER_ROOT", "."))
    out_path = Path(out) if out else root / "dataset" / "ashaar"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]downloading[/] {repo} ({split}) → {out_path}")
    ds = load_dataset(repo)[split]
    ds.save_to_disk(str(out_path))
    console.print(f"[green]saved[/] {len(ds):,} records")


@app.command()
def ashaar(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run the ashaar pipeline (HuggingFace arbml/ashaar → Postgres)."""
    _run_sources("ashaar", dry_run)


@app.command("all")
def all_sources(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Run every registered pipeline."""
    _run_sources("all", dry_run)


@app.command("rules-seed")
def rules_seed() -> None:
    """Upsert every rule discovered in etl.qafiya_rules into the rules table.

    Idempotent. Existing rules are updated (title/description refreshed, status
    preserved unless missing) but poems_matched / applied_at are left alone.
    """
    import os
    from datetime import datetime, timezone
    from sqlalchemy import create_engine, text as _sql
    from etl.qafiya_rules import all_rules

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
    ).replace("+asyncpg", "+psycopg")

    rules = all_rules()
    if not rules:
        console.print("[yellow]no rules found in etl.qafiya_rules[/]")
        return

    engine = create_engine(db_url, pool_pre_ping=True)
    now = datetime.now(timezone.utc)
    for r in rules:
        fn = f"etl.qafiya_rules.{r.__class__.__module__.split('.')[-1]}:match"
        # Better: point at the actual module of the match function
        fn = f"{r.match.__module__}:{r.match.__name__}"
        with engine.begin() as conn:
            existing = conn.execute(
                _sql("SELECT id, status FROM rules WHERE code = :c"),
                {"c": r.code},
            ).fetchone()
            if existing:
                conn.execute(_sql("""
                    UPDATE rules
                       SET title_ar = :title,
                           description_ar = :desc,
                           function_name = :fn
                     WHERE code = :c
                """), {"c": r.code, "title": r.title_ar, "desc": r.description_ar, "fn": fn})
                console.print(f"  [cyan]updated[/] {r.code}")
            else:
                import uuid as _uuid
                conn.execute(_sql("""
                    INSERT INTO rules (id, code, title_ar, description_ar,
                                       status, function_name, coded_at)
                    VALUES (:id, :c, :title, :desc, 'coded', :fn, :ts)
                """), {
                    "id": str(_uuid.uuid4()),
                    "c": r.code,
                    "title": r.title_ar,
                    "desc": r.description_ar,
                    "fn": fn,
                    "ts": now,
                })
                console.print(f"  [green]inserted[/] {r.code}")
    console.print(f"[green]done: seeded {len(rules)} rules[/]")


@app.command("rules-export")
def rules_export() -> None:
    """Dump the rules table to rules/*.json (one file per rule)."""
    import json as _json
    import os
    from pathlib import Path
    from sqlalchemy import create_engine, text as _sql

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
    ).replace("+asyncpg", "+psycopg")
    rules_dir = Path(os.getenv("ALSHAER_ROOT", ".")) / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(_sql("""
            SELECT id, code, title_ar, description_ar, status, function_name,
                   example_poem_id, example_qafiya_json
            FROM rules ORDER BY code NULLS LAST, created_at
        """)).mappings().all()

    for r in rows:
        data = {
            "id": str(r["id"]),
            "code": r["code"],
            "title_ar": r["title_ar"],
            "description_ar": r["description_ar"],
            "status": "coded" if r["function_name"] else "proposed",
            "function_name": r["function_name"],
            "example_poem_id": str(r["example_poem_id"]) if r["example_poem_id"] else None,
            "example_qafiya_json": r["example_qafiya_json"],
        }
        name = r["code"] if r["code"] else str(r["id"])[:8]
        (rules_dir / f"{name}.json").write_text(
            _json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        console.print(f"  wrote {name}.json")
    console.print(f"[green]done: wrote {len(rows)} rules to {rules_dir}[/]")


@app.command("rules-import")
def rules_import() -> None:
    """Import rules/*.json files into the rules table (upsert, keeps runtime state)."""
    import json as _json
    import os
    from datetime import datetime, timezone
    from pathlib import Path
    from sqlalchemy import create_engine, text as _sql

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
    ).replace("+asyncpg", "+psycopg")
    rules_dir = Path(os.getenv("ALSHAER_ROOT", ".")) / "rules"

    files = sorted(rules_dir.glob("*.json")) if rules_dir.exists() else []
    if not files:
        console.print(f"[yellow]no rules/*.json files found in {rules_dir}[/]")
        return

    engine = create_engine(db_url, pool_pre_ping=True)
    now = datetime.now(timezone.utc)
    inserted = updated = 0
    for f in files:
        data = _json.loads(f.read_text(encoding="utf-8"))
        with engine.begin() as conn:
            existing = conn.execute(
                _sql("SELECT id FROM rules WHERE id = CAST(:id AS uuid)"),
                {"id": data["id"]},
            ).fetchone()
            if existing:
                conn.execute(_sql("""
                    UPDATE rules SET
                      code = CAST(:code AS varchar),
                      title_ar = CAST(:title AS text),
                      description_ar = CAST(:desc AS text),
                      status = CASE WHEN status='applied' THEN 'applied'
                                    ELSE CAST(:status AS varchar) END,
                      function_name = CAST(:fn AS varchar),
                      example_poem_id = CAST(:pid AS uuid),
                      example_qafiya_json = CAST(:ex AS text),
                      coded_at = COALESCE(coded_at, CASE WHEN :fn IS NOT NULL THEN CAST(:now AS timestamptz) END)
                    WHERE id = CAST(:id AS uuid)
                """), {
                    "id": data["id"], "code": data.get("code"),
                    "title": data["title_ar"], "desc": data["description_ar"],
                    "status": data.get("status", "proposed"),
                    "fn": data.get("function_name"),
                    "pid": data.get("example_poem_id"),
                    "ex": data.get("example_qafiya_json"),
                    "now": now,
                })
                updated += 1
            else:
                conn.execute(_sql("""
                    INSERT INTO rules
                      (id, code, title_ar, description_ar, status, function_name,
                       example_poem_id, example_qafiya_json, coded_at)
                    VALUES
                      (CAST(:id AS uuid), CAST(:code AS varchar),
                       CAST(:title AS text), CAST(:desc AS text),
                       CAST(:status AS varchar), CAST(:fn AS varchar),
                       CAST(:pid AS uuid), CAST(:ex AS text),
                       CASE WHEN :fn IS NOT NULL THEN CAST(:now AS timestamptz) END)
                """), {
                    "id": data["id"], "code": data.get("code"),
                    "title": data["title_ar"], "desc": data["description_ar"],
                    "status": data.get("status", "proposed"),
                    "fn": data.get("function_name"),
                    "pid": data.get("example_poem_id"),
                    "ex": data.get("example_qafiya_json"),
                    "now": now,
                })
                inserted += 1
    console.print(f"[green]done: inserted {inserted}, updated {updated}[/]")


@app.command("qafiya-apply")
def qafiya_apply(
    code: str = typer.Argument(None, help="Rule code, e.g. r001. Omit when using --all."),
    all_: bool = typer.Option(False, "--all", help="Apply every registered rule in order."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    from etl.qafiya_apply import apply_rule
    from etl.qafiya_rules import all_rules

    if all_:
        rules = all_rules()
        if not rules:
            console.print("[yellow]no rules registered[/]")
            raise typer.Exit(0)
        for r in rules:
            apply_rule(r.code, dry_run=dry_run)
        return

    if not code:
        console.print("[red]must pass a rule code or --all[/]")
        raise typer.Exit(1)
    apply_rule(code, dry_run=dry_run)


@app.command("qafiya-reapply")
def qafiya_reapply(
    code: str = typer.Argument(..., help="Rule code, e.g. r001. Resets its matches then re-applies."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Reset all poems matched by <code> then apply the rule again from scratch."""
    from etl.qafiya_apply import apply_rule, reset_rule

    if not yes:
        ok = typer.confirm(f"This will reset all poems matched by {code} and re-apply. Continue?")
        if not ok:
            raise typer.Exit(0)
    if not dry_run:
        reset_rule(code)
    apply_rule(code, dry_run=dry_run)


@app.command("tashkeel-export")
def tashkeel_export(
    out: str = typer.Argument(..., help="Path to output parquet."),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    """Dump verses needing tashkeel to a parquet (id, text)."""
    from pathlib import Path
    from etl.tashkeel_bulk import export
    export(Path(out), limit=limit)


@app.command("tashkeel-import")
def tashkeel_import(
    path: str = typer.Argument(..., help="Path to result parquet (id, text_tashkeel)."),
) -> None:
    """Apply a result parquet to verses.text_tashkeel."""
    from pathlib import Path
    from etl.tashkeel_bulk import import_
    import_(Path(path))


@app.command("tashkeel-pull")
def tashkeel_pull(
    repo: str = typer.Option("mysamai/ashaar-tashkeel", "--repo"),
    path_in_repo: str = typer.Option("data/ashaar-tashkeel.parquet", "--file"),
) -> None:
    """Pull the diacritized verses parquet from HuggingFace and apply it."""
    from etl.tashkeel_bulk import pull, import_
    local = pull(repo_id=repo, path_in_repo=path_in_repo)
    import_(local)


@app.command("tashkeel-fill")
def tashkeel_fill(
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    limit: int | None = typer.Option(None, "--limit", help="Max verses to process."),
    service: str = typer.Option("http://100.76.65.1:8502", "--service"),
    retry: int = typer.Option(3, "--retry"),
) -> None:
    """Populate verses.text_tashkeel via the Fine-Tashkeel service (idempotent)."""
    from etl.tashkeel_fill import fill
    fill(batch_size=batch_size, limit=limit, service=service, retry=retry)


@app.command("qafiya-reset")
def qafiya_reset(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """NULL every poem's qafiya fields + qafiya_rule_id. Use to start over."""
    from etl.qafiya_apply import reset_all

    if not yes:
        ok = typer.confirm("This will clear ALL qafiya classification. Continue?")
        if not ok:
            raise typer.Exit(0)
    reset_all()


if __name__ == "__main__":
    app()
