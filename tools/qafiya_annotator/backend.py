"""FastAPI backend for the Qafiya Annotator (rule-proposal workflow)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text

DB = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alshaer:alshaer@localhost:5433/alshaer",
).replace("+asyncpg", "+psycopg")

engine = create_engine(DB, pool_pre_ping=True, pool_size=5)
app = FastAPI()

STATIC = Path(__file__).parent / "static"


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/poem")
def get_poem():
    """Random unclassified poem (qafiya_rule_id IS NULL) with at least 6 verses."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT p.id, p.title, p.poet_name, p.meter, p.rhyme_letter,
                   p.verse_count, p.era,
                   array_agg(v.text ORDER BY v.position) as verses
            FROM poems p JOIN verses v ON v.poem_id = p.id
            WHERE p.qafiya_rule_id IS NULL
              AND p.verse_count >= 6
            GROUP BY p.id
            ORDER BY random()
            LIMIT 1
        """)).fetchone()

    if not row:
        return JSONResponse({"error": "no_poems"}, status_code=404)

    d = dict(row._mapping)
    d["id"] = str(d["id"])
    return d


@app.get("/api/poem/{pid}")
def get_poem_by_id(pid: str):
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT p.id, p.title, p.poet_name, p.meter, p.rhyme_letter,
                   p.verse_count, p.era,
                   array_agg(v.text ORDER BY v.position) as verses
            FROM poems p JOIN verses v ON v.poem_id = p.id
            WHERE p.id = CAST(:id AS uuid)
            GROUP BY p.id
        """), {"id": pid}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="poem not found")
    d = dict(row._mapping)
    d["id"] = str(d["id"])
    return d


@app.get("/api/stats")
def get_stats():
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT
              count(*) FILTER (WHERE verse_count >= 6) AS total,
              count(*) FILTER (WHERE verse_count >= 6 AND qafiya_rule_id IS NOT NULL) AS classified,
              count(*) FILTER (WHERE verse_count >= 6 AND qafiya_rule_id IS NULL) AS unclassified
            FROM poems
        """)).fetchone()
        rules_total, rules_proposed, rules_coded, rules_applied = conn.execute(text("""
            SELECT
              count(*),
              count(*) FILTER (WHERE status = 'proposed'),
              count(*) FILTER (WHERE status = 'coded'),
              count(*) FILTER (WHERE status = 'applied')
            FROM rules
        """)).fetchone()
    return {
        **dict(r._mapping),
        "rules_total": rules_total,
        "rules_proposed": rules_proposed,
        "rules_coded": rules_coded,
        "rules_applied": rules_applied,
    }


@app.get("/api/rules")
def list_rules():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, code, title_ar, description_ar, status, function_name,
                   poems_matched, example_poem_id, created_at, coded_at, applied_at
            FROM rules
            ORDER BY created_at DESC
            LIMIT 200
        """)).fetchall()
    return [
        dict(r._mapping) | {
            "id": str(r._mapping["id"]),
            "example_poem_id": str(r._mapping["example_poem_id"])
            if r._mapping["example_poem_id"] else None,
        }
        for r in rows
    ]


class RuleIn(BaseModel):
    title_ar: str
    description_ar: str
    example_poem_id: str | None = None
    # Expected qafiya fields for the example poem (any subset is fine)
    example_rawiy: str | None = None
    example_radf: str | None = None
    example_wasl: str | None = None
    example_harakah: str | None = None
    example_type: str | None = None


def _example_json(body: RuleIn) -> str | None:
    ex = {
        k: v for k, v in {
            "rawiy": body.example_rawiy,
            "radf": body.example_radf,
            "wasl": body.example_wasl,
            "harakah": body.example_harakah,
            "type": body.example_type,
        }.items() if v
    }
    return json.dumps(ex, ensure_ascii=False) if ex else None


@app.post("/api/rules")
def add_rule(body: RuleIn):
    title = (body.title_ar or "").strip()
    desc = (body.description_ar or "").strip()
    if not title or not desc:
        raise HTTPException(status_code=400, detail="title_ar and description_ar are required")

    rid = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO rules
              (id, title_ar, description_ar, status, example_poem_id, example_qafiya_json)
            VALUES
              (:id, :title, :desc, 'proposed', :pid, :ex)
        """), {
            "id": rid,
            "title": title,
            "desc": desc,
            "pid": body.example_poem_id or None,
            "ex": _example_json(body),
        })
    return {"ok": True, "id": rid}


@app.get("/api/rules/{rid}")
def get_rule(rid: str):
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, code, title_ar, description_ar, status, function_name,
                   poems_matched, example_poem_id, example_qafiya_json,
                   created_at, coded_at, applied_at
            FROM rules WHERE id = CAST(:id AS uuid)
        """), {"id": rid}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="rule not found")
    d = dict(row._mapping)
    d["id"] = str(d["id"])
    d["example_poem_id"] = str(d["example_poem_id"]) if d["example_poem_id"] else None
    return d


@app.patch("/api/rules/{rid}")
def update_rule(rid: str, body: RuleIn):
    title = (body.title_ar or "").strip()
    desc = (body.description_ar or "").strip()
    if not title or not desc:
        raise HTTPException(status_code=400, detail="title_ar and description_ar are required")

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT status FROM rules WHERE id = CAST(:id AS uuid)"),
            {"id": rid},
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="rule not found")
        if row[0] != "proposed":
            raise HTTPException(
                status_code=409,
                detail=f"rule is {row[0]}, only 'proposed' rules can be edited",
            )
        conn.execute(text("""
            UPDATE rules SET
                title_ar = :title,
                description_ar = :desc,
                example_poem_id = :pid,
                example_qafiya_json = :ex
            WHERE id = CAST(:id AS uuid)
        """), {
            "id": rid,
            "title": title,
            "desc": desc,
            "pid": body.example_poem_id or None,
            "ex": _example_json(body),
        })
    return {"ok": True, "id": rid}


@app.delete("/api/rules/{rid}")
def delete_rule(rid: str):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT status FROM rules WHERE id = CAST(:id AS uuid)"),
            {"id": rid},
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="rule not found")
        if row[0] != "proposed":
            raise HTTPException(
                status_code=409,
                detail=f"rule is {row[0]}, only 'proposed' rules can be deleted",
            )
        conn.execute(
            text("DELETE FROM rules WHERE id = CAST(:id AS uuid)"),
            {"id": rid},
        )
    return {"ok": True}


# ── Static files + SPA fallback ───────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/{full_path:path}")
def spa(full_path: str):
    return FileResponse(str(STATIC / "index.html"))
