"""FastAPI backend for the Qafiya Annotator."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI
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
def get_poem(mode: str = "classified", confidence: str = "all"):
    where = ["p.verse_count >= 6"]

    if mode == "classified":
        where.append("p.qafiya_confidence IS NOT NULL AND p.qafiya_confidence != 'none'")
        if confidence != "all":
            where.append(f"p.qafiya_confidence = '{confidence}'")
    elif mode == "unclassified":
        where.append("(p.qafiya_confidence IS NULL OR p.qafiya_confidence = 'none')")

    where.append("NOT EXISTS (SELECT 1 FROM qafiya_annotations qa WHERE qa.poem_id = p.id)")

    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT p.id, p.title, p.poet_name, p.meter, p.rhyme_letter,
                   p.verse_count, p.era, p.qafiya_rawiy, p.qafiya_radf,
                   p.qafiya_wasl, p.qafiya_harakah, p.qafiya_type,
                   p.qafiya_pattern, p.qafiya_confidence,
                   array_agg(v.text ORDER BY v.position) as verses
            FROM poems p JOIN verses v ON v.poem_id = p.id
            WHERE {' AND '.join(where)}
            GROUP BY p.id ORDER BY random() LIMIT 1
        """)).fetchone()

    if not row:
        return JSONResponse({"error": "no_poems"}, status_code=404)

    d = dict(row._mapping)
    d["id"] = str(d["id"])
    return d


@app.get("/api/stats")
def get_stats():
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT
              count(*) FILTER (WHERE verse_count >= 6) as total,
              count(*) FILTER (WHERE qafiya_confidence='high') as high,
              count(*) FILTER (WHERE qafiya_confidence='medium') as medium,
              count(*) FILTER (WHERE qafiya_confidence='low') as low,
              count(*) FILTER (WHERE qafiya_confidence='none'
                               OR qafiya_confidence IS NULL) as unclassified
            FROM poems
        """)).fetchone()
        done = conn.execute(text("SELECT count(*) FROM qafiya_annotations")).scalar()
        pending = conn.execute(
            text("SELECT count(*) FROM classifier_guidelines WHERE NOT applied")
        ).scalar()
    d = dict(r._mapping)
    d["done"] = done
    d["pending_guidelines"] = pending
    return d


@app.get("/api/guidelines")
def get_guidelines():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, guideline, applied, created_at,
                   (SELECT poet_name FROM poems WHERE id = example_poem_id) as poet
            FROM classifier_guidelines ORDER BY created_at DESC LIMIT 50
        """)).fetchall()
    return [dict(r._mapping) | {"id": str(r._mapping["id"])} for r in rows]


class AnnotationIn(BaseModel):
    poem_id: str
    verdict: str
    notes: str | None = None
    guideline: str | None = None
    correct_rawiy: str | None = None
    correct_radf: str | None = None
    correct_wasl: str | None = None
    correct_harakah: str | None = None
    correct_type: str | None = None


@app.post("/api/annotate")
def annotate(body: AnnotationIn):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO qafiya_annotations
              (id, poem_id, verdict, correct_rawiy, correct_radf, correct_wasl,
               correct_harakah, correct_type, notes)
            VALUES (:id,:pid,:v,:r,:rd,:w,:h,:t,:n)
        """), {
            "id": str(uuid.uuid4()), "pid": body.poem_id,
            "v": body.verdict,
            "r": body.correct_rawiy or None,
            "rd": body.correct_radf or None,
            "w": body.correct_wasl or None,
            "h": body.correct_harakah or None,
            "t": body.correct_type or None,
            "n": body.notes or None,
        })
        if body.guideline and body.guideline.strip():
            conn.execute(text("""
                INSERT INTO classifier_guidelines (id, guideline, example_poem_id)
                VALUES (:id, :g, :pid)
            """), {
                "id": str(uuid.uuid4()),
                "g": body.guideline.strip(),
                "pid": body.poem_id,
            })
    return {"ok": True}


@app.post("/api/guidelines/{gid}/apply")
def mark_applied(gid: str):
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE classifier_guidelines SET applied = true WHERE id = :id"),
            {"id": gid},
        )
    return {"ok": True}


# ── Static files + SPA fallback ───────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/{full_path:path}")
def spa(full_path: str):
    return FileResponse(str(STATIC / "index.html"))
