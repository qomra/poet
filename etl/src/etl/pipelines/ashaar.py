"""Pipeline: arbml/ashaar (HuggingFace local copy) → stream of ProcessedPoem"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console

from etl.normalize import normalize_meter, normalize_tradition
from schema import ProcessedPoem, Tradition

console = Console()
SOURCE = "ashaar"
_ROOT = Path(os.getenv("ALSHAER_ROOT", Path(__file__).parents[5]))
_ARROW_DIR = _ROOT / "dataset" / "ashaar"


def extract() -> Iterator[ProcessedPoem]:
    try:
        from datasets import load_from_disk
    except ImportError:
        raise RuntimeError("Install `datasets`: uv sync --package etl")

    console.print(f"[cyan]Loading {_ARROW_DIR}…[/]")
    ds = load_from_disk(str(_ARROW_DIR))
    console.print(f"[cyan]{len(ds):,} records found[/]")

    skipped = 0
    for rec in ds:
        verses: list[str] = rec.get("poem verses") or []
        verses = [v.strip() for v in verses if v and v.strip()]
        if not verses:
            skipped += 1
            continue

        meter_raw: str = rec.get("poem meter") or ""
        theme_raw: str = rec.get("poem theme") or ""
        lang_raw: str = rec.get("poem language type") or ""
        rhyme_raw: str = rec.get("rhyme") or ""
        era_raw: str = rec.get("poet era") or ""
        meter = normalize_meter(meter_raw)

        # ashaar is all classical قصيدة — form defaults to qasida
        yield ProcessedPoem(
            source=SOURCE,
            source_id=rec.get("poem url") or "",
            title=rec.get("poem title") or None,
            poet_name=rec.get("poet name") or None,
            poet_description=rec.get("poet description") or None,
            poet_era_raw=era_raw or None,
            poet_location_raw=rec.get("poet location") or None,
            poet_url=rec.get("poet url") or None,
            verses=verses,
            meter_raw=meter_raw or None,
            rhyme_raw=rhyme_raw or None,
            theme_raw=theme_raw or None,
            language_type_raw=lang_raw or None,
            tradition=normalize_tradition(lang_raw, meter),
            form="qasida",
            source_url=rec.get("poem url") or None,
            has_diacritics=False,
        )

    if skipped:
        console.print(f"[yellow]{SOURCE}: skipped {skipped} empty records[/]")
