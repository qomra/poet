"""
Cross-source deduplication.

Strategy: hash the first verse (cleaned) + poet name (cleaned).
If two records share the same key, keep the one with more metadata.
Sources are ordered by priority — the first one seen wins.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator

from schema import ProcessedPoem

# Source priority — higher index = lower priority when deduplicating
_SOURCE_PRIORITY = ["ashaar", "dwianai"]


def _dedup_key(poem: ProcessedPoem) -> str:
    """Stable hash for cross-source duplicate detection."""
    # Use first verse + poet name, heavily normalised
    first_verse = poem.verses[0] if poem.verses else ""
    poet = poem.poet_name or ""

    # Strip everything non-alphabetic Arabic
    def _clean(s: str) -> str:
        s = unicodedata.normalize("NFC", s)
        # Remove diacritics
        s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", s)
        # Normalize alef variants
        s = re.sub(r"[أإآٱ]", "ا", s)
        # Remove tatweel, spaces, punctuation
        s = re.sub(r"[^\u0600-\u06FF]", "", s)
        return s.lower()

    raw = _clean(first_verse) + "|" + _clean(poet)
    return hashlib.sha1(raw.encode()).hexdigest()


def _source_rank(source: str) -> int:
    try:
        return _SOURCE_PRIORITY.index(source)
    except ValueError:
        return len(_SOURCE_PRIORITY)


def dedup(streams: Iterable[Iterable[ProcessedPoem]]) -> Iterator[ProcessedPoem]:
    """
    Merge multiple streams and yield deduplicated ProcessedPoem objects.
    When duplicates are found, the record from the higher-priority source wins.
    """
    seen: dict[str, ProcessedPoem] = {}

    for stream in streams:
        for poem in stream:
            key = _dedup_key(poem)
            if key not in seen:
                seen[key] = poem
            else:
                existing = seen[key]
                # Replace if this source has higher priority (lower rank)
                if _source_rank(poem.source) < _source_rank(existing.source):
                    seen[key] = poem
                # Or if this record has more metadata
                elif _source_rank(poem.source) == _source_rank(existing.source):
                    if _metadata_score(poem) > _metadata_score(existing):
                        seen[key] = poem

    yield from seen.values()


def _metadata_score(poem: ProcessedPoem) -> int:
    """Higher score = richer record. Used as tiebreaker within the same source."""
    score = 0
    if poem.title:
        score += 1
    if poem.poet_name:
        score += 2
    if poem.poet_era_raw:
        score += 1
    if poem.meter_raw:
        score += 1
    if poem.rhyme_raw:
        score += 1
    if poem.theme_raw:
        score += 1
    if poem.source_url:
        score += 1
    score += min(len(poem.verses), 10)  # longer poems score slightly higher
    return score
