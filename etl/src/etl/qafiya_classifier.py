"""
Mechanistic qafiya classifier.

Processes verse endings to extract:
  - rawiy letter
  - radf (long vowel before rawiy)
  - wasl (letter after rawiy)
  - harakah (inferred from wasl)
  - type (mutaradif if detectable, else indeterminate)
  - ta'siis

Does NOT require diacritics. Uses suffix pattern analysis across 6+ verses.
Writes results back to poems table with a confidence level.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# Arabic character sets
LONG_VOWELS  = set("اوي")
CONSONANTS   = set("بتثجحخدذرزسشصضطظعغفقكلمنهء")
TA_MARBUTA   = "ة"
ALEF_MAQSURA = "ى"
WASL_LETTERS = set("اوين")
TASHKEEL     = set("ًٌٍَُِّْ")
TANWIN_FATH  = "\u064B"   # ً
TANWIN_KASR  = "\u064D"   # ٍ
TANWIN_DAMM  = "\u064C"   # ٌ
FATHA        = "\u064E"   # َ
KASRA        = "\u0650"   # ِ
DAMMA        = "\u064F"   # ُ
SUKUN        = "\u0652"   # ْ


@dataclass
class QafiyaResult:
    rawiy: str | None = None
    radf: str | None = None
    wasl: str | None = None
    harakah: str | None = None   # maftouha / maksura / madmouma / muqayyada
    type_: str | None = None     # mutaradif / indeterminate
    taassis: bool = False
    pattern: str | None = None   # observed ending e.g. "ـون"
    confidence: str = "none"     # high / medium / low / none


def _normalize(text: str) -> str:
    """Minimal normalization — keep tashkeel (needed for harakah detection)."""
    text = unicodedata.normalize("NFC", text)
    # Normalize alef variants
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Remove tatweel
    text = text.replace("\u0640", "")
    return text.strip()


def _get_last_word(verse: str) -> str:
    """Extract last word from verse text."""
    verse = _normalize(verse)
    # Split on spaces and take last non-empty token
    words = [w for w in verse.split() if w.strip()]
    if not words:
        return ""
    last = words[-1]
    # Remove trailing punctuation except Arabic chars + tashkeel
    last = re.sub(r"[^\u0600-\u06FF]", "", last)
    return last


def _strip_tashkeel(text: str) -> str:
    """Remove all diacritic marks from text."""
    return re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", text)


def _detect_harakah_from_tanwin(word: str) -> str | None:
    """Detect harakah from tanwin marks (written even without full tashkeel)."""
    if TANWIN_FATH in word:
        return "maftouha"
    if TANWIN_KASR in word:
        return "maksura"
    if TANWIN_DAMM in word:
        return "madmouma"
    return None


def _detect_harakah_from_explicit(word: str, rawiy_idx: int) -> str | None:
    """Detect harakah from explicit haraka mark on rawiy position."""
    if rawiy_idx < len(word) - 1:
        next_char = word[rawiy_idx + 1]
        if next_char == FATHA:
            return "maftouha"
        if next_char == KASRA:
            return "maksura"
        if next_char == DAMMA:
            return "madmouma"
        if next_char == SUKUN:
            return "muqayyada"
    return None


def _infer_harakah_from_wasl(wasl: str) -> str | None:
    """Infer harakah from the wasl letter (phonological law — no diacritics needed)."""
    if wasl == "ي":
        return "maksura"    # ياء الوصل only follows kasra
    if wasl == "و":
        return "madmouma"   # واو الوصل only follows damma
    if wasl == "ا":
        return "maftouha"   # ألف الوصل only follows fatha
    if wasl == "ن":
        return "maftouha"   # تنوين (ألف التنوين) typically fatha
    return None


def _parse_word_ending(word: str) -> dict:
    """
    Parse the qafiya components from a single word.
    Returns dict with: rawiy, radf, wasl, taassis, harakah_hint
    """
    if not word:
        return {}

    bare = _strip_tashkeel(word)
    result = {}

    # --- Try tanwin detection first (most reliable harakah source) ---
    harakah_from_tanwin = _detect_harakah_from_tanwin(word)
    if harakah_from_tanwin:
        result["harakah_hint"] = harakah_from_tanwin

    # Work from the end of the bare (no-tashkeel) word
    n = len(bare)
    if n == 0:
        return result

    # Pattern: handle ة (ta marbuta) at end — it's the rawiy
    if bare[-1] == TA_MARBUTA or bare[-1] == ALEF_MAQSURA:
        result["rawiy"] = bare[-1]
        # Look for radf (long vowel) one position before
        if n >= 2 and bare[-2] in LONG_VOWELS:
            result["radf"] = bare[-2]
            # Check ta'siis: alif 3 positions back with one consonant between
            if n >= 4 and bare[-4] == "ا" and bare[-3] not in LONG_VOWELS:
                result["taassis"] = True
        return result

    # Pattern: last char is a consonant (most common case)
    last = bare[-1]

    # Check if last char is a wasl candidate (ا،و،ي،ن،ه after rawiy)
    if last in WASL_LETTERS and n >= 2:
        wasl = last
        rawiy = bare[-2]
        result["wasl"] = wasl
        result["rawiy"] = rawiy
        # Infer harakah from wasl if not already known
        if "harakah_hint" not in result:
            h = _infer_harakah_from_wasl(wasl)
            if h:
                result["harakah_hint"] = h
        # Look for radf before rawiy
        if n >= 3 and bare[-3] in LONG_VOWELS:
            result["radf"] = bare[-3]
            # Check ta'siis
            if n >= 5 and bare[-5] == "ا" and bare[-4] not in LONG_VOWELS:
                result["taassis"] = True
    else:
        # Last char is the rawiy
        result["rawiy"] = last
        # Look for radf before rawiy
        if n >= 2 and bare[-2] in LONG_VOWELS:
            result["radf"] = bare[-2]
            if n >= 4 and bare[-4] == "ا" and bare[-3] not in LONG_VOWELS:
                result["taassis"] = True

    return result


def classify(verse_texts: list[str], min_verses: int = 6) -> QafiyaResult:
    """
    Classify the qafiya of a poem from its verse texts.
    Requires min_verses consistent verses.
    """
    result = QafiyaResult()

    # Extract last words
    last_words = [_get_last_word(v) for v in verse_texts]
    last_words = [w for w in last_words if w]

    if len(last_words) < min_verses:
        result.confidence = "none"
        return result

    # --- Step 1: Find the dominant suffix pattern ---
    # Look at last 1, 2, 3, 4 characters — find what is MOST consistent
    suffix_lengths = [1, 2, 3, 4]
    best_suffix = None
    best_agreement = 0.0

    for length in suffix_lengths:
        suffixes = [_strip_tashkeel(w)[-length:] for w in last_words if len(_strip_tashkeel(w)) >= length]
        if not suffixes:
            continue
        counter = Counter(suffixes)
        top, top_count = counter.most_common(1)[0]
        agreement = top_count / len(suffixes)
        # Prefer longer suffixes with high agreement
        if agreement >= 0.70 and len(top) > len(best_suffix or ""):
            best_suffix = top
            best_agreement = agreement

    if not best_suffix:
        result.confidence = "none"
        return result

    result.pattern = "ـ" + best_suffix

    # --- Step 2: Parse rawiy/radf/wasl from the dominant suffix ---
    # Use the most common last word (matching the dominant suffix)
    bare_words = [_strip_tashkeel(w) for w in last_words]
    matching = [w for w in last_words if _strip_tashkeel(w).endswith(best_suffix)]

    if not matching:
        result.confidence = "none"
        return result

    # Parse from a representative matching word
    parsed = _parse_word_ending(matching[0])

    # Cross-validate rawiy across all matching words
    if "rawiy" in parsed:
        rawiy_votes = Counter(
            _parse_word_ending(w).get("rawiy") for w in matching
            if _parse_word_ending(w).get("rawiy")
        )
        if rawiy_votes:
            result.rawiy = rawiy_votes.most_common(1)[0][0]

    # Radf — must be consistent
    if "radf" in parsed:
        radf_votes = Counter(
            _parse_word_ending(w).get("radf") for w in matching
            if _parse_word_ending(w).get("radf")
        )
        if radf_votes:
            top_radf, radf_count = radf_votes.most_common(1)[0]
            if radf_count / len(matching) >= 0.70:
                result.radf = top_radf

    # Wasl — must be consistent
    if "wasl" in parsed:
        wasl_votes = Counter(
            _parse_word_ending(w).get("wasl") for w in matching
            if _parse_word_ending(w).get("wasl")
        )
        if wasl_votes:
            result.wasl = wasl_votes.most_common(1)[0][0]

    # Harakah — from wasl (most reliable) then from tanwin
    if result.wasl:
        result.harakah = _infer_harakah_from_wasl(result.wasl)
    if not result.harakah:
        harakah_votes = Counter(
            _parse_word_ending(w).get("harakah_hint") for w in matching
            if _parse_word_ending(w).get("harakah_hint")
        )
        if harakah_votes:
            top_h, h_count = harakah_votes.most_common(1)[0]
            if h_count / len(matching) >= 0.60:
                result.harakah = top_h

    # Ta'siis
    taassis_count = sum(
        1 for w in matching if _parse_word_ending(w).get("taassis")
    )
    if taassis_count / len(matching) >= 0.70:
        result.taassis = True

    # --- Step 3: Type detection ---
    # مترادف: last two chars are both consonants (no long vowel between)
    if result.rawiy and not result.radf and not result.wasl:
        # Check if rawiy is preceded by a consonant (not a long vowel)
        bare_endings = [_strip_tashkeel(w)[-3:] for w in matching if len(_strip_tashkeel(w)) >= 3]
        if bare_endings:
            # If the char before rawiy is a consonant (not ا،و،ي) → possible مترادف
            pre_rawiy = Counter(e[-2] if len(e) >= 2 else "" for e in bare_endings)
            top_pre, pre_count = pre_rawiy.most_common(1)[0]
            if top_pre and top_pre not in LONG_VOWELS and pre_count / len(bare_endings) >= 0.70:
                result.type_ = "mutaradif"

    # --- Step 4: Confidence ---
    if result.rawiy and result.harakah and result.radf is not None:
        result.confidence = "high" if best_agreement >= 0.85 else "medium"
    elif result.rawiy and (result.harakah or result.radf is not None):
        result.confidence = "medium" if best_agreement >= 0.75 else "low"
    elif result.rawiy:
        result.confidence = "low"
    else:
        result.confidence = "none"

    return result


def _select_qafiya_verses(verse_texts: list[str]) -> list[str]:
    """
    Determine which verses carry the qafiya.
    In ashaar data, verses are stored as hemistichs (شطر).
    The qafiya falls on every OTHER hemistich.
    Try odd positions (1,3,5..) then even (0,2,4..) — return whichever
    has better internal consistency.
    """
    if len(verse_texts) < 4:
        return verse_texts

    odd  = verse_texts[1::2]   # عجز hemistichs
    even = verse_texts[0::2]   # صدر hemistichs

    def agreement(verses: list[str]) -> float:
        if not verses:
            return 0.0
        endings = [_strip_tashkeel(_get_last_word(v))[-1:] for v in verses if _get_last_word(v)]
        if not endings:
            return 0.0
        top, cnt = Counter(endings).most_common(1)[0]
        return cnt / len(endings)

    odd_agreement  = agreement(odd)
    even_agreement = agreement(even)
    all_agreement  = agreement(verse_texts)

    # Return the subset with best agreement, fallback to all
    best = max(
        (odd_agreement, odd),
        (even_agreement, even),
        (all_agreement, verse_texts),
    )[1]
    return best


# Patch classify() to use qafiya-verse selection
_original_classify = classify

def classify(verse_texts: list[str], min_verses: int = 6) -> QafiyaResult:  # noqa: F811
    qafiya_verses = _select_qafiya_verses(verse_texts)
    return _original_classify(qafiya_verses, min_verses=min(min_verses, len(qafiya_verses)))
