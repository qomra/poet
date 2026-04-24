"""
r014 — المتراكب (mutarakib)

Classical definition: قافية متراكبة has **3 متحركات between the last two
سواكن** at the end of each ajuz.

Detected by walking the diacritized ajuz (text_tashkeel):
    1. Parse each Arabic letter and the diacritic(s) that follow it into
       (letter, state) where state ∈ {mu (vowelled), sa (saakin)}.
       Shadda expands the letter into (sa)(mu) of the same consonant.
    2. Find the indices of the last two saakins.
    3. Require exactly 3 متحركات strictly between them.
    4. Require the *final* saakin to be a wasl-eligible letter
       (ا / ى / و / ي / ن / ه); otherwise the qafiya is muqayyada and this
       rule doesn't apply.

If ≥ 80 % of ajuzes match the pattern AND share the same rawiy (= the last
متحرك before the final saakin, hamza-normalised), classify the poem:
    rawiy   = that letter
    harakah = the vowel carried by that rawiy across ajuzes
    wasl    = kasra → ي, damma → و, fatha → ا
    type    = mutarakib
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TANWIN_FATH, _TANWIN_DAMM, _TANWIN_KASR = "ً", "ٌ", "ٍ"
_SHADDA, _SUKUN, _DAGGER_ALIF = "ّ", "ْ", "ٰ"

_FATHA_SET = {_FATHA, _TANWIN_FATH, _DAGGER_ALIF}
_DAMMA_SET = {_DAMMA, _TANWIN_DAMM}
_KASRA_SET = {_KASRA, _TANWIN_KASR}
_VOWELS = _FATHA_SET | _DAMMA_SET | _KASRA_SET

_WASL_LETTERS = {"ا", "ى", "و", "ي", "ن", "ه"}
_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_MIN_AJUZES = 3
_RATIO = 0.8


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip_trailing(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _parse_letters(s: str) -> list[tuple[str, str, str | None]]:
    """Return [(letter, 'mu'|'sa', harakah_or_None)] for each letter slot."""
    result: list[tuple[str, str, str | None]] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if not _is_arabic_letter(c):
            i += 1
            continue

        # Collect subsequent diacritics up to the next letter
        j = i + 1
        shadda = False
        vowel: str | None = None
        sukun = False
        while j < n and not _is_arabic_letter(s[j]):
            ch = s[j]
            if ch == _SHADDA:
                shadda = True
            elif ch in _VOWELS:
                vowel = ch
            elif ch == _SUKUN:
                sukun = True
            j += 1

        if shadda:
            # Shadda = gemination: first copy is saakin, second carries vowel
            result.append((c, "sa", None))
            if vowel is not None:
                result.append((c, "mu", vowel))
            elif sukun:
                result.append((c, "sa", None))
            else:
                result.append((c, "sa", None))
        else:
            if vowel is not None:
                result.append((c, "mu", vowel))
            elif sukun:
                result.append((c, "sa", None))
            else:
                # Unmarked letter: treat as saakin (typical for word-final)
                result.append((c, "sa", None))
        i = j
    return result


def _append_implicit_wasl(letters: list[tuple[str, str, str | None]]) -> None:
    """If the ajuz ends in a vowelled letter (not sukun), append an implicit
    wasl saakin derived from the final harakah:
        kasra → ي,  damma → و,  fatha → ا
    This is the classical qafiya convention: the wasl is phonologically present
    even when the poet omits it in writing.
    """
    if not letters or letters[-1][1] != "mu":
        return
    harakah = letters[-1][2]
    if harakah in _KASRA_SET:
        letters.append(("ي", "sa", None))
    elif harakah in _DAMMA_SET:
        letters.append(("و", "sa", None))
    elif harakah in _FATHA_SET:
        letters.append(("ا", "sa", None))


def _analyze(tashkeel: str) -> tuple[str, str] | None:
    """Return (rawiy_letter, harakah_char) if this ajuz matches mutarakib."""
    s = _strip_trailing(tashkeel)
    if len(s) < 4:
        return None
    letters = _parse_letters(s)
    if not letters:
        return None
    _append_implicit_wasl(letters)
    if len(letters) < 5:
        return None

    saakin_idx = [i for i, (_, st, _) in enumerate(letters) if st == "sa"]
    if len(saakin_idx) < 2:
        return None

    last_sa = saakin_idx[-1]
    prev_sa = saakin_idx[-2]

    # Final saakin must be a wasl letter (otherwise muqayyada)
    if norm_char(letters[last_sa][0]) not in _WASL_LETTERS:
        return None

    # Count متحركات strictly between the two saakins
    muharrikat_positions = [
        i for i in range(prev_sa + 1, last_sa)
        if letters[i][1] == "mu"
    ]
    if len(muharrikat_positions) != 3:
        return None

    # Rawiy = the last متحرك (closest to the final saakin)
    rawiy_pos = muharrikat_positions[-1]
    rawiy_letter, _, rawiy_harakah = letters[rawiy_pos]
    if rawiy_harakah is None:
        return None
    return rawiy_letter, rawiy_harakah


def match(poem: dict) -> dict | None:
    verses_tk = poem.get("verses_tashkeel") or []
    ajuzes_tk = [v for i, v in enumerate(verses_tk) if i % 2 == 1 and v]
    if len(ajuzes_tk) < _MIN_AJUZES:
        return None

    hits = [_analyze(a) for a in ajuzes_tk]
    good = [h for h in hits if h is not None]
    if len(good) < _RATIO * len(ajuzes_tk):
        return None

    # Pick the dominant rawiy (hamza-normalised)
    rawiy_counts = Counter(norm_char(r[0]) for r in good)
    rawiy, rawiy_n = rawiy_counts.most_common(1)[0]
    if rawiy_n < _RATIO * len(good):
        return None

    # Pick the dominant harakah among ajuzes with that rawiy
    harakah_counts = Counter(r[1] for r in good if norm_char(r[0]) == rawiy)
    if not harakah_counts:
        return None
    top_harakah, top_n = harakah_counts.most_common(1)[0]
    if top_n < _RATIO * rawiy_n:
        return None

    if top_harakah in _KASRA_SET:
        harakah, wasl = "maksura", "ي"
    elif top_harakah in _DAMMA_SET:
        harakah, wasl = "madmouma", "و"
    elif top_harakah in _FATHA_SET:
        harakah, wasl = "maftouha", "ا"
    else:
        return None

    return {
        "rawiy": rawiy,
        "harakah": harakah,
        "wasl": wasl,
        "type": "mutarakib",
    }


RULE = Rule(
    code="r014",
    title_ar="المتراكب",
    description_ar=(
        "إذا وُجدت ثلاثة متحركات بين آخر ساكِنَيْن في عَجُز القصيدة — كما يظهر "
        "في التشكيل — ومع اشتراط أن يكون الساكن النهائي من حروف الوصل "
        "(ا/ى/و/ي/ن/ه) لا من صامت ساكن (أي لا تكون القافية مُقيَّدة) — "
        "فالنَّوع متراكب. يُؤخذ الرَّوِيّ هو آخر المتحركات الثلاثة (الأقرب إلى "
        "الساكن النهائي). تُحدَّد حركة القافية من حركة الرَّوِيّ، ويُشتق منها "
        "الوصل: كسرة ⇒ ياء، ضمة ⇒ واو، فتحة ⇒ ألف. تنطبق القاعدة إذا تكرَّر "
        "النمط في ٨٠٪ من الأعجاز على الأقل وتطابق الرَّوِيّ."
    ),
    match=match,
)
