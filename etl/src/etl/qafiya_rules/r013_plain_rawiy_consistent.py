"""
r013 — الرَّوِيّ الثابت (من النص الخام بدون تشكيل)

Fallback for poems that have a clear consistent rawiy detectable from the
undiacritized plain text, but where tashkeel-based rules (r004/r008/r009)
failed — typically because the tashkeel model assigned grammatically
"correct" harakah that doesn't match the poetic qafiya harakah.

Detection (from plain text only):
  Strip trailing wasl letters (ي، ا، ى، و، ه، ن) from each ajuz.
  The last remaining Arabic letter is the rawiy candidate.
  If ≥ 80 % of ajuzes share the same rawiy → consistent rawiy.

Harakah inference (best-effort, in priority order):
  1. Preposition before qafiya word → kasra
  2. Trailing ي (written wasl) in some verses → kasra
  3. Trailing ون / وا (waw group endings) → madmouma (already r003)
  4. Tashkeel majority vote if available (≥ 50 % any single harakah)
  5. Else → harakah unset

Radf: if the letter immediately before the rawiy is consistently a long
vowel (ا/و/ي) across ≥ 80 % of ajuzes → radf = that letter, type = mutawatir.

This rule runs AFTER all tashkeel-based rules so it only catches what
they missed.
"""
from __future__ import annotations
from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_WASL_STRIP = {"ي", "ا", "ى", "و", "ه", "ن"}
_MADD       = {"ا", "و", "ي"}
_PREPS      = {"من", "في", "على", "الى", "عن", "حتى", "منذ", "مذ", "كي", "رب"}
_KASRA_CH   = "ِ"
_DAMMA_CH   = "ُ"
_FATHA_CH   = "َ"
_TK         = "ٍ"
_TD         = "ٌ"
_TF         = "ً"
_PUNCT      = set(".,!?;:،؛؟…—-\"'()[]{}«»")


def _strip(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _arabic(c: str) -> bool:
    return "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ"


def _last_letters(s: str, n: int = 3) -> list[str]:
    result = []
    for c in reversed(s):
        if _arabic(c):
            result.append(norm_char(c))
            if len(result) == n:
                break
    return result


def _rawiy_from_plain(s: str) -> tuple[str | None, str | None]:
    """Returns (rawiy, pre_rawiy) after stripping wasl."""
    letters = []
    for c in reversed(s):
        if _arabic(c):
            letters.append(norm_char(c))
    # strip trailing wasl candidates
    i = 0
    while i < len(letters) and letters[i] in _WASL_STRIP:
        i += 1
    rawiy = letters[i] if i < len(letters) else None
    pre   = letters[i + 1] if i + 1 < len(letters) else None
    return rawiy, pre


def _has_prep(plain: str) -> bool:
    words = plain.split()
    for w in words[-3:-1] or []:
        if w in _PREPS:
            return True
    if len(words) >= 3 and words[-3] in _PREPS:
        return True
    if len(words) >= 2 and words[-2] in _PREPS:
        return True
    last = words[-1] if words else ""
    if len(last) >= 4 and last[0] in {"ب", "ل", "ك"} and last[1:3] == "ال":
        return True
    return False


def _tashkeel_harakah(diacs: str, rawiy: str) -> str | None:
    """Best-effort harakah from tashkeel for a specific rawiy."""
    if not diacs:
        return None
    buf = ""
    toks = []
    for c in diacs:
        if _arabic(c):
            if buf:
                toks.append(buf)
            buf = c
        else:
            buf += c
    if buf:
        toks.append(buf)
    for tok in reversed(toks):
        b = norm_char(tok[0]) if tok and _arabic(tok[0]) else ""
        if b and b not in _WASL_STRIP:
            if norm_char(b) == rawiy:
                if _KASRA_CH in tok or _TK in tok:
                    return "maksura"
                if _DAMMA_CH in tok or _TD in tok:
                    return "madmouma"
                if _FATHA_CH in tok or _TF in tok:
                    return "maftouha"
            break
    return None


def match(poem: dict) -> dict | None:
    verses   = poem.get("verses") or []
    verses_t = poem.get("verses_tashkeel") or []

    ajuzes   = [_strip(v) for i, v in enumerate(verses)   if i % 2 == 1]
    ajuzes_t = [_strip(v) for i, v in enumerate(verses_t) if i % 2 == 1]
    ajuzes   = [a for a in ajuzes if a]

    if len(ajuzes) < 3:
        return None

    # Extract rawiy and pre-rawiy from plain text
    parsed = [_rawiy_from_plain(a) for a in ajuzes]
    parsed = [(r, p) for r, p in parsed if r and r not in _MADD]

    if not parsed:
        return None

    n = len(parsed)
    rawiy_counts = Counter(r for r, _ in parsed)
    top_rawiy, top_cnt = rawiy_counts.most_common(1)[0]
    if top_cnt / n < 0.80:
        return None
    rawiy = top_rawiy

    # Radf: consistent pre-rawiy long vowel
    pre_chars = [p for r, p in parsed if r == rawiy and p and p in _MADD]
    radf: str | None = None
    type_: str | None = None
    if len(pre_chars) / n >= 0.80:
        top_pre, pre_cnt = Counter(pre_chars).most_common(1)[0]
        if pre_cnt / n >= 0.70:
            radf = top_pre
            type_ = "mutawatir"

    # Harakah inference — priority order
    harakah: str | None = None
    wasl: str | None = None

    # P1: preposition
    if any(_has_prep(a) for a in ajuzes):
        harakah = "maksura"
        wasl = "ي"

    # P2: trailing ي in some (but not all) ajuzes → kasra
    if harakah is None:
        ends_ya = [a.endswith("ي") for a in ajuzes]
        if any(ends_ya) and not all(ends_ya):
            harakah = "maksura"
            wasl = "ي"

    # P3: tashkeel majority
    if harakah is None and ajuzes_t:
        h_votes = []
        for a, t in zip(ajuzes, ajuzes_t):
            if t:
                h = _tashkeel_harakah(t, rawiy)
                if h:
                    h_votes.append(h)
        if h_votes:
            top_h, h_cnt = Counter(h_votes).most_common(1)[0]
            if h_cnt / len(h_votes) >= 0.50:
                harakah = top_h
                if harakah == "maksura":
                    wasl = "ي"
                elif harakah == "madmouma":
                    wasl = "و"
                elif harakah == "maftouha":
                    wasl = "ا"

    result: dict = {"rawiy": rawiy}
    if harakah:
        result["harakah"] = harakah
    if wasl:
        result["wasl"] = wasl
    if radf:
        result["radf"] = radf
    if type_:
        result["type"] = type_
    return result


RULE = Rule(
    code="r013",
    title_ar="الرَّوِيّ الثابت من النص الخام (احتياطي)",
    description_ar=(
        "قاعدة احتياطية للقصائد التي يتضح فيها الرَّوِيّ من النص غير المشكَّل "
        "لكن فشلت فيها القواعد المعتمدة على التشكيل (r004/r008/r009) — غالبًا "
        "لأن النموذج أعطى القافية شكلًا نحويًا صحيحًا يخالف حركة القافية "
        "الشعرية. تُجرَّد حروف الوصل من نهاية كل عجز ويُحدَّد الرَّوِيّ، "
        "فإن ثبت في ≥80٪ من الأعجاز قُبِل. الحركة: بالأولوية (جر→كسر، "
        "ياء الوصل→كسر، التشكيل بالأغلبية، وإلا تُترَك). "
        "الردف: حرف المد الثابت قبل الرَّوِيّ في ≥80٪."
    ),
    match=match,
)
