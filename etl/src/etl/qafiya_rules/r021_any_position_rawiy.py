"""
r021 — مطابقة الرَّوِيّ على جميع الأبيات الحقيقية (احتياطي نهائي)

Final fallback. Some poems store ajuzes in irregular layouts (mixed
placeholders, half-paired half-flat, dialect verse with unconventional
formatting). After all earlier rules fail, this one looks at EVERY real
verse — regardless of position parity — and accepts the qafiya if ≥ 80 %
of those verses end in the same consonant (after hamza/ta-marbuta
normalisation).

Conservative: emits rawiy only, no harakah. Type left unset.
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, is_real_verse, norm_char

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _last_letter(s: str | None) -> str | None:
    if not s:
        return None
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    for c in reversed(s):
        if _is_arabic_letter(c):
            return norm_char(c)
    return None


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    real = [v for v in verses if is_real_verse(v)]
    if len(real) < 4:
        return None
    letters = [_last_letter(v) for v in real]
    letters = [l for l in letters if l]
    if len(letters) < 4:
        return None
    counts = Counter(letters)
    top, top_n = counts.most_common(1)[0]
    if top_n / len(letters) < 0.80:
        return None
    return {"rawiy": top}


RULE = Rule(
    code="r021",
    title_ar="مطابقة الرَّوِيّ على جميع الأبيات الحقيقية (احتياطي نهائي)",
    description_ar=(
        "احتياطي أخير يعمل بعد فشل جميع القواعد السابقة. يفحص الحرف "
        "الأخير في كل بيت حقيقي (مع تجاهل ‹...› والأبيات الفارغة) بصرف "
        "النظر عن موقعه (زوجي / فردي). إن تطابق هذا الحرف في ٨٠٪ من "
        "الأبيات على الأقل، يُعدّ هو الرَّوِيّ. لا تُحدَّد الحركة هنا."
    ),
    match=match,
)
