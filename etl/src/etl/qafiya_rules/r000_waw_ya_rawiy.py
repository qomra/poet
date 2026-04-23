"""
r000 — الواو والياء رَوِيًّا

Must run BEFORE r001 (hence code "r000" which sorts first).

r001 excludes و and ي as rawiy candidates on the assumption they are
always radf/wasl. But in some poems و or ي IS the rawiy, with the
trailing ا/ى being the wasl that indicates fatha.

Patterns detected:
  WAW as rawiy:
    ≥ 80 % of ajuzes end in "وى" or "وا"
    → rawiy = و, wasl = ا (fatha → alif-wasl), harakah = maftouha

  YA as rawiy:
    ≥ 80 % of ajuzes end in "يا" or "يًا"
    → rawiy = ي, wasl = ا (fatha → alif-wasl), harakah = maftouha
    (rare — يا ending where ي is root and ا is wasl, e.g. "قضا لِيَا")

Radf: the letter immediately before و/ي, if consistently a long vowel
across ≥ 80 % of ajuzes → radf = that letter, type = mutawatir.

Examples:
  هوى / الشكوى / الماوى  → rawiy=و, radf=?,  wasl=ى
  قضا لِيَا / ليلى        → rawiy=ي, radf=?, wasl=ا
"""
from __future__ import annotations
from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_MADD  = {"ا", "و", "ي"}
_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")
# Alif forms that can serve as wasl after rawiy و/ي
_ALIF  = {"ا", "ى"}


def _strip_punct(s: str | None) -> str:
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


def _detect_rawiy(ajuzes: list[str], target: str) -> tuple[float, list[str]]:
    """
    Check what fraction of ajuzes end in target+ا_form,
    and collect the pre-rawiy letters.
    """
    matches = 0
    pre_letters = []
    for a in ajuzes:
        letters = _last_letters(a, 3)
        if len(letters) >= 2 and letters[0] in _ALIF and letters[1] == target:
            matches += 1
            if len(letters) >= 3:
                pre_letters.append(letters[2])
    return matches / len(ajuzes) if ajuzes else 0.0, pre_letters


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    ajuzes = [_strip_punct(v) for i, v in enumerate(verses) if i % 2 == 1]
    ajuzes = [a for a in ajuzes if a]

    if len(ajuzes) < 3:
        return None

    n = len(ajuzes)

    # Try و as rawiy first (more common)
    waw_pct, waw_pre = _detect_rawiy(ajuzes, "و")
    if waw_pct >= 0.80:
        rawiy = "و"
        pre_chars = waw_pre
    else:
        # Try ي as rawiy
        ya_pct, ya_pre = _detect_rawiy(ajuzes, "ي")
        if ya_pct >= 0.80:
            rawiy = "ي"
            pre_chars = ya_pre
        else:
            return None

    # Radf: consistent long vowel before و/ي
    radf: str | None = None
    type_: str | None = None
    if pre_chars and len(pre_chars) / n >= 0.80:
        top, cnt = Counter(pre_chars).most_common(1)[0]
        if top in _MADD and cnt / len(pre_chars) >= 0.70:
            radf = top
            type_ = "mutawatir"

    # harakah: alif wasl after rawiy → fatha
    result: dict = {"rawiy": rawiy, "harakah": "maftouha", "wasl": "ا"}
    if radf:
        result["radf"] = radf
    if type_:
        result["type"] = type_
    return result


RULE = Rule(
    code="r000",
    title_ar="الواو والياء رَوِيًّا (قبل r001)",
    description_ar=(
        "تعالج هذه القاعدة حالة كون الواو أو الياء رَوِيًّا — لا وصلًا ولا ردفًا — "
        "وهو ما تستبعده r001 افتراضيًا. إذا انتهى ≥80٪ من الأعجاز بنمط 'وى' أو 'وا' "
        "فالرَّوِيّ واوٌ والوصل ألف وحركته الفتح. وإذا انتهى بـ 'يا' فالرَّوِيّ ياءٌ "
        "والوصل ألف والحركة فتح. الردف: حرف المد الثابت قبل الرَّوِيّ إن وجد. "
        "تُنفَّذ قبل r001 لتسبقها في المطالبة بهذه الأنماط."
    ),
    match=match,
)
