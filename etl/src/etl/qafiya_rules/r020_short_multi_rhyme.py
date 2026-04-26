"""
r020 — قصائد متعددة القافية القصيرة (مزدوج / متناوب / مركَّب قصير)

Catches short poems whose ajuzes use two distinct rawiys arranged in
clear groups — e.g. AABB rhyme schemes (مزدوج), tercets (AAA + B),
or short merged records that r016 missed for being below its 5-ajuz-
per-half minimum.

Detection on the deep-stripped rawiy sequence (using r019's strip):
    1. Exactly 2 distinct rawiys (after hamza/ta-marbuta normalisation).
    2. The minority rawiy still covers ≥ 25 % of ajuzes (so it's not a
       single-line outlier — that's r017's job).
    3. The 2 rawiys form GROUPS, not a perfect alternation: at least one
       run of ≥ 2 consecutive identical rawiys exists for each.

Output: type = "free" (closest available label — the poem doesn't have a
single qafiya across the whole text). Specific rawiys are intentionally
left empty.

Runs after r000-r019.
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, ajuzes_of
from etl.qafiya_rules.r019_pronoun_strip_rawiy import _deep_strip_to_rawiy

_MIN_AJUZ = 4
_MINORITY_FLOOR = 0.25
_RUN_MIN = 2


def _runs(seq: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if not seq:
        return out
    cur = seq[0]
    n = 1
    for x in seq[1:]:
        if x == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = x, 1
    out.append((cur, n))
    return out


def match(poem: dict) -> dict | None:
    plain, _ = ajuzes_of(poem)
    if len(plain) < _MIN_AJUZ:
        return None

    rawiys: list[str] = []
    for a in plain:
        r, _ = _deep_strip_to_rawiy(a)
        if r:
            rawiys.append(r)
    if len(rawiys) < _MIN_AJUZ:
        return None

    counts = Counter(rawiys)
    if len(counts) != 2:
        return None
    (top, top_n), (sec, sec_n) = counts.most_common(2)
    if sec_n / len(rawiys) < _MINORITY_FLOOR:
        return None

    # Require at least one run of ≥ 2 for each rawiy
    runs = _runs(rawiys)
    runs_top = [n for r, n in runs if r == top and n >= _RUN_MIN]
    runs_sec = [n for r, n in runs if r == sec and n >= _RUN_MIN]
    if not runs_top or not runs_sec:
        return None

    return {"type": "free"}


RULE = Rule(
    code="r020",
    title_ar="قصائد متعددة القافية القصيرة (مزدوج / متناوب)",
    description_ar=(
        "تُصنّف القصائد القصيرة التي تتبع نمط القافيتين المتعاقبتين — مثل "
        "AABB أو AAB AAB — والتي تفلت من r016 (التي تتطلّب ٥ أعجاز لكل "
        "نصف) ومن r006 (التي تتطلّب ٣ أحرف مميّزة فأكثر). تستخدم هذه "
        "القاعدة الاقتطاع العميق لـ r019 لاستخراج الرَّوِيّ، ثم تتأكد من:\n"
        "  ١) أنّ هناك رَوِيَّيْن مختلفَيْن فقط (بعد التطبيع).\n"
        "  ٢) أنّ الأقلّ منهما يمثّل ٢٥٪ فأكثر من الأعجاز (لا مجرّد بيت "
        "شاذ).\n"
        "  ٣) أنّ كل رَوِيٍّ يظهر في تسلسلٍ متّصل من بيتين على الأقل (نمط "
        "مجموعات لا تناوب فردي).\n"
        "تُحدَّد type=\"free\" دون رويٍّ واحد يمثّل القصيدة كلها."
    ),
    match=match,
)
