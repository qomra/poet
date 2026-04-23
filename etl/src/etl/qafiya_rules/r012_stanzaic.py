"""
r012 — الشعر المقطعي (مخمس / مربع / موشح ونحوها)

Detects stanzaic poems where the internal rhyme changes per stanza —
مخمس (5-line), مربع (4-line), موشح, etc. — but the poem still has
clear stanzaic structure, so it is NOT free verse in the usual sense.

r006 misses these because after stripping single wasl letters, the last
letter looks deceptively uniform (e.g., everything reduces to "م" when
stanzas alternate between ـمَا, ـهُمْ, ـتِي endings — all end in م or
collapse after stripping).

Detection: use the last 2 Arabic letters as the "suffix key" (without
stripping wasl). If:
  - ≥ 3 distinct 2-char suffixes across ajuzes, AND
  - The dominant suffix covers < 65 % of ajuzes
→ the poem has no single rhyme → type = "free"
  (best available label; the form is stanzaic, not truly free verse,
  but the qafiya is indeterminate at poem level)

Requires at least 6 ajuzes (3 bayt) to have enough data.
"""
from __future__ import annotations
from collections import Counter

from etl.qafiya_rules import Rule

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")


def _strip_punct(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _last2(s: str) -> str:
    """Last 2 Arabic letters of s, as a 2-char string (or fewer if short)."""
    result = []
    for c in reversed(s):
        if "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ":
            result.append(c)
            if len(result) == 2:
                break
    return "".join(reversed(result))


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    ajuzes = [_strip_punct(v) for i, v in enumerate(verses) if i % 2 == 1 and v]
    ajuzes = [a for a in ajuzes if a]

    if len(ajuzes) < 6:   # need at least 3 stanzas worth
        return None

    suffixes = [_last2(a) for a in ajuzes]
    suffixes = [s for s in suffixes if len(s) == 2]

    if not suffixes:
        return None

    counts   = Counter(suffixes)
    distinct = len(counts)
    dominant_pct = counts.most_common(1)[0][1] / len(suffixes)

    if distinct >= 3 and dominant_pct < 0.65:
        return {"type": "free"}

    return None


RULE = Rule(
    code="r012",
    title_ar="الشعر المقطعي (مخمس / مربع / موشح)",
    description_ar=(
        "يرصد القصائد ذات البنية المقطعية التي تتغير فيها القافية الداخلية من "
        "مقطع إلى آخر، كالمخمس والمربع والموشح. تفشل r006 في اكتشاف هذه "
        "الأشكال لأن تجريد حرف الوصل الأخير يُخفي التنوع الحقيقي. تعتمد "
        "r012 على آخر حرفَيْن عربيَّيْن (دون تجريد) مؤشرًا للتنوع: إذا بلغ "
        "عدد اللاحقات الثنائية المختلفة ثلاثةً فأكثر وغطّت الأغلبية أقل من "
        "65٪ من الأعجاز، فالقصيدة مقطعية لا رويّ واحد ثابتًا على مستوى "
        "النص كله. يُعيَّن النوع 'free' إشارةً إلى غياب قافية موحدة."
    ),
    match=match,
)
