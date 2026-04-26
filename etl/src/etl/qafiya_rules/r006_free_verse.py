"""
r006 — الشعر الحر / شعر التفعيلة

Identifies poems with no consistent qafiya (free verse / tafila).

Algorithm:
  For each ajuz (even-position hemistich), strip trailing wasl decoration:
    - "وا" (waw + alif — plural marker) stripped as a unit first
    - Then individual trailing: ي ، ا ، ى ، و ، ه ، ن
  Take the last remaining Arabic letter = the rawiy candidate.

  Count distinct rawiy candidates across the poem.

  → If ≥ 3 distinct AND the dominant letter covers < 80 % of ajuzes:
      the poem has no consistent qafiya → شعر حر.

This rule runs LAST (r006) so it only fires on poems that could not be
classified by any of the qafiya-pattern rules (r001–r005).

The result sets qafiya_type = "free" with no rawiy/harakah/wasl — the
poem is marked as classified (has a rule_id) so it is excluded from the
unclassified pool, but its qafiya fields remain empty to signal the
absence of a structured rhyme scheme.
"""
from __future__ import annotations
from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_PUNCT   = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_ARABIC  = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآةىؤئ")
# wasl letters stripped individually (after وا pair)
_WASL_1  = {"ي", "ا", "ى", "و", "ه", "ن"}


def _strip_wasl(s: str) -> str:
    """Strip trailing wasl decoration to expose the rawiy."""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    # strip وا as a unit first (plural suffix)
    if len(s) >= 2 and s[-2:] == "وا":
        s = s[:-2]
    # strip individual wasl letters
    while s and s[-1] in _WASL_1:
        s = s[:-1]
    return s


def _last_letter(s: str) -> str | None:
    """Last Arabic letter after wasl stripping."""
    stripped = _strip_wasl(s)
    for c in reversed(stripped):
        if c in _ARABIC:
            return norm_char(c)
    return None


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    # Odd verse_count → no classical sadr/ajuz pairing (free verse layout):
    # every line stands alone, so we scan them all.
    if len(verses) % 2 == 1:
        ajuzes = [v for v in verses if v and v.strip()]
    else:
        ajuzes = [v for i, v in enumerate(verses) if i % 2 == 1 and v and v.strip()]

    if len(ajuzes) < 3:
        return None

    letters = [_last_letter(a) for a in ajuzes]
    letters = [l for l in letters if l]

    if not letters:
        return None

    n = len(letters)
    counts = Counter(letters)
    dominant_pct = counts.most_common(1)[0][1] / n
    distinct = len(counts)

    # Free verse / mixed:
    #   - 3+ distinct rawiy candidates AND no dominant letter ≥ 80%, OR
    #   - 2+ distinct AND no dominant letter ≥ 70%  (catches the
    #     "ابتت / ابقق / ابتت" alternating couplet patterns that fall
    #     between r006's old strict threshold and r020's 2-distinct-with-
    #     deep-strip detection).
    if distinct >= 3 and dominant_pct < 0.80:
        return {"type": "free"}
    if distinct >= 2 and dominant_pct < 0.70:
        return {"type": "free"}

    return None


RULE = Rule(
    code="r006",
    title_ar="الشعر الحر (لا قافية ثابتة)",
    description_ar=(
        "يُحدَّد الشعر الحر بعدم وجود قافية ثابتة عبر أعجاز القصيدة. "
        "تُجرَّد من نهاية كل عجز حروفُ الوصل (وا، ي، ا، ى، و، ه، ن)، "
        "ثم يُستخرج الحرف الأخير الباقي (المرشَّح للرَّوِيّ). "
        "فإن بلغ عدد الحروف المختلفة ثلاثةً فأكثر ولم يتجاوز الحرفُ الأكثر "
        "تكرارًا 80٪ من الأعجاز، فالقصيدة شعرٌ حر لا قافية له. "
        "يُضبَط النَّوع بـ'free' وتُترَك حقول القافية فارغة."
    ),
    match=match,
)
