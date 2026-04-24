"""
r016 — القصائد المُرَكَّبة (composite / mis-merged records)

Detects records where a single poem entry actually contains two distinct
qasidas concatenated — a common data-ingestion artifact when the upstream
source shares a title prefix across two works.

Signal: scan the ajuz rawiy sequence in order. If there is a single split
point where:
  - the first segment is dominated (≥ 75 %) by one rawiy A
  - the second segment is dominated (≥ 75 %) by a *different* rawiy B
  - each segment has ≥ 3 ajuzes
then the record is composite.

Runs after r006 (free verse) so it only catches STRUCTURED composites
(each half has its own consistent rhyme); fully varied poems are already
classified as free verse by r006.

Output: type = "composite", other qafiya fields left empty. The poem
gets a rule_id so it drops out of the unclassified pool, but the empty
qafiya fields signal to downstream consumers not to treat it as a
single-rhyme qasida.
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_PUNCT  = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_WASL_1 = {"ي", "ا", "ى", "و", "ه", "ن"}

_MIN_AJUZ = 10          # genuine merged composites are long (two qasidas)
_HALF_MIN = 5           # 5 baits per half as minimum evidence
_DOMINANCE = 0.90       # strict — avoid false-positives on phonological variation


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip_trailing(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _last_letter(ajuz: str) -> str | None:
    """Strip trailing wasl decoration, return the last remaining consonant."""
    s = _strip_trailing(ajuz)
    # Strip the double "وا" pair first
    if s.endswith("وا"):
        s = s[:-2]
    # Then peel trailing single wasl letters
    while s and s[-1] in _WASL_1:
        s = s[:-1]
    for c in reversed(s):
        if _is_arabic_letter(c):
            return norm_char(c)
    return None


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    # Odd-verse-count → no paired structure; leave to r006
    if len(verses) % 2 == 1:
        return None
    ajuzes = [v for i, v in enumerate(verses) if i % 2 == 1 and v and v.strip()]
    if len(ajuzes) < _MIN_AJUZ:
        return None

    rawiys = [_last_letter(a) for a in ajuzes]
    if any(r is None for r in rawiys):
        return None
    n = len(rawiys)

    # Scan every viable split point and accept the first one that satisfies
    # dominance in both halves.
    for split in range(_HALF_MIN, n - _HALF_MIN + 1):
        first, second = rawiys[:split], rawiys[split:]
        dom1, c1 = Counter(first).most_common(1)[0]
        dom2, c2 = Counter(second).most_common(1)[0]
        if dom1 == dom2:
            continue
        if c1 >= _DOMINANCE * len(first) and c2 >= _DOMINANCE * len(second):
            return {"type": "composite"}
    return None


RULE = Rule(
    code="r016",
    title_ar="القصائد المُرَكَّبة (دَمْجُ قصيدتَيْن في سجلٍّ واحد)",
    description_ar=(
        "تكتشف السجلات التي تحتوي في الواقع على قصيدتين متتاليتين مدموجتين "
        "في مدخل واحد — وهو خطأ شائع في استيراد البيانات عندما يتشارك "
        "مصدران في بداية العنوان.\n"
        "الكشف: يُقرأ تسلسل الأروية (rawiy) في الأعجاز من أول القصيدة إلى "
        "آخرها؛ إذا وُجدت نقطة قطع وحيدة يكون فيها:\n"
        "  — القسم الأول مهيمِناً فيه روي (A) بنسبة ≥٧٥٪\n"
        "  — القسم الثاني مهيمِناً فيه روي آخر (B) بنسبة ≥٧٥٪\n"
        "  — كل قسم يحتوي على ٣ أعجاز فأكثر\n"
        "فالقصيدة مُركَّبة. يُضبط النَّوع بـ'composite' وتُترك حقول القافية "
        "فارغة، وبذلك تُستبعد من مجموعة القصائد غير المصنَّفة دون أن تُعامَل "
        "على أنها قافية ثابتة واحدة."
    ),
    match=match,
)
