"""
a002 — تحديد نوع القافية بمعيار أرفق (٦٠٪)

Same logic as a001 but with a 60 % threshold instead of 70 %, and only
accepts the top count if it's a single count (not "2-tied"). Runs after
a001 so only fills cases that strict mode rejected.

Conservative additions:
    • If top count covers 60–69 % AND the SECOND count is also ≤ 2
      (mutawatir / mutadaarik), accept the top — both halves are still
      "common-Arabic" types and the dominance is real.
    • Otherwise leave it for manual / future analysis.
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import is_real_verse
from etl.qafiya_augment import Augment
from etl.qafiya_augment.a001_type_from_tashkeel import _count_muharrikat, _TYPE_BY_COUNT


def fn(poem: dict) -> dict:
    if poem.get("qafiya_type"):
        return {}
    verses_tk = poem.get("verses_tashkeel") or []
    ajuzes_tk = [v for i, v in enumerate(verses_tk) if i % 2 == 1 and is_real_verse(v)]
    if len(ajuzes_tk) < 3:
        return {}
    counts = [c for c in (_count_muharrikat(a) for a in ajuzes_tk) if c is not None]
    if len(counts) < 3:
        return {}
    cc = Counter(counts)
    common = cc.most_common(2)
    top, top_n = common[0]
    pct = top_n / len(counts)
    if pct < 0.60:
        return {}
    if pct >= 0.70:
        # a001 should have caught this; if it didn't (very rare race), still emit.
        pass
    if len(common) > 1:
        # Only accept the looser case when the second-most is also ≤2 muharrikat
        # (i.e., a mutawatir/mutadaarik mix, not exotic patterns).
        sec = common[1][0]
        if not (top <= 2 and sec <= 2):
            return {}
    if top >= 4:
        return {"type": "mutakaasis"}
    if top in _TYPE_BY_COUNT:
        return {"type": _TYPE_BY_COUNT[top]}
    return {}


AUGMENT = Augment(
    code="a002",
    title_ar="تحديد النوع بمعيار أرفق",
    description_ar=(
        "تكميل لـ a001 بمعيار ٦٠٪ بدلاً من ٧٠٪، مع شرط ألّا يتعدى التنوّع "
        "بين متواتر/متدارك (≤ ٢ متحركات في كل القياسات الأكثر شيوعًا). "
        "يلتقط القصائد التي تشكيل تشكيل ٦٠–٦٩٪ من أعجازها على نوع واحد، "
        "مع ضمان أن البديل أيضًا من الأنواع الشائعة. لا يكتب إن كانت الحقول "
        "مملوءة."
    ),
    fn=fn,
)
