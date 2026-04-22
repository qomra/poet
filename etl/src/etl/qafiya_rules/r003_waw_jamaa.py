"""
r003 — واو الجماعة في القافية

If AT LEAST ONE ajuz ends in the written pattern "وا" (the plural-verb subject
suffix + its silent alif), then the presence of that واو الجماعة is a signal
that the qafiya harakah is damma. Symmetric to r002: the signal verses expose
the wasl explicitly; the other verses end in the rawiy directly.

Outputs:
    - harakah = madmouma
    - wasl    = و
    - rawiy   = for ajuz ending in "وا", the char at [-3];
                for any other ajuz, the char at [-1] (its last letter).
                ALL these candidates must be identical.
    - radf    = letter one position before the rawiy ([-4] for waw-ending,
                [-2] for others) iff it repeats across all ajuz AND is a
                madd letter (ا/و/ي).
    - type    = "mutawatir" iff every ajuz has SOME madd letter at the
                pre-rawiy position (even if the specific madd letter varies).
                Otherwise, = "mutadaarik" iff none of the ajuz has a madd at
                pre-rawiy AND every ajuz has SOME madd at the pre-pre-rawiy
                position ([-5] for waa-ending, [-3] for non-waa-ending).
                Otherwise unset.

Notes:
    - Rawiy itself must not be a madd letter (ا/و/ي).
    - This rule catches e.g. "قالوا / قال / المقال / الطلول" where some
      verses show the plural-و wasl and others carry the rawiy directly.
"""
from __future__ import annotations

from etl.qafiya_rules import Rule, norm_char

_PUNCT_EDGE = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_MADD = {"ا", "و", "ي"}


def _strip_trailing(s: str) -> str:
    s = s.rstrip()
    while s and s[-1] in _PUNCT_EDGE:
        s = s[:-1].rstrip()
    return s


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    ajuzes = [_strip_trailing(v) for i, v in enumerate(verses) if i % 2 == 1]

    if len(ajuzes) < 3:
        return None
    if any(not a for a in ajuzes):
        return None

    ends_waa = [a.endswith("وا") for a in ajuzes]

    # At least one ajuz must carry the waw-jamaa signal
    if not any(ends_waa):
        return None

    # Rawiy candidate per ajuz: [-3] for waa-ending, [-1] otherwise.
    # Pre-rawiy and pre-pre-rawiy follow accordingly.
    rawiys: set[str] = set()
    pre_rawiy: list[str | None] = []
    pre_pre: list[str | None] = []
    for a, is_waa in zip(ajuzes, ends_waa):
        if is_waa:
            if len(a) < 3:
                return None
            rawiys.add(norm_char(a[-3]))
            pre_rawiy.append(a[-4] if len(a) >= 4 else None)
            pre_pre.append(a[-5] if len(a) >= 5 else None)
        else:
            rawiys.add(norm_char(a[-1]))
            pre_rawiy.append(a[-2] if len(a) >= 2 else None)
            pre_pre.append(a[-3] if len(a) >= 3 else None)

    if len(rawiys) != 1:
        return None
    rawiy = next(iter(rawiys))
    if rawiy in _MADD:
        return None

    radf: str | None = None
    if None not in pre_rawiy:
        uniq = set(pre_rawiy)
        if len(uniq) == 1:
            c = next(iter(uniq))
            if c in _MADD:
                radf = c

    type_: str | None = None
    if None not in pre_rawiy and all(c in _MADD for c in pre_rawiy):
        type_ = "mutawatir"
    elif None not in pre_pre and all(c in _MADD for c in pre_pre):
        type_ = "mutadaarik"

    result: dict = {
        "rawiy": rawiy,
        "wasl": "و",
        "harakah": "madmouma",
    }
    if radf:
        result["radf"] = radf
    if type_:
        result["type"] = type_
    return result


RULE = Rule(
    code="r003",
    title_ar="واو الجماعة في القافية",
    description_ar=(
        "إذا انتهى أحدُ أعجاز القصيدة على الأقل بنمط (وا) — وهو واو الجماعة "
        "المتبوع بالألف الفارقة — فهذا يدلّ على أن حركة القافية مضمومة، "
        "والوصل هو الواو. والرَّوِيّ هو الحرف الذي قبل الواو في الأعجاز "
        "المنتهية بـ(وا)، وهو ذاته آخر حرف في الأعجاز الأخرى — ويشترط "
        "تطابقه في الجميع. أما الرِّدف فهو الحرف السابق للرَّوِيّ إن تكرَّر "
        "في كل الأعجاز وكان حرف مدٍّ (ا/و/ي). وإن كان الحرف الذي قبل الرَّوِيّ "
        "دائمًا حرف مدٍّ — حتى وإن اختلف نوع المدّ بين الأعجاز — فالنَّوع متواتر. "
        "وإن لم يكن الحرف الذي قبل الرَّوِيّ مدّيًا في أيّ من الأعجاز ولكن الحرف "
        "الذي قبله بحرفين كان دائمًا حرف مدٍّ في كل الأعجاز، فالنَّوع متدارك. "
        "وإلا تُرك النَّوع دون تحديد."
    ),
    match=match,
)
