"""
r002 — وجود حرف جر وياء

Heuristic: if an Arabic preposition precedes the qafiya word in at least one
ajuz, AND some ajuz lines end in ي while others end in a consonant, then:
    - harakah = maksura  (preposition puts the qafiya noun in majrur → kasra)
    - wasl    = ي          (the ya at the end of some verses is the wasl)
    - rawiy   = the letter BEFORE the ya in ya-ending ajuz, which MUST equal
                the last letter of non-ya-ending ajuz. Single consistent value
                across all ajuz is required.
    - radf    = letter at one position before the rawiy ([-3] for ya-ending,
                [-2] for non-ya-ending). Must be a madd letter (ا / و / ي) and
                identical across all ajuz.
    - type    = "mutawatir" iff every ajuz has SOME madd letter at the pre-rawiy
                position (even if the specific madd letter varies). Otherwise,
                = "mutadaarik" iff none of the ajuz has a madd at pre-rawiy AND
                every ajuz has SOME madd at the pre-pre-rawiy position
                ([-4] for ya-ending, [-3] for non-ya-ending). Otherwise unset.

The "some end in ي / some don't" contrast is the disambiguator: without it we
cannot distinguish wasl-ya from rawiy-ya. The preposition is the grammatical
signal confirming kasra.

Preposition detection is intentionally conservative:
    - standalone prepositions (من، في، على، إلى→الى، عن، حتى، منذ، مذ، كي، رب)
      in any of the last 3 tokens of the ajuz
    - attached prefix بال / لل / كال on the qafiya word itself
More nuanced cases (attached بـ/لـ/كـ without ال, prep inside idafa chains)
are left to future rules.
"""
from __future__ import annotations

from etl.qafiya_rules import Rule, norm_char

_PUNCT_EDGE = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_MADD = {"ا", "و", "ي"}
_PREPS = {"من", "في", "على", "الى", "عن", "حتى", "منذ", "مذ", "كي", "رب"}


def _strip_trailing(s: str) -> str:
    s = s.rstrip()
    while s and s[-1] in _PUNCT_EDGE:
        s = s[:-1].rstrip()
    return s


def _has_preposition(ajuz: str) -> bool:
    words = ajuz.split()
    if not words:
        return False
    # Standalone preposition in last 3 tokens
    for w in words[-3:-1] or []:
        if w in _PREPS:
            return True
    # Check words[-3] explicitly (above slice skips if fewer than 3 tokens)
    if len(words) >= 3 and words[-3] in _PREPS:
        return True
    if len(words) >= 2 and words[-2] in _PREPS:
        return True
    # Attached preposition prefix + definite article on the qafiya word
    last = words[-1]
    if len(last) >= 4 and last[0] in {"ب", "ل", "ك"} and last[1:3] == "ال":
        return True
    return False


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    ajuzes = [_strip_trailing(v) for i, v in enumerate(verses) if i % 2 == 1]

    if len(ajuzes) < 3:
        return None
    if any(not a for a in ajuzes):
        return None

    # Disambiguator: must have BOTH ya-ending and non-ya-ending ajuz
    ends_ya = [a.endswith("ي") for a in ajuzes]
    if not any(ends_ya) or all(ends_ya):
        return None

    # At least one ajuz must carry a preposition signal
    if not any(_has_preposition(a) for a in ajuzes):
        return None

    # Extract rawiy candidate from each ajuz.
    # norm_char collapses hamza forms: ئ / ؤ / أ / إ / آ → ء
    # so "اعدائي" (rawiy=ئ) and "الماء" (rawiy=ء) resolve to the same ء.
    rawiys: set[str] = set()
    for a, is_ya in zip(ajuzes, ends_ya):
        if is_ya:
            if len(a) < 2:
                return None
            rawiys.add(norm_char(a[-2]))
        else:
            rawiys.add(norm_char(a[-1]))
    if len(rawiys) != 1:
        return None
    rawiy = next(iter(rawiys))
    if rawiy in _MADD:
        # rawiy shouldn't be a madd letter itself under this rule
        return None

    # Pre-rawiy letters: [-3] for ya-ending ajuz, [-2] for non-ya-ending ajuz.
    # Pre-pre-rawiy: one position further back.
    pre_rawiy: list[str | None] = []
    pre_pre: list[str | None] = []
    for a, is_ya in zip(ajuzes, ends_ya):
        pre_idx = -3 if is_ya else -2
        ppre_idx = -4 if is_ya else -3
        pre_rawiy.append(a[pre_idx] if len(a) >= abs(pre_idx) else None)
        pre_pre.append(a[ppre_idx] if len(a) >= abs(ppre_idx) else None)

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
        "wasl": "ي",
        "harakah": "maksura",
    }
    if radf:
        result["radf"] = radf
    if type_:
        result["type"] = type_
    return result


RULE = Rule(
    code="r002",
    title_ar="وجود حرف جر وياء",
    description_ar=(
        "إذا وُجد في أحد أعجاز القصيدة حرفُ جرٍّ قبل كلمة القافية، وانتهى "
        "بعض الأعجاز بالياء (ي) وبقيّتها بحرف صامت، فإن القافية مكسورة، "
        "والوصل ياء، والرَّوِيّ هو الحرف الذي قبل الياء في الأعجاز المنتهية "
        "بها، وهو ذاته آخر حرف في الأعجاز الأخرى — ويشترط تطابقه في الجميع. "
        "أما الرِّدف فهو الحرف السابق للرَّوِيّ إن تكرّر في كل الأعجاز وكان "
        "حرف مدٍّ (ا/و/ي). وإن كان الحرف الذي قبل الرَّوِيّ دائمًا حرف مدٍّ "
        "(ا/ي/و) في كل الأعجاز — حتى وإن اختلف نوع المدّ بينها — فالنَّوع "
        "متواتر. وإن لم يكن الحرف الذي قبل الرَّوِيّ مدّيًا في أيّ من الأعجاز "
        "ولكن الحرف الذي قبله بحرفين كان دائمًا حرف مدٍّ في كل الأعجاز، "
        "فالنَّوع متدارك. وإلا تُرك النَّوع دون تحديد."
    ),
    match=match,
)
