"""
r001 — الألف في نهاية القافية في كل أبيات القصيدة

If every ajuz (odd-indexed verse) ends in an alif ا after trimming trailing
punctuation / whitespace, then:
    - harakah = maftouha
    - wasl    = ا
    - rawiy   = the character before the alif (must be identical across all ajuz)
    - radf    = char at [-3] iff it repeats across all ajuz AND is a madd letter
                (ا / و / ي). Otherwise radf is left unset.
    - type    = "mutawatir" iff every ajuz has SOME madd letter (ا/و/ي) at [-3],
                even if the specific madd letter varies across ajuz. Otherwise,
                = "mutadaarik" iff none of the ajuz has a madd at [-3] AND every
                ajuz has SOME madd at [-4]. Otherwise type is left unset.
                (Phonological basis: the nearest saakin behind the wasl-alif is
                the second saakin; 1 متحرك between saakins = متواتر,
                2 متحركات = متدارك.)

Notes:
    - verses.text is pre-normalized: diacritics stripped, أ/إ/آ/ٱ → ا, tatweel removed.
    - alif maqsura (ى) at the end of a word is treated as equivalent to ا here
      — it is an orthographic variant of the same terminal /aː/ sound.
    - A poem must have at least 3 ajuz lines for the rule to consider applying,
      otherwise the signal is too weak.
"""
from __future__ import annotations

from etl.qafiya_rules import Rule

_PUNCT_EDGE = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_MADD = {"ا", "و", "ي"}
_ALIF_FINAL = {"ا", "ى"}  # ى at word-end is pronounced as ا


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

    # every ajuz must end in alif (ا or ى) and have a preceding character
    for a in ajuzes:
        if len(a) < 2 or a[-1] not in _ALIF_FINAL:
            return None

    rawiys = {a[-2] for a in ajuzes}
    if len(rawiys) != 1:
        return None
    rawiy = next(iter(rawiys))
    # Exclusions:
    #   - rawiy=ا: two consecutive alifs, ambiguous
    #   - rawiy=و: "-وا" is waaw al-jamaa morphology handled by r003
    # rawiy=ي IS allowed — "-يا" endings with ya as rawiy are a valid pattern
    # (e.g. accusative nouns like "اللياليا / المعاليا").
    if rawiy in {"ا", "و"}:
        return None

    # Pre-rawiy letters (position [-3]) — drive radf and the mutawatir check.
    pre_rawiy = [a[-3] if len(a) >= 3 else None for a in ajuzes]
    # Pre-pre-rawiy (position [-4]) — drives the mutadaarik check.
    pre_pre = [a[-4] if len(a) >= 4 else None for a in ajuzes]

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
        # mutawatir did not apply (at least one pre-rawiy is not madd); the
        # next-out position is consistently madd across all ajuz → مُتدارك.
        type_ = "mutadaarik"

    result: dict = {
        "rawiy": rawiy,
        "wasl": "ا",
        "harakah": "maftouha",
    }
    if radf:
        result["radf"] = radf
    if type_:
        result["type"] = type_
    return result


RULE = Rule(
    code="r001",
    title_ar="الألف في نهاية القافية في كل أبيات القصيدة",
    description_ar=(
        "إذا انتهت جميع أعجاز القصيدة (بعد تجاهل علامات الترقيم في الأخير) "
        "بحرف الألف — سواءً كانت ألفًا صريحة (ا) أو ألفًا مقصورة (ى) — فإن "
        "حركة القافية مفتوحة، والوصل ألف، والرَّوِيّ هو الحرف الذي قبل الألف "
        "(ويشترط أن يكون هو نفسه في كل الأعجاز). أما الرِّدف فهو الحرف عند "
        "الموضع الثالث من الآخر [-3] إن تكرر في كل الأعجاز وكان حرف مدٍّ "
        "(ا/و/ي)، وإلا فلا ردف. وإن كان الحرف الذي قبل الرَّوِيّ دائمًا "
        "حرف مدٍّ (ا/ي/و) في كل الأعجاز — حتى وإن اختلف نوع المدّ بينها — "
        "فالنَّوع متواتر. وإن لم يكن الحرف الذي قبل الرَّوِيّ مدّيًا في أيّ من "
        "الأعجاز ولكن الحرف الذي قبله (الموضع قبل الرَّوِيّ بحرفين) كان دائمًا "
        "حرف مدٍّ في كل الأعجاز، فالنَّوع متدارك. وإلا تُرك النَّوع دون تحديد."
    ),
    match=match,
)
