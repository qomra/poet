"""
r004 — المكسور المتواتر (tashkeel-based)

Uses the diacritized text (text_tashkeel) to detect kasra on the rawiy.

When tashkeel is available:
- Strip trailing wasl letters (ي / و / ا / ن / ه) from each diacritized ajuz.
- Read the explicit harakah mark on the last remaining consonant.
- If ≥ 80 % of ajuz endings carry kasra (ِ) on the rawiy, classify:
    harakah = maksura
    wasl    = ي   (kasra → ya-wasl is the standard Arabic phonological rule)
- If the character at pre-rawiy position (before the rawiy) is consistently
  a long vowel (ا / و / ي) → type = mutawatir.
- Rawiy is taken from the undiacritized ajuz (last non-wasl consonant,
  consistent across all ajuzes); same norm_char hamza normalisation as r001–r003.

Rationale: r002 detects kasra via grammatical preposition signals. r004 is
complementary — it works even when no preposition is visible but tashkeel is
present (e.g. ياء المتكلم possessive endings, majrur via idafa, adverbs, etc.).
"""
from __future__ import annotations
import re

from etl.qafiya_rules import Rule, norm_char

_KASRA  = "ِ"   # ِ
_FATHA  = "َ"   # َ
_DAMMA  = "ُ"   # ُ
_SUKUN  = "ْ"   # ْ
_MADD   = {"ا", "و", "ي"}
_WASL   = {"ا", "و", "ي", "ن", "ه"}
_PUNCT  = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_DIACS  = re.compile(r"[ؐ-ًؚ-ٰٟ]")


def _strip_trailing(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _last_harakah_on_rawiy(diacritized: str) -> str | None:
    """
    From a diacritized ajuz string, strip trailing wasl candidates
    (ي / و / ا / ن / ه) and their diacritics, then return the harakah
    mark (if any) on the last remaining consonant.
    """
    s = _strip_trailing(diacritized)
    if not s:
        return None

    # Walk backwards, skipping diacritics and wasl-eligible letters.
    # Stop when we hit a consonant that is NOT a wasl candidate.
    i = len(s) - 1
    while i >= 0:
        c = s[i]
        if c in _DIACS.pattern or c in "ًٌٍَُِّْ":
            i -= 1
            continue
        # c is an Arabic letter
        if c in _WASL:
            i -= 1
            continue
        # c is the rawiy consonant — look for the harakah that follows it
        j = i + 1
        while j < len(s):
            nc = s[j]
            if nc == _KASRA:
                return "kasra"
            if nc == _FATHA:
                return "fatha"
            if nc == _DAMMA:
                return "damma"
            if nc == _SUKUN:
                return "sukun"
            if nc in _DIACS.pattern or nc in "ًٌٍَُِّْ":
                j += 1
                continue
            # Another letter — no explicit harakah found
            break
        return None
    return None


def _rawiy_from_plain(ajuz: str) -> str | None:
    """Last non-wasl consonant from the undiacritized ajuz."""
    s = _strip_trailing(ajuz)
    for c in reversed(s):
        if c in _PUNCT or c in _DIACS.pattern:
            continue
        nc = norm_char(c)
        if nc not in _WASL:
            return nc
    return None


def _pre_rawiy_from_plain(ajuz: str) -> str | None:
    """The character immediately before the rawiy (and before any wasl)."""
    s = _strip_trailing(ajuz)
    found_rawiy = False
    for c in reversed(s):
        if c in _PUNCT:
            continue
        nc = norm_char(c)
        if not found_rawiy:
            if nc not in _WASL:
                found_rawiy = True
        else:
            if nc not in _WASL:
                return nc
    return None


def match(poem: dict) -> dict | None:
    verses   = poem.get("verses") or []
    verses_t = poem.get("verses_tashkeel") or []

    # Tashkeel must be present for at least half the ajuzes
    ajuzes   = [_strip_trailing(v) for i, v in enumerate(verses)   if i % 2 == 1]
    ajuzes_t = [_strip_trailing(v) for i, v in enumerate(verses_t) if i % 2 == 1]

    if len(ajuzes) < 3:
        return None

    has_t = sum(1 for v in ajuzes_t if v and v.strip())
    if has_t < len(ajuzes) * 0.5:
        return None   # not enough tashkeel coverage

    # -- Step 1: consistent rawiy from undiacritized text --
    rawiys = {_rawiy_from_plain(a) for a in ajuzes if a}
    rawiys.discard(None)
    if len(rawiys) != 1:
        return None
    rawiy = next(iter(rawiys))
    if rawiy in _MADD:
        return None   # rawiy shouldn't be a madd letter

    # -- Step 2: harakah from tashkeel --
    kasra_count = 0
    valid_t = 0
    for at in ajuzes_t:
        if not at:
            continue
        h = _last_harakah_on_rawiy(at)
        if h is not None:
            valid_t += 1
            if h == "kasra":
                kasra_count += 1

    if valid_t == 0 or kasra_count / valid_t < 0.80:
        return None

    # -- Step 3: pre-rawiy → radf + type --
    pre_chars = [_pre_rawiy_from_plain(a) for a in ajuzes if a]
    pre_chars = [c for c in pre_chars if c]

    radf: str | None = None
    type_: str | None = None

    if pre_chars:
        uniq_pre = set(pre_chars)
        if len(uniq_pre) == 1:
            c = next(iter(uniq_pre))
            if c in _MADD:
                radf = c
        # mutawatir: consistently a long vowel before the rawiy
        if all(c in _MADD for c in pre_chars):
            type_ = "mutawatir"

    result: dict = {
        "rawiy": rawiy,
        "harakah": "maksura",
        "wasl": "ي",
    }
    if radf:
        result["radf"] = radf
    if type_:
        result["type"] = type_
    return result


RULE = Rule(
    code="r004",
    title_ar="المكسور المتواتر (من التشكيل)",
    description_ar=(
        "إذا كانت القصيدة المشكَّلة تُظهر في ≥ 80٪ من أعجازها كسرةً على آخر "
        "حرف صامت جوهري (الرَّوِيّ)، وكان ما قبل الرَّوِيّ حرفَ مدٍّ في كل "
        "الأعجاز، فإن القافية مكسورة والوصل ياء والنوع متواتر. تعمل هذه القاعدة "
        "على التشكيل المستنتج من النموذج، وتكمّل r002 في حالات لا يظهر فيها "
        "حرف الجر صراحةً (كياء المتكلم، والإضافة، والظروف المجرورة)."
    ),
    match=match,
)
