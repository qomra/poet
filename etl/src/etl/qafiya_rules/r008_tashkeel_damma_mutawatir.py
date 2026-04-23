"""
r008 — المضموم المتواتر (من التشكيل)

Mirror of r004 (المكسور المتواتر) but for damma (مضموم) endings.

If ≥ 80 % of diacritized ajuzes carry damma (ُ or tanwin damma ٌ) on the
rawiy, AND the pre-rawiy letter is consistently a long vowel (ا / و / ي)
across ≥ 80 % of ajuzes → harakah=madmouma, wasl=و, type=mutawatir.

Radf: the consistent pre-rawiy long vowel.

Example:
  لَفَرِيدُ  → rawiy=د, pre=ي (radf), harakah=damma → متواتر مضموم
  الْمَمْدُودُ → rawiy=د, pre=و (radf), harakah=damma → متواتر مضموم
"""
from __future__ import annotations

from etl.qafiya_rules import Rule, norm_char

_DAMMA       = "ُ"
_TANWIN_DAMM = "ٌ"
_MADD        = {"ا", "و", "ي"}
_WASL        = {"ا", "و", "ي", "ن", "ه"}
_PUNCT       = set(".,!?;:،؛؟…—-\"'()[]{}«»")


def _strip(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _tokens(s: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for c in s:
        if "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ":
            if buf:
                out.append(buf)
            buf = c
        else:
            buf += c
    if buf:
        out.append(buf)
    return out


def _base(tok: str) -> str:
    for c in tok:
        if "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ":
            return norm_char(c)
    return ""


def _has_damma(tok: str) -> bool:
    return _DAMMA in tok or _TANWIN_DAMM in tok


def _is_long_vowel(tok: str) -> bool:
    b = _base(tok)
    if b not in _MADD:
        return False
    return not any(c in tok for c in "ًٌٍَُِّ")


def _analyze(plain: str, diacs: str) -> dict | None:
    p, t = _strip(plain), _strip(diacs)
    if not p or not t:
        return None
    toks = _tokens(t)
    if not toks:
        return None

    # Walk from end, skip wasl tokens
    i = len(toks) - 1
    while i >= 0 and _base(toks[i]) in _WASL:
        i -= 1
    if i < 0:
        return None

    rawiy_tok = toks[i]
    rawiy     = _base(rawiy_tok)
    if not rawiy or rawiy in _MADD:
        return None

    damma    = _has_damma(rawiy_tok)
    pre_tok  = toks[i - 1] if i >= 1 else None
    pre_lv   = _is_long_vowel(pre_tok) if pre_tok else False
    pre_base = _base(pre_tok) if pre_tok else None

    return {"rawiy": rawiy, "damma": damma, "pre_lv": pre_lv,
            "pre_base": pre_base}


def match(poem: dict) -> dict | None:
    verses   = poem.get("verses") or []
    verses_t = poem.get("verses_tashkeel") or []

    ajuzes   = [_strip(v) for i, v in enumerate(verses)   if i % 2 == 1]
    ajuzes_t = [_strip(v) for i, v in enumerate(verses_t) if i % 2 == 1]

    if len(ajuzes) < 3:
        return None

    pairs = [(p, t) for p, t in zip(ajuzes, ajuzes_t)
             if p and t and t.strip()]
    if len(pairs) < len(ajuzes) * 0.5:
        return None

    analyses = [_analyze(p, t) for p, t in pairs]
    analyses = [a for a in analyses if a]
    if not analyses:
        return None

    n = len(analyses)

    # Consistent rawiy
    rawiys = {a["rawiy"] for a in analyses}
    if len(rawiys) != 1:
        return None
    rawiy = next(iter(rawiys))

    # ≥ 80 % damma on rawiy
    if sum(a["damma"] for a in analyses) / n < 0.80:
        return None

    # ≥ 80 % pre-rawiy is a long vowel → متواتر + radf
    pre_lv_count = sum(a["pre_lv"] for a in analyses)
    if pre_lv_count / n < 0.80:
        return None

    # Radf: which long vowel
    radf: str | None = None
    from collections import Counter
    pre_chars = [a["pre_base"] for a in analyses if a["pre_lv"] and a["pre_base"]]
    if pre_chars:
        top, cnt = Counter(pre_chars).most_common(1)[0]
        if cnt / n >= 0.70:
            radf = top

    result: dict = {"rawiy": rawiy, "harakah": "madmouma",
                    "wasl": "و", "type": "mutawatir"}
    if radf:
        result["radf"] = radf
    return result


RULE = Rule(
    code="r008",
    title_ar="المضموم المتواتر (من التشكيل)",
    description_ar=(
        "إذا أظهر التشكيل ضمةً (أو تنوين ضم) على الرَّوِيّ في ≥ 80٪ من "
        "الأعجاز، وكان الحرف الذي قبل الرَّوِيّ مباشرةً حرفَ مدٍّ (ا/و/ي) "
        "في ≥ 80٪ من الأعجاز، فإن القافية مضمومة والوصل واو والنوع متواتر. "
        "الردف: حرف المد الثابت قبل الرَّوِيّ. "
        "نظير r004 (المكسور المتواتر) للمضموم."
    ),
    match=match,
)
