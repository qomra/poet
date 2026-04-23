"""
r007 — المتواتر المشكوك في حركته

For poems with a consistent متواتر qafiya structure where the harakah
cannot be determined confidently from tashkeel alone (mixed/missing).

Detection:
  متواتر identified by: ≥ 80 % of ajuzes have sukun (ْ) OR shadda (ّ)
  on the pre-rawiy letter (the character immediately before the rawiy).

Harakah resolution priority:
  1. If any ajuz has a preposition (حرف جر) in its last 3 tokens →
     harakah = maksura (preposition puts its object in genitive = kasra).
  2. Else if ≥ 50 % of diacritized ajuzes show damma (ُ / ٌ) on the
     rawiy → harakah = madmouma.
  3. Else → harakah left unset (uncertain; tashkeel insufficient).

Result: rawiy + type=mutawatir + harakah (if determinable).
Radf: if pre-pre-rawiy ([-3] from end for non-wasl, [-4] for wasl endings)
is consistently a long vowel → radf = that letter.
"""
from __future__ import annotations

from etl.qafiya_rules import Rule, norm_char

_SUKUN  = "ْ"
_SHADDA = "ّ"
_DAMMA  = "ُ"
_TANWIN_DAMM = "ٌ"
_MADD   = {"ا", "و", "ي"}
_WASL   = {"ا", "و", "ي", "ن", "ه"}
_PUNCT  = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_PREPS  = {"من", "في", "على", "الى", "عن", "حتى", "منذ", "مذ", "كي", "رب",
           "ب", "ل", "ك", "لل", "بال", "كال"}


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


def _has_sukun_or_shadda(tok: str) -> bool:
    return _SUKUN in tok or _SHADDA in tok


def _has_damma(tok: str) -> bool:
    return _DAMMA in tok or _TANWIN_DAMM in tok


def _has_preposition(plain_ajuz: str) -> bool:
    words = plain_ajuz.split()
    if not words:
        return False
    # check last 3 words (excluding final qafiya word)
    for w in words[-3:-1] or []:
        if w in _PREPS:
            return True
    if len(words) >= 3 and words[-3] in _PREPS:
        return True
    if len(words) >= 2 and words[-2] in _PREPS:
        return True
    # attached prefix بال/لل on qafiya word
    last = words[-1]
    if len(last) >= 4 and last[0] in {"ب", "ل", "ك"} and last[1:3] == "ال":
        return True
    return False


def _analyze(plain: str, diacs: str) -> dict | None:
    p, t = _strip(plain), _strip(diacs)
    if not p:
        return None
    toks = _tokens(t) if t else []

    # From plain: find rawiy (walk from end, skip wasl)
    plain_chars = [c for c in reversed(p) if c not in _PUNCT and "؀" <= c <= "ۿ"]
    rawiy = None
    pre_base = None
    ppre_is_madd = False
    for i, c in enumerate(plain_chars):
        nc = norm_char(c)
        if rawiy is None:
            if nc not in _WASL:
                rawiy = nc
        elif pre_base is None:
            pre_base = nc
        else:
            ppre_is_madd = nc in _MADD
            break

    if not rawiy or rawiy in _MADD:
        return None

    # Pre-rawiy sukun/shadda: check tashkeel
    pre_suk_sha = False
    if toks and pre_base:
        # find pre-rawiy token in toks
        for i in range(len(toks) - 1, -1, -1):
            b = _base(toks[i])
            if b == rawiy:
                if i >= 1:
                    pre_suk_sha = _has_sukun_or_shadda(toks[i - 1])
                break
        # fallback: check second-to-last non-wasl token
        if not pre_suk_sha:
            non_wasl = [tk for tk in toks if _base(tk) not in _WASL and _base(tk)]
            if len(non_wasl) >= 2:
                pre_suk_sha = _has_sukun_or_shadda(non_wasl[-2])

    # Damma on rawiy (from tashkeel)
    damma = False
    if toks:
        for i in range(len(toks) - 1, -1, -1):
            b = _base(toks[i])
            if b == rawiy:
                damma = _has_damma(toks[i])
                break

    return {
        "rawiy": rawiy,
        "pre_base": pre_base,
        "ppre_is_madd": ppre_is_madd,
        "pre_suk_sha": pre_suk_sha,
        "damma": damma,
        "has_prep": _has_preposition(p),
    }


def match(poem: dict) -> dict | None:
    verses   = poem.get("verses") or []
    verses_t = poem.get("verses_tashkeel") or []

    ajuzes   = [_strip(v) for i, v in enumerate(verses)   if i % 2 == 1]
    ajuzes_t = [_strip(v) for i, v in enumerate(verses_t) if i % 2 == 1]

    if len(ajuzes) < 3:
        return None

    pairs = list(zip(ajuzes, ajuzes_t))
    analyses = [_analyze(p, t or "") for p, t in pairs if p]
    analyses = [a for a in analyses if a]
    if not analyses:
        return None

    n = len(analyses)

    # Consistent rawiy
    rawiys = {a["rawiy"] for a in analyses}
    if len(rawiys) != 1:
        return None
    rawiy = next(iter(rawiys))

    # ≥ 80 % pre-rawiy has sukun or shadda → متواتر structure
    suk_sha_pct = sum(a["pre_suk_sha"] for a in analyses) / n
    if suk_sha_pct < 0.80:
        return None

    # Harakah resolution
    harakah: str | None = None

    # Priority 1: any preposition → kasra
    if any(a["has_prep"] for a in analyses):
        harakah = "maksura"

    # Priority 2: ≥ 50 % damma on rawiy → madmouma
    if harakah is None:
        damma_pct = sum(a["damma"] for a in analyses) / n
        if damma_pct >= 0.50:
            harakah = "madmouma"

    # Priority 3: harakah stays None (uncertain)

    # Radf: if pre-pre-rawiy is consistently a long vowel
    ppre_madd = [a["ppre_is_madd"] for a in analyses]
    radf: str | None = None
    if sum(ppre_madd) / n >= 0.80:
        # determine which madd letter
        plain_chars_list = []
        for a_plain, _ in [(ajuzes[i], ajuzes_t[i] if i < len(ajuzes_t) else "")
                           for i in range(len(ajuzes)) if ajuzes[i]]:
            chars = [c for c in reversed(a_plain)
                     if c not in _PUNCT and "؀" <= c <= "ۿ"]
            if len(chars) >= 3:
                plain_chars_list.append(norm_char(chars[2]))
        if plain_chars_list:
            from collections import Counter
            top, cnt = Counter(plain_chars_list).most_common(1)[0]
            if top in _MADD and cnt / len(plain_chars_list) >= 0.80:
                radf = top

    result: dict = {"rawiy": rawiy, "type": "mutawatir"}
    if harakah:
        result["harakah"] = harakah
        if harakah == "maksura":
            result["wasl"] = "ي"
        elif harakah == "madmouma":
            result["wasl"] = "و"
    if radf:
        result["radf"] = radf
    return result


RULE = Rule(
    code="r007",
    title_ar="المتواتر المشكوك في حركته",
    description_ar=(
        "للقصائد ذات القافية المتواترة الواضحة — 80٪+ من الأعجاز بسكون أو "
        "شدة على الحرف قبل الرَّوِيّ — عندما لا يكون التشكيل كافيًا لتحديد "
        "الحركة بثقة. أولوية الحركة: (١) إن وُجد حرف جر في أي بيت قبل "
        "كلمة القافية → مكسورة. (٢) إن ظهرت الضمة على الرَّوِيّ في ≥50٪ "
        "من الأبيات المشكَّلة → مضمومة. (٣) وإلا تُترَك الحركة غير محددة."
    ),
    match=match,
)
