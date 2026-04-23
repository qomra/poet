"""
r010 — ياء الوصل كمؤشر للكسر (بدون حرف جر)

Like r002 (preposition + ya) but WITHOUT requiring a preposition.

The written ي at the end of some ajuzes is the orthographic kasra signal —
in Arabic poetry, a ي appended to the end of a verse indicates the rawiy
carries kasra (ياء الوصل follows kasra phonologically).

The disambiguator — same as r002:
  SOME ajuzes end in "C-ي" (C = rawiy, ي = wasl)
  OTHER ajuzes end in just "C" or "C" with kasra mark

When ي ends EVERY ajuz, ي itself is the rawiy (not wasl) — that case is
excluded. When only SOME ajuzes have ي, ي is the wasl and C is the rawiy.

Type detection (from undiacritized plain text and/or tashkeel):
  - If pre-rawiy consistently has sukun/shadda in tashkeel → mutadaarik
  - If pre-rawiy consistently is a long vowel → mutawatir
  - Else → type left unset

Requires no tashkeel — works from plain text alone.
"""
from __future__ import annotations
from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_MADD  = {"ا", "و", "ي"}
_WASL  = {"ا", "و", "ي", "ن", "ه"}
_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_SHADDA = "ّ"
_SUKUN  = "ْ"


def _strip(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _arabic(c: str) -> bool:
    return "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ"


def _last_letters(s: str, n: int = 4) -> list[str]:
    """Last n Arabic letters from s, reversed order (index 0 = last)."""
    result = []
    for c in reversed(s):
        if _arabic(c):
            result.append(norm_char(c))
            if len(result) == n:
                break
    return result


def _tokens(s: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for c in s:
        if _arabic(c):
            if buf:
                out.append(buf)
            buf = c
        else:
            buf += c
    if buf:
        out.append(buf)
    return out


def _pre_rawiy_type(diacs: str, rawiy: str) -> str | None:
    """
    From diacritized text, find the pre-rawiy token and return:
      'madd'     — it is a long vowel (متواتر signal)
      'suk_sha'  — it has sukun or shadda (متدارك signal)
      None       — unknown
    """
    if not diacs:
        return None
    toks: list[str] = []
    buf = ""
    for c in diacs:
        if _arabic(c):
            if buf:
                toks.append(buf)
            buf = c
        else:
            buf += c
    if buf:
        toks.append(buf)
    if not toks:
        return None
    # find rawiy token from end (skip wasl)
    i = len(toks) - 1
    while i >= 0:
        b = norm_char(toks[i][0]) if toks[i] and _arabic(toks[i][0]) else ""
        if b and b not in _WASL:
            # check matches rawiy
            if norm_char(b) == rawiy and i >= 1:
                pre = toks[i - 1]
                pb = norm_char(pre[0]) if pre and _arabic(pre[0]) else ""
                if pb in _MADD and not any(c in pre for c in "ًٌٍَُِّ"):
                    return "madd"
                if _SUKUN in pre or _SHADDA in pre:
                    return "suk_sha"
                return None
            break
        i -= 1
    return None


def match(poem: dict) -> dict | None:
    verses   = poem.get("verses") or []
    verses_t = poem.get("verses_tashkeel") or []

    ajuzes   = [_strip(v) for i, v in enumerate(verses)   if i % 2 == 1]
    ajuzes_t = [_strip(v) for i, v in enumerate(verses_t) if i % 2 == 1]

    if len(ajuzes) < 3:
        return None
    ajuzes = [a for a in ajuzes if a]
    if not ajuzes:
        return None

    # Classify each ajuz: does it end in ي (potential wasl)?
    ends_ya   = [a.endswith("ي") for a in ajuzes]
    ya_count  = sum(ends_ya)
    non_count = len(ajuzes) - ya_count

    # Disambiguator: SOME end in ي, SOME don't
    if ya_count == 0 or non_count == 0:
        return None

    # Extract rawiy from each ajuz
    rawiys: list[str] = []
    for a, is_ya in zip(ajuzes, ends_ya):
        letters = _last_letters(a, 3)
        if is_ya:
            # ي is wasl → rawiy is letters[1] (letter before ي)
            if len(letters) < 2:
                continue
            r = letters[1]
        else:
            # rawiy is last letter (may or may not be in _WASL; if it is,
            # might be a bare wasl — still take it as rawiy candidate)
            if not letters:
                continue
            r = letters[0]
        if r and r not in _MADD:
            rawiys.append(r)

    if not rawiys:
        return None

    # Rawiy must be consistent (≥ 80 %)
    top_rawiy, cnt = Counter(rawiys).most_common(1)[0]
    if cnt / len(rawiys) < 0.80:
        return None
    rawiy = top_rawiy

    # The rawiy from ي-verses must match the rawiy from non-ي-verses
    ya_rawiys  = [r for a, is_ya in zip(ajuzes, ends_ya)
                  if is_ya
                  for r in [_last_letters(a, 3)[1]
                             if len(_last_letters(a, 3)) >= 2 else None]
                  if r and r not in _MADD]
    noy_rawiys = [r for a, is_ya in zip(ajuzes, ends_ya)
                  if not is_ya
                  for r in [_last_letters(a, 2)[0]
                             if _last_letters(a, 2) else None]
                  if r and r not in _MADD]

    if ya_rawiys and noy_rawiys:
        ya_top  = Counter(ya_rawiys).most_common(1)[0][0]
        noy_top = Counter(noy_rawiys).most_common(1)[0][0]
        if ya_top != noy_top:
            return None    # different rawiy in ي vs non-ي groups

    # Type detection from tashkeel
    pre_types = [_pre_rawiy_type(t, rawiy)
                 for a, t in zip(ajuzes, ajuzes_t) if a and t]
    pre_types = [x for x in pre_types if x]

    type_: str | None = None
    if pre_types:
        type_counts = Counter(pre_types)
        dominant, dcnt = type_counts.most_common(1)[0]
        if dcnt / len(pre_types) >= 0.60:
            type_ = "mutadaarik" if dominant == "suk_sha" else "mutawatir"

    result: dict = {"rawiy": rawiy, "harakah": "maksura", "wasl": "ي"}
    if type_:
        result["type"] = type_
    return result


RULE = Rule(
    code="r010",
    title_ar="ياء الوصل كمؤشر للكسر (بدون حرف جر)",
    description_ar=(
        "إذا انتهى بعض الأعجاز بحرف متحرك مسبوق بياء (كـ قِي) وانتهى "
        "بعضها الآخر بنفس الحرف بدون ياء، فإن الياء وصلٌ يدل على أن القافية "
        "مكسورة، والرَّوِيّ هو الحرف الذي قبل الياء — ويجب أن يكون هو ذاته "
        "آخر حرف في الأعجاز الخالية من الياء. لا تشترط هذه القاعدة حرفَ جر "
        "(كما تفعل r002)، بل تعتمد على الياء المكتوبة ذاتها مؤشرًا صرفيًا. "
        "النوع: متدارك إن وُجد سكون/شدة على الحرف قبل الرَّوِيّ، متواتر إن "
        "وُجد حرف مد."
    ),
    match=match,
)
