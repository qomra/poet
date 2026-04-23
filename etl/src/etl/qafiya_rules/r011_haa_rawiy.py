"""
r011 — الهاء رَوِيًّا (هاء الضمير / هاء التأنيث)

Handles poems where ه is the rawiy — not a wasl letter.
All other rules include ه in the _WASL skip set, so they miss this pattern.

Common patterns:
  ...اهُ  → radf=ا, rawiy=ه, harakah=madmouma  (يَلْقَاهُ، رَعَاهُ)
  ...اهَا → radf=ا, rawiy=ه, harakah=maftouha  (رَاسَهَا، but ajuz only)
  ...هِ   → rawiy=ه, harakah=maksura

Detection (from undiacritized plain text):
  ≥ 80 % of ajuzes have ه as their last Arabic letter.

Harakah (from tashkeel):
  Look at the harakah on the final ه token:
  - Damma (ُ or ٌ) → madmouma
  - Kasra (ِ or ٍ) → maksura
  - Fatha (َ or ً) → maftouha
  - Majority vote across all diacritized ajuzes.

Radf (from undiacritized plain text):
  The letter immediately before ه — if it is a long vowel (ا/و/ي)
  consistently across ≥ 80 % of ajuzes → radf = that letter.

Type: متواتر if radf is present (long vowel immediately before rawiy).
"""
from __future__ import annotations
from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_DAMMA       = "ُ"
_KASRA       = "ِ"
_FATHA       = "َ"
_TANWIN_D    = "ٌ"
_TANWIN_K    = "ٍ"
_TANWIN_F    = "ً"
_MADD        = {"ا", "و", "ي"}
_PUNCT       = set(".,!?;:،؛؟…—-\"'()[]{}«»")


def _strip(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _arabic(c: str) -> bool:
    return "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ"


def _last_arabic(s: str, n: int = 3) -> list[str]:
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


def _base(tok: str) -> str:
    for c in tok:
        if _arabic(c):
            return norm_char(c)
    return ""


def _harakah_of(tok: str) -> str | None:
    if _DAMMA in tok or _TANWIN_D in tok:
        return "madmouma"
    if _KASRA in tok or _TANWIN_K in tok:
        return "maksura"
    if _FATHA in tok or _TANWIN_F in tok:
        return "maftouha"
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

    # ≥ 80 % of ajuzes end in ه
    haa_count = sum(1 for a in ajuzes if _last_arabic(a, 1) == ["ه"])
    if haa_count / len(ajuzes) < 0.80:
        return None

    # Harakah from tashkeel (majority vote)
    harakahs = []
    for t in ajuzes_t:
        if not t:
            continue
        toks = _tokens(t)
        if not toks:
            continue
        # find last ه token
        for tok in reversed(toks):
            if _base(tok) == "ه":
                h = _harakah_of(tok)
                if h:
                    harakahs.append(h)
                break

    harakah: str | None = None
    wasl: str | None = None
    if harakahs:
        top, cnt = Counter(harakahs).most_common(1)[0]
        if cnt / len(harakahs) >= 0.50:
            harakah = top
            if harakah == "madmouma":
                wasl = "و"
            elif harakah == "maksura":
                wasl = "ي"
            elif harakah == "maftouha":
                wasl = "ا"

    # Radf: letter before ه — must be a long vowel, consistent ≥ 80 %
    radfs = []
    for a in ajuzes:
        letters = _last_arabic(a, 2)
        if len(letters) >= 2 and letters[0] == "ه":
            pre = letters[1]
            if pre in _MADD:
                radfs.append(pre)

    radf: str | None = None
    if radfs and len(radfs) / len(ajuzes) >= 0.70:
        top, cnt = Counter(radfs).most_common(1)[0]
        if cnt / len(radfs) >= 0.80:
            radf = top

    result: dict = {"rawiy": "ه"}
    if harakah:
        result["harakah"] = harakah
    if wasl:
        result["wasl"] = wasl
    if radf:
        result["radf"] = radf
        result["type"] = "mutawatir"   # radf immediately before rawiy → متواتر
    return result


RULE = Rule(
    code="r011",
    title_ar="الهاء رَوِيًّا (هاء الضمير)",
    description_ar=(
        "للقصائد التي تكون فيها الهاء (هاء الضمير أو هاء التأنيث) هي الرَّوِيّ. "
        "تشترك بقية القواعد في تجاهل الهاء بوصفها وصلًا، لذا تُعالج هذه القاعدة "
        "هذه الحالة بشكل منفصل. الشرط: ≥ 80٪ من الأعجاز تنتهي بالهاء. "
        "الحركة: من التشكيل بالأغلبية (ضمة → مضموم، كسرة → مكسور، فتحة → مفتوح). "
        "الردف: الحرف الذي قبل الهاء إن كان حرف مد ثابتًا في ≥ 80٪ من الأعجاز. "
        "النوع: متواتر إن وُجد الردف."
    ),
    match=match,
)
