"""
r018 — الرَّوِيّ هو الحرف الأخير ذاته (دون تجريد الوصل)

Rules r001 … r017 all strip ن / ه / و / ي / ا as "wasl candidates" before
identifying the rawiy. That works when the wasl is genuinely an extension,
but it FAILS when the wasl letter itself is the rawiy — e.g.:
    سُكُونٌ / يَكُونُ / يُبَيِّنُ
all end in ن with damma. Stripping ن as wasl reveals inconsistent
"rawiys" (ك, ك, ب) and every previous rule rejects.

This rule treats the **raw final consonant** as the rawiy candidate. It
only fires when:
    1. ≥ 85 % of ajuzes share the same final consonant.
    2. The tashkeel on that consonant agrees on a single harakah in
       ≥ 85 % of those ajuzes.
    3. We did not already classify the poem (runs late, after r000-r017).

Outputs harakah-derived wasl:
    kasra → ي,  damma → و,  fatha → ا,  sukun → no wasl (muqayyada).
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")

_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TF, _TD, _TK = "ً", "ٌ", "ٍ"
_SHADDA, _SUKUN, _DAGGER = "ّ", "ْ", "ٰ"
_FATHA_SET = {_FATHA, _TF, _DAGGER}
_DAMMA_SET = {_DAMMA, _TD}
_KASRA_SET = {_KASRA, _TK}
_ALL_VOWELS = _FATHA_SET | _DAMMA_SET | _KASRA_SET

_MIN_AJUZ = 3
_LETTER_THRESH = 0.85
_HARAKAH_THRESH = 0.85


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip_trailing(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _final_letter(plain: str) -> str | None:
    s = _strip_trailing(plain)
    for c in reversed(s):
        if _is_arabic_letter(c):
            return norm_char(c)
    return None


def _harakah_on_final(tashkeel: str) -> str | None:
    s = _strip_trailing(tashkeel)
    if not s:
        return None
    i = len(s) - 1
    while i >= 0 and not _is_arabic_letter(s[i]):
        i -= 1
    if i < 0:
        return None
    j = i + 1
    vowel = None
    sukun = False
    while j < len(s) and not _is_arabic_letter(s[j]):
        ch = s[j]
        if ch in _ALL_VOWELS:
            vowel = ch
        elif ch == _SUKUN:
            sukun = True
        j += 1
    if sukun and vowel is None:
        return "sukun"
    if vowel in _KASRA_SET:
        return "kasra"
    if vowel in _FATHA_SET:
        return "fatha"
    if vowel in _DAMMA_SET:
        return "damma"
    return None


def match(poem: dict) -> dict | None:
    verses = poem.get("verses") or []
    if len(verses) % 2 == 1:
        return None
    pairs = [(i, v) for i, v in enumerate(verses) if i % 2 == 1 and v and v.strip()]
    if len(pairs) < _MIN_AJUZ:
        return None

    last_letters = [(idx, _final_letter(v)) for idx, v in pairs]
    last_letters = [(idx, l) for idx, l in last_letters if l]
    if len(last_letters) < _MIN_AJUZ:
        return None

    counts = Counter(l for _, l in last_letters)
    top, top_n = counts.most_common(1)[0]
    if top_n < _LETTER_THRESH * len(last_letters):
        return None

    tashkeel = poem.get("verses_tashkeel") or []
    harakat: list[str] = []
    for idx, l in last_letters:
        if l != top:
            continue
        if idx >= len(tashkeel):
            continue
        tk = tashkeel[idx]
        if not tk:
            continue
        h = _harakah_on_final(tk)
        if h is not None:
            harakat.append(h)
    if len(harakat) < _MIN_AJUZ:
        return None

    hc = Counter(harakat)
    top_h, top_hn = hc.most_common(1)[0]
    if top_hn < _HARAKAH_THRESH * len(harakat):
        return None

    if top_h == "kasra":
        return {"rawiy": top, "harakah": "maksura", "wasl": "ي"}
    if top_h == "damma":
        return {"rawiy": top, "harakah": "madmouma", "wasl": "و"}
    if top_h == "fatha":
        return {"rawiy": top, "harakah": "maftouha", "wasl": "ا"}
    if top_h == "sukun":
        return {"rawiy": top, "harakah": "muqayyada"}
    return None


RULE = Rule(
    code="r018",
    title_ar="الرَّوِيّ هو الحرف الأخير ذاته (دون تجريد الوصل)",
    description_ar=(
        "تكتشف القصائد التي يكون فيها الحرف الأخير ذاته هو الرَّوِيّ، أي إن "
        "حرف الوصل المعتاد (ن/ه/و/ي/ا) لا يُجرَّد لأنه نفسه يحمل القافية. "
        "كل القواعد السابقة تتعامل مع هذه الحروف على أنها وصل وتجرِّدها، "
        "فإذا وُجد روي ثابت فعلاً (مثل: سُكُونٌ / يَكُونُ / يُبَيِّنُ — كلها تنتهي "
        "بـ ن مضمومة) تُغفله السابقات.\n"
        "تُطلَق هذه القاعدة فقط حين يتطابق الحرف الأخير في ٨٥٪ من الأعجاز "
        "وتتطابق الحركة المُشكَّلة عليه في ٨٥٪ منها أيضًا. تُحدَّد الحركة "
        "والوصل من التشكيل: كسرة ⇒ ياء، ضمة ⇒ واو، فتحة ⇒ ألف، سكون ⇒ "
        "مقيَّدة بلا وصل."
    ),
    match=match,
)
