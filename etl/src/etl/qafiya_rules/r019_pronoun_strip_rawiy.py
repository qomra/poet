"""
r019 — اقتطاع الضمير لاستخراج الرَّوِيّ

Many Arabic poems carry pronoun suffixes that vary line-by-line — قَلْبُهُ /
قَلْبُهَا / قَلْبَكُمْ — so the surface "last letter" alternates between
ه / ا / م. The actual rawiy is the consonant before the pronoun (ب in this
case). Earlier rules that strip a fixed wasl set miss this because:
    • r013/r017 don't strip ة or full pronoun forms
    • r011 only handles cases where ه IS the rawiy, not the suffix
    • r018 needs the surface last-letter to agree

This rule peels everything that could be wasl decoration:
    1. Trailing punctuation
    2. Pronoun suffix (ها / هم / هن / كم / كن / نا / ه / ك / ي)
    3. Trailing ة (feminine ending — phonologically a wasl-like decoration)
    4. Trailing wasl letters (ا / ى / و / ي / ن)

Then the last remaining consonant is the rawiy candidate. If ≥ 85 % of
ajuzes share it AND tashkeel on the rawiy agrees on a single harakah in
≥ 75 % of those ajuzes, classify with that harakah-derived wasl.

Runs late (after r000-r018). Conservative thresholds because the deep
strip is permissive.
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, ajuzes_of, norm_char

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")

# Try longer pronoun suffixes first, since most are 2-3 chars
_PRONOUN_SUFFIXES = [
    "كما", "كنّ", "هما", "هنّ", "كنا",
    "كم", "كن", "نا", "ها", "هم", "هن", "تن", "تم",
    "ه",  "ك",  "ي",
]
_WASL_LETTERS = {"ا", "ى", "و", "ي", "ن"}

_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TF, _TD, _TK = "ً", "ٌ", "ٍ"
_FATHA_SET = {_FATHA, _TF, "ٰ"}
_DAMMA_SET = {_DAMMA, _TD}
_KASRA_SET = {_KASRA, _TK}
_ALL_VOWELS = _FATHA_SET | _DAMMA_SET | _KASRA_SET
_SUKUN = "ْ"
_DIACRITICS = "ًٌٍَُِّْٰ"

_MIN_AJUZ = 3
_RAWIY_THRESH = 0.85
_HARAKAH_THRESH = 0.75


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip_punct(s: str) -> str:
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _deep_strip_to_rawiy(plain_ajuz: str) -> tuple[str | None, int]:
    """Return (rawiy_letter, depth_stripped) where depth_stripped indicates
    how many trailing chars were peeled. depth helps locate the rawiy in
    the diacritized version too."""
    s = _strip_punct(plain_ajuz or "")
    depth = 0
    # 1. Pronoun suffix
    for suf in _PRONOUN_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[:-len(suf)]
            depth += len(suf)
            break
    # 2. ة
    if s.endswith("ة"):
        s = s[:-1]
        depth += 1
    # 3. Trailing wasl letters
    while s and s[-1] in _WASL_LETTERS:
        s = s[:-1]
        depth += 1
    # 4. Last remaining consonant
    for c in reversed(s):
        if _is_arabic_letter(c):
            return norm_char(c), depth
    return None, depth


def _harakah_at_rawiy(tashkeel: str | None, rawiy_letter: str) -> str | None:
    """Walk backwards through the tashkeel, applying the same deep-strip,
    and return the harakah marked on the rawiy consonant.
    """
    if not tashkeel:
        return None
    s = _strip_punct(tashkeel)
    if not s:
        return None
    # Get the plain version of this string (no diacritics) so we can run the
    # same deep-strip and compare what the rawiy position should be.
    plain = "".join(c for c in s if c not in _DIACRITICS)
    rawiy_norm, _ = _deep_strip_to_rawiy(plain)
    if rawiy_norm is None or rawiy_norm != rawiy_letter:
        return None

    # Walk backwards: find the rawiy letter (after deep-strip equivalent
    # peeling), then read its harakah.
    # Approach: walk through `plain` backwards, find first letter from the
    # END that, after norm_char, equals rawiy_letter — but skip past any
    # pronoun-suffix letters, ة, and wasl letters.
    # Simpler: replicate the deep strip on the diacritized string and find
    # the rawiy letter in it.
    work = s
    # 1. Pronoun suffix — strip from work
    plain_work = "".join(c for c in work if c not in _DIACRITICS)
    suffix_chars = 0
    for suf in _PRONOUN_SUFFIXES:
        if plain_work.endswith(suf) and len(plain_work) > len(suf) + 1:
            suffix_chars = len(suf)
            break
    # Walk back through diacritized string, removing `suffix_chars` Arabic letters
    if suffix_chars:
        i = len(work) - 1
        removed = 0
        while i >= 0 and removed < suffix_chars:
            if _is_arabic_letter(work[i]):
                removed += 1
            i -= 1
        work = work[: i + 1]
    # 2. ة
    plain_work = "".join(c for c in work if c not in _DIACRITICS)
    if plain_work.endswith("ة"):
        i = len(work) - 1
        while i >= 0 and not _is_arabic_letter(work[i]):
            i -= 1
        work = work[:i]
    # 3. Strip wasl letters at end
    while True:
        plain_work = "".join(c for c in work if c not in _DIACRITICS)
        if not plain_work or plain_work[-1] not in _WASL_LETTERS:
            break
        i = len(work) - 1
        while i >= 0 and not _is_arabic_letter(work[i]):
            i -= 1
        work = work[:i]
    # Now the last Arabic letter in `work` is the rawiy. Read its harakah.
    i = len(work) - 1
    while i >= 0 and not _is_arabic_letter(work[i]):
        i -= 1
    if i < 0:
        return None
    j = i + 1
    vowel = None
    sukun = False
    while j < len(work) and not _is_arabic_letter(work[j]):
        ch = work[j]
        if ch in _ALL_VOWELS:
            vowel = ch
        elif ch == _SUKUN:
            sukun = True
        j += 1
    if vowel in _KASRA_SET:
        return "kasra"
    if vowel in _DAMMA_SET:
        return "damma"
    if vowel in _FATHA_SET:
        return "fatha"
    if sukun:
        return "sukun"
    return None


def match(poem: dict) -> dict | None:
    plain, tk = ajuzes_of(poem)
    if len(plain) < _MIN_AJUZ:
        return None

    rawiys: list[str] = []
    for a in plain:
        r, _ = _deep_strip_to_rawiy(a)
        if r:
            rawiys.append(r)
    if len(rawiys) < _MIN_AJUZ:
        return None

    counts = Counter(rawiys)
    top, top_n = counts.most_common(1)[0]
    if top_n < _RAWIY_THRESH * len(rawiys):
        return None

    # Harakah from tashkeel
    harakat: list[str] = []
    for a, atk in zip(plain, tk):
        r, _ = _deep_strip_to_rawiy(a)
        if r != top:
            continue
        h = _harakah_at_rawiy(atk, top)
        if h is not None:
            harakat.append(h)

    result: dict = {"rawiy": top}
    if harakat:
        hc = Counter(harakat)
        top_h, top_hn = hc.most_common(1)[0]
        if top_hn >= _HARAKAH_THRESH * len(harakat):
            if top_h == "kasra":
                result["harakah"], result["wasl"] = "maksura", "ي"
            elif top_h == "damma":
                result["harakah"], result["wasl"] = "madmouma", "و"
            elif top_h == "fatha":
                result["harakah"], result["wasl"] = "maftouha", "ا"
            elif top_h == "sukun":
                result["harakah"] = "muqayyada"
    return result


RULE = Rule(
    code="r019",
    title_ar="اقتطاع الضمير لاستخراج الرَّوِيّ",
    description_ar=(
        "كثيرٌ من القصائد تتغيّر فيها لاحقات الضمائر بين بيتٍ وآخر "
        "(قَلْبُهُ / قَلْبُهَا / قَلْبَكُمْ)، فيتبدّل الحرفُ الأخير سطحياً (ه/ا/م) "
        "بينما الرَّوِيّ الحقيقي هو الحرف السابق للضمير (ب في المثال). تُقتطع "
        "هذه القاعدة:\n"
        "  ١) لاحقات الضمائر (ها، هم، هن، كم، كن، نا، ه، ك، ي ...)\n"
        "  ٢) التاء المربوطة (ة)\n"
        "  ٣) حروف الوصل المعتادة (ا/ى/و/ي/ن)\n"
        "ثم يُعدُّ الحرف المتبقّي الأخير رويًّا. تنطبق القاعدة إذا تطابق هذا "
        "الرَّوِيّ في ٨٥٪ من الأعجاز، وتُحدَّد الحركة من التشكيل عند موضع "
        "الرَّوِيّ بعد إجراء نفس الاقتطاع: كسرة ⇒ ياء، ضمة ⇒ واو، فتحة ⇒ "
        "ألف، سكون ⇒ مقيَّدة بلا وصل."
    ),
    match=match,
)
