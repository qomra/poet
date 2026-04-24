"""
r017 — الرَّوِيّ الغالب مع تسامح الشذوذ

Fallback for poems where ≥ 75 % of ajuzes share a rawiy but earlier strict
rules (requiring 80 %+) reject them. Common causes:
  - A single source typo inside one ajuz word.
  - A slightly-different form (e.g. diminutive or dialectal variant) in
    one line.
  - Manuscript variation that drifted one letter.

Detection:
  1. Extract rawiy candidate per ajuz using the r006-style wasl stripping
     (وا → strip, then ي/ا/ى/و/ه/ن peeled).
  2. If the dominant rawiy covers ≥ 75 % of ajuzes AND there are at least
     4 ajuzes total, proceed.
  3. Harakah is decided only from the MATCHING ajuzes' tashkeel:
       - ≥ 70 % sukun on rawiy  → muqayyada (no wasl)
       - ≥ 70 % kasra           → maksura, wasl = ي
       - ≥ 70 % damma           → madmouma, wasl = و
       - ≥ 70 % fatha           → maftouha, wasl = ا
       - otherwise              → harakah unset (we only claim the rawiy)

Deliberately runs late. Sets only rawiy (+ harakah/wasl when clear) so
downstream analysis knows a single outlier was tolerated.
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_WASL_1 = {"ي", "ا", "ى", "و", "ه", "ن"}

_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TF, _TD, _TK = "ً", "ٌ", "ٍ"
_SHADDA, _SUKUN, _DAGGER = "ّ", "ْ", "ٰ"
_FATHA_SET = {_FATHA, _TF, _DAGGER}
_DAMMA_SET = {_DAMMA, _TD}
_KASRA_SET = {_KASRA, _TK}
_ALL_VOWELS = _FATHA_SET | _DAMMA_SET | _KASRA_SET

_MIN_AJUZ = 4
_DOMINANCE = 0.75
_HARAKAH_THRESH = 0.70


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip_trailing(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _rawiy_of(plain_ajuz: str) -> str | None:
    """Strip wasl decoration, return the last remaining consonant (normed)."""
    s = _strip_trailing(plain_ajuz)
    if s.endswith("وا"):
        s = s[:-2]
    while s and s[-1] in _WASL_1:
        s = s[:-1]
    for c in reversed(s):
        if _is_arabic_letter(c):
            return norm_char(c)
    return None


def _harakah_on_last_consonant(tashkeel: str) -> str | None:
    """Return 'sukun' | 'kasra' | 'fatha' | 'damma' | None for the *last*
    Arabic letter (after stripping trailing wasl letters for a fair match
    against the rawiy)."""
    s = _strip_trailing(tashkeel)
    if not s:
        return None
    # Find the last letter that isn't a wasl candidate. We walk backwards
    # skipping diacritics + trailing wasl letters so we land on the rawiy
    # consonant and read the harakah that follows it.
    i = len(s) - 1
    while i >= 0:
        c = s[i]
        if not _is_arabic_letter(c):
            i -= 1
            continue
        if c in _WASL_1:
            # Skip wasl letter and its trailing diacritics
            i -= 1
            continue
        break
    if i < 0:
        return None
    # i is now at the rawiy consonant. Read the diacritic(s) that follow it.
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
    ajuzes = [(i, v) for i, v in enumerate(verses) if i % 2 == 1 and v and v.strip()]
    if len(ajuzes) < _MIN_AJUZ:
        return None

    # Dominant rawiy on plain text
    rawiys = [(idx, _rawiy_of(v)) for idx, v in ajuzes]
    rawiys = [(idx, r) for idx, r in rawiys if r]
    if not rawiys:
        return None
    counts = Counter(r for _, r in rawiys)
    top_rawiy, top_n = counts.most_common(1)[0]
    if top_n < _DOMINANCE * len(rawiys):
        return None

    # Harakah on the MATCHING ajuzes using tashkeel
    tashkeel_verses = poem.get("verses_tashkeel") or []
    harakat: list[str] = []
    for idx, r in rawiys:
        if r != top_rawiy:
            continue
        if idx >= len(tashkeel_verses):
            continue
        tk = tashkeel_verses[idx]
        if not tk:
            continue
        h = _harakah_on_last_consonant(tk)
        if h is not None:
            harakat.append(h)

    result: dict = {"rawiy": top_rawiy}
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
                # no wasl for sakin
    return result


RULE = Rule(
    code="r017",
    title_ar="الرَّوِيّ الغالب مع تسامح الشذوذ",
    description_ar=(
        "قاعدة احتياطية للقصائد التي يتشارك فيها ٧٥٪ من الأعجاز في روي "
        "واحد، لكنها لم تُصنَّف بالقواعد السابقة لأنها ترتكز على نسبة ٨٠٪ "
        "أو أعلى. تُعالج هذه القاعدة حالة وجود بيت أو بيتين شاذَّين بسبب "
        "خطأ إملائي في المصدر أو اختلاف في صيغة الكلمة.\n"
        "١) يُستخرج الروي لكل عجز بعد تجريد حروف الوصل (وا، ي، ا، ى، و، "
        "ه، ن).\n"
        "٢) إن بلغت نسبة الروي الغالب ٧٥٪ فأكثر من الأعجاز — ولا يقل عدد "
        "الأعجاز عن أربعة — تُضبَط قيمة الروي.\n"
        "٣) تُستقرأ الحركة من تشكيل الأعجاز المطابقة فقط (لا المُخالفة)؛ "
        "إن اتّفقت ٧٠٪ منها على حركة واحدة، يُضبَط من خلالها الوصل:\n"
        "   كسرة ⇒ ياء، ضمة ⇒ واو، فتحة ⇒ ألف، سكون ⇒ مقيدة بلا وصل.\n"
        "خلاف ذلك، يُسجَّل الروي فحسب."
    ),
    match=match,
)
