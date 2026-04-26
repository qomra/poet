"""
a003 — تكميل الحركة والوصل من التشكيل

For poems that have a rawiy but no harakah / wasl, infer them from the
tashkeel of ajuzes:

  1. If qafiya_harakah is NULL: read the harakah on the rawiy across
     ajuzes; if ≥ 70 % agree on a single one, fill it.
  2. If qafiya_wasl is NULL but harakah is now (or already) set:
       kasra   → ي
       damma   → و
       fatha   → ا
       muqayyada → leave wasl NULL (no wasl in sakin)
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import is_real_verse, norm_char
from etl.qafiya_augment import Augment

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TF, _TD, _TK = "ً", "ٌ", "ٍ"
_FATHA_SET = {_FATHA, _TF, "ٰ"}
_DAMMA_SET = {_DAMMA, _TD}
_KASRA_SET = {_KASRA, _TK}
_SUKUN = "ْ"
_VOWELS = _FATHA_SET | _DAMMA_SET | _KASRA_SET


def _is_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip(s: str) -> str:
    s = (s or "").rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _harakah_at_letter(tashkeel: str, target_letter: str) -> str | None:
    """Find the LAST occurrence of `target_letter` (after norm_char) in
    `tashkeel` and return the harakah on it.
    """
    s = _strip(tashkeel)
    if not s:
        return None
    # Scan backwards. For each Arabic letter, normalize and compare.
    i = len(s) - 1
    while i >= 0:
        if _is_letter(s[i]) and norm_char(s[i]) == target_letter:
            # Found; read diacritics that follow it
            j = i + 1
            vowel = None
            sukun = False
            while j < len(s) and not _is_letter(s[j]):
                ch = s[j]
                if ch in _VOWELS: vowel = ch
                elif ch == _SUKUN: sukun = True
                j += 1
            if sukun and vowel is None: return "sukun"
            if vowel in _KASRA_SET: return "kasra"
            if vowel in _FATHA_SET: return "fatha"
            if vowel in _DAMMA_SET: return "damma"
            return None
        i -= 1
    return None


def fn(poem: dict) -> dict:
    out: dict = {}

    rawiy = poem.get("qafiya_rawiy")
    harakah = poem.get("qafiya_harakah")
    wasl = poem.get("qafiya_wasl")

    if rawiy is None and harakah is None and wasl is None:
        return out

    # 1. Infer harakah if missing and we have a rawiy + tashkeel
    if harakah is None and rawiy:
        verses_tk = poem.get("verses_tashkeel") or []
        ajuzes_tk = [v for i, v in enumerate(verses_tk) if i % 2 == 1 and is_real_verse(v)]
        if len(ajuzes_tk) >= 3:
            har_obs = [_harakah_at_letter(a, rawiy) for a in ajuzes_tk]
            har_obs = [h for h in har_obs if h is not None]
            if len(har_obs) >= 3:
                cc = Counter(har_obs)
                top, top_n = cc.most_common(1)[0]
                if top_n / len(har_obs) >= 0.70:
                    if top == "kasra":   harakah = "maksura";   out["harakah"] = "maksura"
                    elif top == "damma": harakah = "madmouma"; out["harakah"] = "madmouma"
                    elif top == "fatha": harakah = "maftouha"; out["harakah"] = "maftouha"
                    elif top == "sukun": harakah = "muqayyada"; out["harakah"] = "muqayyada"

    # 2. Infer wasl from harakah if wasl is missing
    if wasl is None and harakah:
        if   harakah == "maksura":  out["wasl"] = "ي"
        elif harakah == "madmouma": out["wasl"] = "و"
        elif harakah == "maftouha": out["wasl"] = "ا"
        # muqayyada → no wasl

    return out


AUGMENT = Augment(
    code="a003",
    title_ar="تكميل الحركة والوصل من التشكيل",
    description_ar=(
        "للقصائد التي حُدّد فيها الرَّوِيّ ولكن الحركة و/أو الوصل ناقصان:\n"
        "  ١) تُستقرأ حركة الرَّوِيّ من تشكيل الأعجاز، وإن اتفق ٧٠٪ منها "
        "على حركة واحدة تُكتب: كسرة ⇒ مكسورة، ضمة ⇒ مضمومة، فتحة ⇒ مفتوحة، "
        "سكون ⇒ مقيَّدة.\n"
        "  ٢) إن وُجدت الحركة بعد ذلك (أو كانت مكتوبة أصلاً) ولم يكن "
        "الوصل مكتوباً، يُشتق من الحركة: مكسورة ⇒ ياء، مضمومة ⇒ واو، "
        "مفتوحة ⇒ ألف. (المقيَّدة بلا وصل)."
    ),
    fn=fn,
)
