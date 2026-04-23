"""
r005 — المكسور المتدارك (tashkeel-based)

Detects two-vowel (mutadaarik) kasra qafiya from diacritized text.

Signal A — sukun at [-3]:
  Pattern: C-sukun · C-moving · rawiy-kasra
  e.g. قَبْضَةٍ → بْ(sukun) · ض(moving) · ة(kasra)

Signal B — shadda at [-2]:
  Pattern: C-shadda · rawiy-kasra
  The shadda carries an implicit sukun on its first instance.
  e.g. يّةِ → يّ(shadda) · ة(kasra)

Both signals indicate two moving units between the last two sakins → متدارك.

If ≥ 80 % of diacritized ajuzes carry kasra on the rawiy AND
≥ 80 % show Signal A or B → classify as mutadaarik.

Radf: if the pre-rawiy letter is consistent across ≥ 80 % of ajuzes
(same character), store it as radf.

Complement to r004 (المتواتر): r004 fires when the pre-rawiy is itself
a long vowel; r005 fires when it is not (a consonant bearing sukun/shadda
relative to the rawiy).
"""
from __future__ import annotations
import re

from etl.qafiya_rules import Rule, norm_char

_KASRA  = "ِ"   # ِ
_SUKUN  = "ْ"   # ْ
_SHADDA = "ّ"   # ّ
_MADD   = {"ا", "و", "ي"}
_WASL   = {"ا", "و", "ي", "ن", "ه"}
_PUNCT  = set(".,!?;:،؛؟…—-\"'()[]{}«»")


def _strip_trailing(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _tokens(s: str) -> list[str]:
    """Split diacritized string into (letter, diacritics...) clusters."""
    clusters: list[str] = []
    buf = ""
    for c in s:
        if "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ":
            if buf:
                clusters.append(buf)
            buf = c
        else:
            buf += c
    if buf:
        clusters.append(buf)
    return clusters


def _has_kasra(cluster: str) -> bool:
    return _KASRA in cluster or "ٍ" in cluster  # kasra or tanwin kasra


def _has_sukun(cluster: str) -> bool:
    return _SUKUN in cluster


def _has_shadda(cluster: str) -> bool:
    return _SHADDA in cluster


def _base(cluster: str) -> str:
    """Base Arabic letter of a cluster."""
    for c in cluster:
        if "؀" <= c <= "ۿ" and c not in "ًٌٍَُِّْٰ":
            return norm_char(c)
    return ""


def _analyze_ajuz(ajuz_plain: str, ajuz_t: str) -> dict | None:
    """
    Returns {rawiy, pre_rawiy, kasra, mutadaarik} for one ajuz pair.
    kasra: bool — the rawiy carries kasra.
    mutadaarik: bool — Signal A or B detected.
    """
    plain = _strip_trailing(ajuz_plain)
    diacs = _strip_trailing(ajuz_t)
    if not plain or not diacs:
        return None

    toks = _tokens(diacs)
    if not toks:
        return None

    # Walk from end, skip wasl tokens
    i = len(toks) - 1
    while i >= 0 and _base(toks[i]) in _WASL:
        i -= 1
    if i < 0:
        return None

    rawiy_tok = toks[i]
    rawiy = _base(rawiy_tok)
    if not rawiy or rawiy in _MADD:
        return None

    kasra = _has_kasra(rawiy_tok)

    # pre-rawiy token
    pre_tok = toks[i - 1] if i >= 1 else None
    pre_rawiy = _base(pre_tok) if pre_tok else None

    # Signal B: shadda on pre-rawiy
    sig_b = pre_tok is not None and _has_shadda(pre_tok)

    # Signal A: sukun on pre-pre-rawiy
    ppre_tok = toks[i - 2] if i >= 2 else None
    def _is_saakin(tok: str) -> bool:
        if _has_sukun(tok): return True
        b = _base(tok)
        return b in _MADD and not any(c in tok for c in "ًٌٍَُِّ")
    sig_a = ppre_tok is not None and _is_saakin(ppre_tok)

    return {
        "rawiy": rawiy,
        "pre_rawiy": pre_rawiy,
        "kasra": kasra,
        "mutadaarik": sig_a or sig_b,
    }


def match(poem: dict) -> dict | None:
    verses   = poem.get("verses") or []
    verses_t = poem.get("verses_tashkeel") or []

    ajuzes   = [_strip_trailing(v) for i, v in enumerate(verses)   if i % 2 == 1]
    ajuzes_t = [_strip_trailing(v) for i, v in enumerate(verses_t) if i % 2 == 1]

    if len(ajuzes) < 3:
        return None

    # Pair up, require tashkeel for at least half
    pairs = [(p, t) for p, t in zip(ajuzes, ajuzes_t) if p and t and t.strip()]
    if len(pairs) < len(ajuzes) * 0.5:
        return None

    analyses = [_analyze_ajuz(p, t) for p, t in pairs]
    analyses = [a for a in analyses if a]
    if not analyses:
        return None

    # Rawiy must be consistent
    rawiys = {a["rawiy"] for a in analyses}
    if len(rawiys) != 1:
        return None
    rawiy = next(iter(rawiys))

    n = len(analyses)

    # 80%+ kasra on rawiy
    if sum(a["kasra"] for a in analyses) / n < 0.80:
        return None

    # 80%+ mutadaarik signal
    if sum(a["mutadaarik"] for a in analyses) / n < 0.80:
        return None

    # Radf: pre-rawiy consistent and long vowel
    pre_chars = [a["pre_rawiy"] for a in analyses if a["pre_rawiy"]]
    radf: str | None = None
    if pre_chars:
        uniq = set(pre_chars)
        if len(uniq) == 1:
            c = next(iter(uniq))
            if c in _MADD and sum(pre_chars.count(c) for _ in [1]) / n >= 0.80:
                radf = c

    result: dict = {"rawiy": rawiy, "harakah": "maksura", "wasl": "ي", "type": "mutadaarik"}
    if radf:
        result["radf"] = radf
    return result


RULE = Rule(
    code="r005",
    title_ar="المكسور المتدارك (من التشكيل)",
    description_ar=(
        "إذا أظهر التشكيل كسرةً على الرَّوِيّ في ≥ 80٪ من الأعجاز، وكان "
        "الحرف الذي قبل قبل الأخير ساكنًا (إشارة أ) أو كان الحرف الذي قبل "
        "الأخير مشدَّدًا (إشارة ب) في ≥ 80٪ من الأعجاز، فإن القافية مكسورة "
        "والوصل ياء والنوع متدارك. الردف: إن كان الحرف قبل الأخير حرفَ مدٍّ "
        "متكررًا في ≥ 80٪ من الأعجاز فهو الردف. تكمّل هذه القاعدة r004 "
        "(المتواتر): r004 تعمل حين يكون ما قبل الرَّوِيّ حرفَ مدٍّ، بينما "
        "r005 تعمل حين يكون حرفًا صامتًا بسكون أو تشديد."
    ),
    match=match,
)
