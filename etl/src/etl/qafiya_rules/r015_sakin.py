"""
r015 — القافية الساكنة (muqayyada)

Sakin is the hardest qafiya class to identify from tashkeel because neural
diacritization models rarely mark a word-final sukun — they default to
whatever vowel fits the grammatical context. So the signals we trust are:

    A. MAJORITY SUKUN
       If the diacritizer did choose sukun on the rawiy in ≥ 60 % of ajuzes,
       trust it: the qafiya is sakin.

    B. VOWEL SWITCHING (no single case dominates)
       If the tashkeel assigns multiple different vowels (kasra / fatha /
       damma) to the same rawiy across ajuzes, no single grammatical case
       applies to the whole rhyme — the rhyme carries no intrinsic vowel,
       i.e., it's sakin. The tashkeel is just following the local syntactic
       case of each line.

       Threshold: at least 2 of {kasra, fatha, damma} each appear on the
       rawiy in ≥ 20 % of ajuzes, AND no single vowel reaches 70 %.

For both paths, rawiy consistency is required: ≥ 80 % of ajuzes must end in
the same final consonant (plain text, hamza-normalised).

Output:
    rawiy   = the consonant
    harakah = muqayyada
    wasl    = None (sakin has no wasl)
    type    = left unset (requires deeper structural analysis per ajuz)

We intentionally under-match here — r015 runs late, after all the vowelled
rules, so it only claims poems no one else could.
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import Rule, norm_char

_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TF, _TD, _TK = "ً", "ٌ", "ٍ"  # tanwin
_SHADDA, _SUKUN = "ّ", "ْ"

_FATHA_SET = {_FATHA, _TF, "ٰ"}
_DAMMA_SET = {_DAMMA, _TD}
_KASRA_SET = {_KASRA, _TK}
_ALL_VOWELS = _FATHA_SET | _DAMMA_SET | _KASRA_SET

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_DIACS = "ًٌٍَُِّْٰ"

_MIN_AJUZ = 3
_RAWIY_CONSISTENCY = 0.80   # ≥80% of ajuzes share the final consonant
_MAJORITY_SUKUN = 0.60       # path A
_VOWEL_FLOOR = 0.20          # path B: each of ≥2 vowels must cover 20% of ajuzes
_VOWEL_CEILING = 0.70        # path B: no single vowel dominates beyond this


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip_trailing(s: str | None) -> str:
    if not s:
        return ""
    s = s.rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _final_consonant(plain_ajuz: str) -> str | None:
    """Return the last Arabic consonant character (hamza-normalized)."""
    s = _strip_trailing(plain_ajuz)
    for c in reversed(s):
        if _is_arabic_letter(c):
            return norm_char(c)
    return None


def _harakah_on_final_consonant(tashkeel_ajuz: str) -> str | None:
    """Return 'sukun' | 'kasra' | 'fatha' | 'damma' | None for the last letter."""
    s = _strip_trailing(tashkeel_ajuz)
    if not s:
        return None
    # Walk to the last Arabic letter
    i = len(s) - 1
    while i >= 0 and not _is_arabic_letter(s[i]):
        i -= 1
    if i < 0:
        return None
    # Collect diacritics immediately after the letter
    j = i + 1
    vowel = None
    sukun_seen = False
    while j < len(s):
        ch = s[j]
        if ch in _ALL_VOWELS:
            vowel = ch
        elif ch == _SUKUN:
            sukun_seen = True
        elif ch == _SHADDA:
            pass  # shadda without trailing vowel at word-end is ambiguous — ignore
        elif _is_arabic_letter(ch):
            break
        j += 1
    if sukun_seen and vowel is None:
        return "sukun"
    if vowel in _KASRA_SET:
        return "kasra"
    if vowel in _FATHA_SET:
        return "fatha"
    if vowel in _DAMMA_SET:
        return "damma"
    # No mark at all — usually means end-of-word sukun in undiacritized text,
    # but here we have tashkeel, so treat as unknown rather than assume sukun.
    return None


def match(poem: dict) -> dict | None:
    ajuzes      = [v for i, v in enumerate(poem.get("verses") or [])         if i % 2 == 1 and v]
    ajuzes_tk   = [v for i, v in enumerate(poem.get("verses_tashkeel") or []) if i % 2 == 1 and v]
    if len(ajuzes) < _MIN_AJUZ:
        return None

    # ── Rawiy must be consistent on the plain text ────────────────────────────
    rawiy_candidates = [_final_consonant(a) for a in ajuzes]
    rawiy_candidates = [r for r in rawiy_candidates if r]
    if len(rawiy_candidates) < _RAWIY_CONSISTENCY * len(ajuzes):
        return None
    rawiy_counts = Counter(rawiy_candidates)
    rawiy, top_n = rawiy_counts.most_common(1)[0]
    if top_n < _RAWIY_CONSISTENCY * len(rawiy_candidates):
        return None

    # ── Inspect tashkeel on that final consonant ─────────────────────────────
    if len(ajuzes_tk) < _MIN_AJUZ:
        return None
    harakat: list[str] = []
    for a, atk in zip(ajuzes, ajuzes_tk):
        if _final_consonant(a) != rawiy:
            continue  # inconsistent rawiy — skip
        h = _harakah_on_final_consonant(atk) if atk else None
        if h is not None:
            harakat.append(h)
    if len(harakat) < _MIN_AJUZ:
        return None

    n = len(harakat)
    c = Counter(harakat)
    n_sukun = c.get("sukun", 0)
    n_kasra = c.get("kasra", 0)
    n_fatha = c.get("fatha", 0)
    n_damma = c.get("damma", 0)

    # ── Path A: majority sukun ───────────────────────────────────────────────
    path_a = n_sukun >= _MAJORITY_SUKUN * n

    # ── Path B: vowel switching (no single case dominates) ──────────────────
    vowel_counts = [n_kasra, n_fatha, n_damma]
    vowels_with_floor = [v for v in vowel_counts if v >= _VOWEL_FLOOR * n]
    has_dominant = max(vowel_counts) / n > _VOWEL_CEILING if n else False
    path_b = len(vowels_with_floor) >= 2 and not has_dominant

    if not (path_a or path_b):
        return None

    return {
        "rawiy":   rawiy,
        "harakah": "muqayyada",
        # wasl intentionally unset (sakin has no wasl)
    }


RULE = Rule(
    code="r015",
    title_ar="القافية الساكنة (المقيدة)",
    description_ar=(
        "القافية الساكنة (المقيدة) من أصعب الأنواع للتعرف عليها من التشكيل، "
        "لأن المشكِّلات العصبية نادراً ما تضع سكوناً في نهاية الكلمة — بل "
        "تختار الحركة المناسبة للسياق الإعرابي. لذلك نثق بإشارتين اثنتين:\n"
        "١) إذا اختار المشكِّل السكون في ٦٠٪ فأكثر من أعجاز القصيدة على "
        "الرَّوِيّ، فالقافية ساكنة.\n"
        "٢) إذا تناوبت الحركات (كسرة / فتحة / ضمة) على الرَّوِيّ عبر الأعجاز "
        "دون أن تهيمن حركة واحدة، فهذا دليل أن الرَّوِيّ لا يحمل حركة "
        "داخلية — فالتشكيل يتبع سياق الإعراب المحلي في كل بيت — وهذا معنى "
        "السكون.\n"
        "يشترط أن يكون الحرف الأخير ثابتاً في ٨٠٪ من الأعجاز على الأقل. "
        "يُترك تحديد النَّوع (متواتر / متدارك…) لقواعد أخرى."
    ),
    match=match,
)
