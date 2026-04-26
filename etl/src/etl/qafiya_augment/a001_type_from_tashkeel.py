"""
a001 — تحديد نوع القافية من التشكيل

Augmentation rule. For poems that already have a rule_id but where
qafiya_type is NULL, infer the type by counting متحركات between the last
two ساكِنَيْن on each ajuz's tashkeel:

    0  متحركات → مترادف
    1            → متواتر
    2            → متدارك
    3            → متراكب
    4+           → متكاوس

If ≥ 70 % of ajuzes (with parsable tashkeel) agree on a single count,
emit that type.

Implicit wasl: if an ajuz ends with a vowelled letter (no sukun), append
an implicit saakin matching the final harakah:
    kasra → ي,  damma → و,  fatha → ا.

Shadda doubles the letter into (saakin)(متحرك).
"""
from __future__ import annotations

from collections import Counter

from etl.qafiya_rules import is_real_verse
from etl.qafiya_augment import Augment

_PUNCT = set(".,!?;:،؛؟…—-\"'()[]{}«»")
_FATHA, _DAMMA, _KASRA = "َ", "ُ", "ِ"
_TF, _TD, _TK = "ً", "ٌ", "ٍ"
_SHADDA, _SUKUN = "ّ", "ْ"
_DAGGER = "ٰ"

_FATHA_SET = {_FATHA, _TF, _DAGGER}
_DAMMA_SET = {_DAMMA, _TD}
_KASRA_SET = {_KASRA, _TK}
_VOWELS = _FATHA_SET | _DAMMA_SET | _KASRA_SET

_TYPE_BY_COUNT = {
    0: "mutaradif",
    1: "mutawatir",
    2: "mutadaarik",
    3: "mutarakib",
    4: "mutakaasis",
}


def _is_letter(c: str) -> bool:
    return "ء" <= c <= "ي" or c in {"ٱ", "ی"}


def _strip(s: str) -> str:
    s = (s or "").rstrip()
    while s and s[-1] in _PUNCT:
        s = s[:-1].rstrip()
    return s


def _parse(s: str) -> list[tuple[str, str]]:
    """Return list of (letter, state) where state ∈ {'mu','sa'}."""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        c = s[i]
        if not _is_letter(c):
            i += 1
            continue
        j = i + 1
        shadda = False
        vowel = None
        sukun = False
        while j < len(s) and not _is_letter(s[j]):
            ch = s[j]
            if ch == _SHADDA: shadda = True
            elif ch in _VOWELS: vowel = ch
            elif ch == _SUKUN: sukun = True
            j += 1
        if shadda:
            out.append((c, "sa"))
            out.append((c, "mu" if vowel else "sa"))
        else:
            if vowel: out.append((c, "mu"))
            elif sukun: out.append((c, "sa"))
            else: out.append((c, "sa"))   # word-final unmarked = saakin
        i = j
    return out


def _append_implicit_wasl(letters: list[tuple[str, str]], tashkeel_tail: str) -> None:
    """If the last letter is متحرك, append an implicit saakin based on its
    final harakah."""
    if not letters or letters[-1][1] != "mu":
        return
    # Read the final vowel from the tail to know which letter to append
    s = tashkeel_tail
    final_v = None
    for c in reversed(s):
        if c in _VOWELS:
            final_v = c
            break
        if _is_letter(c):
            break
    if final_v in _KASRA_SET: letters.append(("ي", "sa"))
    elif final_v in _DAMMA_SET: letters.append(("و", "sa"))
    elif final_v in _FATHA_SET: letters.append(("ا", "sa"))


def _count_muharrikat(tashkeel: str) -> int | None:
    s = _strip(tashkeel)
    if len(s) < 3:
        return None
    letters = _parse(s)
    if not letters:
        return None
    _append_implicit_wasl(letters, s)
    saakin_idx = [i for i, (_, st) in enumerate(letters) if st == "sa"]
    if len(saakin_idx) < 2:
        return None
    last_sa, prev_sa = saakin_idx[-1], saakin_idx[-2]
    return sum(1 for i in range(prev_sa + 1, last_sa) if letters[i][1] == "mu")


def fn(poem: dict) -> dict:
    if poem.get("qafiya_type"):
        return {}
    verses_tk = poem.get("verses_tashkeel") or []
    # Collect tashkeel ajuzes — same parity as runner already normalised
    ajuzes_tk = [v for i, v in enumerate(verses_tk) if i % 2 == 1 and is_real_verse(v)]
    if len(ajuzes_tk) < 3:
        return {}
    counts = [_count_muharrikat(a) for a in ajuzes_tk]
    counts = [c for c in counts if c is not None]
    if len(counts) < 3:
        return {}
    cc = Counter(counts)
    top, top_n = cc.most_common(1)[0]
    if top_n / len(counts) < 0.70:
        return {}
    if top >= 4:
        return {"type": "mutakaasis"}
    if top in _TYPE_BY_COUNT:
        return {"type": _TYPE_BY_COUNT[top]}
    return {}


AUGMENT = Augment(
    code="a001",
    title_ar="تحديد نوع القافية من التشكيل",
    description_ar=(
        "قاعدة تكميلية: تستنتج نوع القافية (متواتر / متدارك / متراكب / "
        "متكاوس / مترادف) من عدد المتحركات بين آخر ساكنَيْن في تشكيل كل عجز.\n"
        "  ٠ متحركات → مترادف،  ١ → متواتر،  ٢ → متدارك،  ٣ → متراكب،  ٤+ → متكاوس\n"
        "تعتمد على تشكيل ٧٠٪ فأكثر من الأعجاز على عددٍ واحد لتقرّر النوع. "
        "تضيف ساكنًا ضمنيًا إذا انتهى العجز بحرف متحرك (كسرة ⇒ ياء، ضمة ⇒ "
        "واو، فتحة ⇒ ألف). تُكتب القيمة فقط إن كان qafiya_type فارغاً، "
        "ولا تتعدى على الحقول المملوءة."
    ),
    fn=fn,
)
