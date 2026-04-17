"""
Normalization maps: raw strings → canonical enum values.

These handle the reality of Arabic poetry datasets:
- Same meter spelled 10 different ways across sources
- Themes as free text, comma-joined, mixed Arabic/English
- Poet names with laqab vs kunya vs full name
- Era labels inconsistent across datasets
"""

from __future__ import annotations

import re
import unicodedata

from schema import Era, LanguageType, Meter, Theme, Tradition

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------


def clean_arabic(text: str) -> str:
    """Strip diacritics, tatweel, normalize alef/hamza variants."""
    # Remove tashkeel (diacritics)
    tashkeel = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
    text = tashkeel.sub("", text)
    # Remove tatweel
    text = text.replace("\u0640", "")
    # Normalize alef variants → bare alef
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Normalize hamza variants
    text = re.sub(r"[ؤئ]", "ء", text)
    # Normalize ta marbuta → ha
    text = text.replace("ة", "ه")
    return text.strip()


def normalize_ws(text: str) -> str:
    """Collapse whitespace, strip."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Meter normalization
# ---------------------------------------------------------------------------

_METER_MAP: dict[str, Meter] = {
    # Taweel
    "طويل": Meter.TAWEEL,
    "البحر الطويل": Meter.TAWEEL,
    "بحر الطويل": Meter.TAWEEL,
    "taweel": Meter.TAWEEL,
    "tawil": Meter.TAWEEL,
    # Baseet
    "بسيط": Meter.BASEET,
    "بحر البسيط": Meter.BASEET,
    "baseet": Meter.BASEET,
    "basit": Meter.BASEET,
    # Kamel
    "كامل": Meter.KAMEL,
    "بحر الكامل": Meter.KAMEL,
    "kamel": Meter.KAMEL,
    # Wafer
    "وافر": Meter.WAFER,
    "بحر الوافر": Meter.WAFER,
    "wafir": Meter.WAFER,
    "wafer": Meter.WAFER,
    # Khafeef
    "خفيف": Meter.KHAFEEF,
    "بحر الخفيف": Meter.KHAFEEF,
    "khafeef": Meter.KHAFEEF,
    "khafif": Meter.KHAFEEF,
    # Ramal
    "رمل": Meter.RAMAL,
    "بحر الرمل": Meter.RAMAL,
    "ramal": Meter.RAMAL,
    # Rajaz
    "رجز": Meter.RAJAZ,
    "بحر الرجز": Meter.RAJAZ,
    "rajaz": Meter.RAJAZ,
    # Hazaj
    "هزج": Meter.HAZAJ,
    "بحر الهزج": Meter.HAZAJ,
    "hazaj": Meter.HAZAJ,
    # Mutakareb
    "متقارب": Meter.MUTAKAREB,
    "بحر المتقارب": Meter.MUTAKAREB,
    "mutakareb": Meter.MUTAKAREB,
    "mutaqarib": Meter.MUTAKAREB,
    # Mutadarak
    "متدارك": Meter.MUTADARAK,
    "محدث": Meter.MUTADARAK,
    "بحر المتدارك": Meter.MUTADARAK,
    "بحر المحدث": Meter.MUTADARAK,
    "mutadarak": Meter.MUTADARAK,
    # Saree
    "سريع": Meter.SAREE,
    "بحر السريع": Meter.SAREE,
    "sari": Meter.SAREE,
    "saree": Meter.SAREE,
    # Munsareh
    "منسرح": Meter.MUNSAREH,
    "بحر المنسرح": Meter.MUNSAREH,
    "munsareh": Meter.MUNSAREH,
    # Mujtath
    "مجتث": Meter.MUJTATH,
    "بحر المجتث": Meter.MUJTATH,
    "mujtath": Meter.MUJTATH,
    # Mudhare
    "مضارع": Meter.MUDHARE,
    "بحر المضارع": Meter.MUDHARE,
    "mudhare": Meter.MUDHARE,
    # Muqtadheb
    "مقتضب": Meter.MUQTADHEB,
    "بحر المقتضب": Meter.MUQTADHEB,
    "muqtadheb": Meter.MUQTADHEB,
    # Madeed
    "مديد": Meter.MADEED,
    "بحر المديد": Meter.MADEED,
    "madeed": Meter.MADEED,
    # Sub-meters (map to parent or sub)
    "مجزوء الكامل": Meter.MAJZOO_KAMEL,
    "مجزوء كامل": Meter.MAJZOO_KAMEL,
    "مجزوء الرجز": Meter.MAJZOO_RAJAZ,
    "مجزوء الرمل": Meter.MAJZOO_RAMAL,
    "مجزوء البسيط": Meter.MAJZOO_BASEET,
    "مخلع البسيط": Meter.MAKHALLA_BASEET,
    # Nabati
    "مسحوب": Meter.MASHHUB,
    "المسحوب": Meter.MASHHUB,
    "mashhub": Meter.MASHHUB,
    "هلالي": Meter.HILALI,
    "الهلالي": Meter.HILALI,
    "هجيني": Meter.HIJINI,
    "سامري": Meter.SAMIRI,
    "زهيري": Meter.ZUHAYRI,
    # Modern
    "تفعيلة": Meter.TAFILA,
    "شعر التفعيله": Meter.TAFILA,
    "شعر حر": Meter.TAFILA,
    "نثر": Meter.NATHR,
    "قصيدة نثر": Meter.NATHR,
}


_METER_MAP_CLEAN: dict[str, Meter] = {
    clean_arabic(k): v for k, v in _METER_MAP.items()
}


def normalize_meter(raw: str | None) -> Meter:
    if not raw:
        return Meter.UNKNOWN
    cleaned = clean_arabic(normalize_ws(raw)).lower()
    # Reject rhyme strings that snuck into the meter field
    if re.search(r"قافيه?", cleaned):
        return Meter.UNKNOWN
    if cleaned in _METER_MAP_CLEAN:
        return _METER_MAP_CLEAN[cleaned]
    # Strip بحر prefix and retry
    stripped = re.sub(r"^بحر\s*", "", cleaned).strip()
    return _METER_MAP_CLEAN.get(stripped, Meter.UNKNOWN)


# ---------------------------------------------------------------------------
# Rhyme letter extraction
# ---------------------------------------------------------------------------

_ARABIC_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآةى")

# ashaar stores rhyme as full letter name — map to the character
_LETTER_NAME_MAP: dict[str, str] = {
    "الف": "ا", "الهمزه": "ء", "الهمزة": "ء",
    "الباء": "ب", "التاء": "ت", "الثاء": "ث",
    "الجيم": "ج", "الحاء": "ح", "الخاء": "خ",
    "الدال": "د", "الذال": "ذ", "الراء": "ر",
    "الزاي": "ز", "السين": "س", "الشين": "ش",
    "الصاد": "ص", "الضاد": "ض", "الطاء": "ط",
    "الظاء": "ظ", "العين": "ع", "الغين": "غ",
    "الفاء": "ف", "القاف": "ق", "الكاف": "ك",
    "اللام": "ل", "الميم": "م", "النون": "ن",
    "الهاء": "ه", "الواو": "و", "الياء": "ي",
    "التاء المربوطه": "ة", "التاء المربوطة": "ة",
    "الالف المقصوره": "ى", "الالف المقصورة": "ى",
}
_LETTER_NAME_MAP_CLEAN = {clean_arabic(k): v for k, v in _LETTER_NAME_MAP.items()}


def extract_rhyme_letter(raw: str | None) -> str | None:
    """
    Extract the rhyme consonant. Handles:
      'الياء'            → 'ي'   (ashaar format: full letter name)
      'قافية الهاء (ه)'  → 'ه'   (DiwanAI format)
      'ه'                → 'ه'   (single character)
      '.'                → None  (missing marker in ashaar)
    """
    if not raw or raw.strip() in (".", "-", ""):
        return None
    raw = normalize_ws(raw)

    # Full letter name (ashaar format) — try first
    cleaned = clean_arabic(raw)
    if cleaned in _LETTER_NAME_MAP_CLEAN:
        return _LETTER_NAME_MAP_CLEAN[cleaned]

    # Letter in parentheses: (ه)
    m = re.search(r"\(([ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآةى])\)", raw)
    if m:
        return m.group(1)

    # Single Arabic letter
    words = raw.split()
    for word in reversed(words):
        w = clean_arabic(word)
        if w in _ARABIC_LETTERS:
            return w
    return None


# ---------------------------------------------------------------------------
# Theme normalization
# ---------------------------------------------------------------------------

_THEME_MAP: dict[str, Theme] = {
    # ashaar format: "قصيدة X"
    "قصيده مدح": Theme.MADIH,
    "قصيده غزل": Theme.GHAZAL,
    "قصيده رومنسيه": Theme.GHAZAL,
    "قصيده شوق": Theme.GHAZAL,
    "قصيده فراق": Theme.GHAZAL,
    "قصيده هجاء": Theme.HIJA,
    "قصيده ذم": Theme.HIJA,
    "قصيده رثاء": Theme.RITHA,
    "قصيده حزينه": Theme.RITHA,
    "قصيده دينيه": Theme.SUFI,
    "قصيده اعتذار": Theme.ITIDHAR,
    "قصيده عتاب": Theme.ITIDHAR,
    "قصيده وطنيه": Theme.FAKHR,
    "قصيده سياسيه": Theme.HAMASA,
    # Short forms
    "غزل": Theme.GHAZAL,
    "الغزل": Theme.GHAZAL,
    "مديح": Theme.MADIH,
    "المدح": Theme.MADIH,
    "مدح": Theme.MADIH,
    "رثاء": Theme.RITHA,
    "الرثاء": Theme.RITHA,
    "هجاء": Theme.HIJA,
    "الهجاء": Theme.HIJA,
    "فخر": Theme.FAKHR,
    "الفخر": Theme.FAKHR,
    "حكمة": Theme.HIKMA,
    "الحكمة": Theme.HIKMA,
    "حماسة": Theme.HAMASA,
    "وصف": Theme.WASF,
    "الوصف": Theme.WASF,
    "وصف الطبيعة": Theme.WASF_TABI,
    "زهد": Theme.ZUHD,
    "الزهد": Theme.ZUHD,
    "خمريات": Theme.KHAMRIYYAT,
    "اعتذار": Theme.ITIDHAR,
    "صوفي": Theme.SUFI,
    "تصوف": Theme.SUFI,
    "مديح نبوي": Theme.MADIH_NABAWI,
    "madih": Theme.MADIH,
    "ghazal": Theme.GHAZAL,
    "ritha": Theme.RITHA,
    "hija": Theme.HIJA,
}


# Pre-clean all theme map keys once so lookups are consistent
_THEME_MAP_CLEAN: dict[str, Theme] = {
    clean_arabic(k): v for k, v in _THEME_MAP.items()
}


def normalize_theme(raw: str | None) -> Theme:
    if not raw:
        return Theme.UNKNOWN
    cleaned = clean_arabic(normalize_ws(raw)).lower()
    if cleaned in _THEME_MAP_CLEAN:
        return _THEME_MAP_CLEAN[cleaned]
    # Theme fields combine themes with و/أ/، — split on word boundary conjunctions
    parts = re.split(r"\s+[وأ]\s+|[،,]", cleaned)
    first = parts[0].strip()
    if first in _THEME_MAP_CLEAN:
        return _THEME_MAP_CLEAN[first]
    # Prefix match: "مدح النبي" starts with key "مدح" → MADIH
    # Only one direction: cleaned starts with a known key (not vice versa)
    for key, val in _THEME_MAP_CLEAN.items():
        if len(key) >= 4 and cleaned.startswith(key):
            return val
    return Theme.UNKNOWN


# ---------------------------------------------------------------------------
# Era normalization
# ---------------------------------------------------------------------------

_ERA_MAP: dict[str, Era] = {
    # Ashaar dataset values ("العصر X" prefix)
    "العصر الحديث": Era.MODERN,
    "العصر العباسي": Era.ABBASID,
    "العصر المملوكي": Era.MAMLUK,
    "العصر العثماني": Era.OTTOMAN,
    "المغرب والاندلس": Era.ANDALUSIAN,
    "العصر الفاطمي": Era.ABBASID,    # Fatimid overlaps Abbasid period
    "العصر الاندلسي": Era.ANDALUSIAN,
    "العصر الاموي": Era.UMAYYAD,
    "العصر الايوبي": Era.AYYUBID,
    "المخضرمين": Era.ISLAMIC,
    "العصر الجاهلي": Era.PRE_ISLAMIC,
    "قبل الاسلام": Era.PRE_ISLAMIC,
    "عصر بين الدولتين": Era.UMAYYAD,
    "العصر الاسلامي": Era.ISLAMIC,
    # Short forms
    "جاهلي": Era.PRE_ISLAMIC,
    "عصر الجاهلية": Era.PRE_ISLAMIC,
    "pre-islamic": Era.PRE_ISLAMIC,
    "pre_islamic": Era.PRE_ISLAMIC,
    "اسلامي": Era.ISLAMIC,
    "إسلامي": Era.ISLAMIC,
    "صدر الاسلام": Era.ISLAMIC,
    "اموي": Era.UMAYYAD,
    "أموي": Era.UMAYYAD,
    "umayyad": Era.UMAYYAD,
    "عباسي": Era.ABBASID,
    "abbasid": Era.ABBASID,
    "اندلسي": Era.ANDALUSIAN,
    "أندلسي": Era.ANDALUSIAN,
    "andalusian": Era.ANDALUSIAN,
    "ايوبي": Era.AYYUBID,
    "أيوبي": Era.AYYUBID,
    "ayyubid": Era.AYYUBID,
    "مملوكي": Era.MAMLUK,
    "mamluk": Era.MAMLUK,
    "عثماني": Era.OTTOMAN,
    "ottoman": Era.OTTOMAN,
    "حديث": Era.MODERN,
    "modern": Era.MODERN,
    "معاصر": Era.CONTEMPORARY,
    "contemporary": Era.CONTEMPORARY,
}

_ERA_MAP_CLEAN: dict[str, Era] = {
    clean_arabic(k): v for k, v in _ERA_MAP.items()
}


def normalize_era(raw: str | None) -> Era:
    if not raw:
        return Era.UNKNOWN
    cleaned = clean_arabic(normalize_ws(raw)).lower()
    if cleaned in _ERA_MAP_CLEAN:
        return _ERA_MAP_CLEAN[cleaned]
    # Strip "العصر" prefix and retry
    stripped = re.sub(r"^العصر\s*", "", cleaned).strip()
    return _ERA_MAP_CLEAN.get(stripped, Era.UNKNOWN)


# ---------------------------------------------------------------------------
# Tradition / language type
# ---------------------------------------------------------------------------


def normalize_tradition(language_type_raw: str | None, meter: Meter) -> Tradition:
    """Infer tradition from raw language_type field and meter."""
    if language_type_raw:
        lt = clean_arabic(normalize_ws(language_type_raw)).lower()
        if any(k in lt for k in ("عامي", "عامية", "نبطي", "شعبي")):
            return Tradition.NABATI
        if "زجل" in lt:
            return Tradition.ZAJAL
    if meter in (Meter.MASHHUB, Meter.HILALI, Meter.HIJINI, Meter.SAMIRI, Meter.ZUHAYRI):
        return Tradition.NABATI
    if meter in (Meter.TAFILA, Meter.NATHR):
        return Tradition.MODERN
    return Tradition.FUSHA


def normalize_language_type(raw: str | None) -> LanguageType:
    if not raw:
        return LanguageType.UNKNOWN
    cleaned = clean_arabic(normalize_ws(raw)).lower()
    if any(k in cleaned for k in ("عامي", "عامية", "نبطي", "شعبي", "دارج")):
        return LanguageType.AMMIYA
    if "مزيج" in cleaned or "mixed" in cleaned:
        return LanguageType.MIXED
    return LanguageType.FUSHA
