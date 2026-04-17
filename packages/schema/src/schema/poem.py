"""
Canonical data models for Arabic poetry entities.

These are the ground-truth types used throughout the system:
ETL pipelines produce them, the DB stores them, the agent works with them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations — normalised vocabularies
# ---------------------------------------------------------------------------


class Tradition(StrEnum):
    FUSHA = "fusha"          # Classical / Modern Standard Arabic
    NABATI = "nabati"        # Nabati / Gulf / Arabian Peninsula colloquial
    ZAJAL = "zajal"          # Lebanese / Levantine zajal
    AMMIYA = "ammiya"        # Other Arabic dialects
    MODERN = "modern"        # Shi'r al-tafila / qasidat al-nathr
    UNKNOWN = "unknown"


class Era(StrEnum):
    PRE_ISLAMIC = "pre_islamic"
    ISLAMIC = "islamic"
    UMAYYAD = "umayyad"
    ABBASID = "abbasid"
    ANDALUSIAN = "andalusian"
    AYYUBID = "ayyubid"
    MAMLUK = "mamluk"
    OTTOMAN = "ottoman"
    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    UNKNOWN = "unknown"


class Theme(StrEnum):
    MADIH = "madih"           # مديح
    HIJA = "hija"             # هجاء
    GHAZAL = "ghazal"         # غزل
    RITHA = "ritha"           # رثاء
    WASF = "wasf"             # وصف
    FAKHR = "fakhr"           # فخر
    HIKMA = "hikma"           # حكمة
    ITIDHAR = "itidhar"       # اعتذار
    ZUHD = "zuhd"             # زهد
    KHAMRIYYAT = "khamriyyat" # خمريات
    HAMASA = "hamasa"         # حماسة
    SUFI = "sufi"             # صوفي
    WASF_TABI = "wasf_tabi"   # وصف الطبيعة
    MADIH_NABAWI = "madih_nabawi"
    OTHER = "other"
    UNKNOWN = "unknown"


class Meter(StrEnum):
    """Canonical Khalilian meter names."""
    TAWEEL = "taweel"
    MADEED = "madeed"
    BASEET = "baseet"
    WAFER = "wafer"
    KAMEL = "kamel"
    HAZAJ = "hazaj"
    RAJAZ = "rajaz"
    RAMAL = "ramal"
    SAREE = "saree"
    MUNSAREH = "munsareh"
    KHAFEEF = "khafeef"
    MUDHARE = "mudhare"
    MUQTADHEB = "muqtadheb"
    MUJTATH = "mujtath"
    MUTAKAREB = "mutakareb"
    MUTADARAK = "mutadarak"
    # Nabati meters
    MASHHUB = "mashhub"
    HILALI = "hilali"
    HIJINI = "hijini"
    SAMIRI = "samiri"
    ZUHAYRI = "zuhayri"
    # Sub-meters / partial
    MAJZOO_KAMEL = "majzoo_kamel"
    MAJZOO_RAJAZ = "majzoo_rajaz"
    MAJZOO_RAMAL = "majzoo_ramal"
    MAJZOO_BASEET = "majzoo_baseet"
    MAKHALLA_BASEET = "makhalla_baseet"
    # Modern / free
    TAFILA = "tafila"          # Free verse retaining a foot
    NATHR = "nathr"            # Prose poem
    UNKNOWN = "unknown"


class LanguageType(StrEnum):
    FUSHA = "fusha"
    AMMIYA = "ammiya"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class Poet(BaseModel):
    """A poet — canonical record."""

    id: UUID = Field(default_factory=uuid4)
    name_ar: str
    name_en: str | None = None
    kunya: str | None = None           # أبو الطيب، أبو نواس ...
    laqab: str | None = None           # المتنبي، أبو تمام ...
    birth_year: int | None = None      # Gregorian
    death_year: int | None = None
    era: Era = Era.UNKNOWN
    tradition: Tradition = Tradition.FUSHA
    region: str | None = None          # Saudi Arabia, Egypt, Andalusia ...
    tribe: str | None = None           # Critical for Nabati attribution
    biography: str | None = None
    signature_meters: list[Meter] = Field(default_factory=list)
    signature_themes: list[Theme] = Field(default_factory=list)
    diwan_url: str | None = None
    wikipedia_url: str | None = None

    @field_validator("name_ar")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name_ar must not be empty")
        return v.strip()


class Verse(BaseModel):
    """A single verse (شطر) — the atomic unit for search and attribution."""

    id: UUID = Field(default_factory=uuid4)
    poem_id: UUID
    position: int                      # 0-indexed position within poem
    hemistich: int                     # 0 = sadr, 1 = ajuz
    text: str                          # without diacritics (for search)
    text_diacritized: str | None = None  # with full tashkeel (if available)


class Poem(BaseModel):
    """A poem — the primary content unit."""

    id: UUID = Field(default_factory=uuid4)
    title: str | None = None
    poet_id: UUID | None = None
    poet_name: str | None = None       # denormalised for display

    verses: list[str]                  # each string is one verse (bayt)
    verse_count: int = Field(default=0)

    tradition: Tradition = Tradition.FUSHA
    language_type: LanguageType = LanguageType.FUSHA
    meter: Meter = Meter.UNKNOWN
    meter_raw: str | None = None       # original string from source
    rhyme_letter: str | None = None    # الروي
    rhyme_harakah: str | None = None   # fatha / kasra / damma / sukun
    sadr_rhyme: str | None = None      # Nabati dual-rhyme: sadr
    ajuz_rhyme: str | None = None      # Nabati dual-rhyme: ajuz
    theme: Theme = Theme.UNKNOWN
    theme_raw: str | None = None
    era: Era = Era.UNKNOWN
    dialect: str | None = None

    # Provenance
    sources: list[str] = Field(default_factory=list)  # source identifiers
    source_urls: list[str] = Field(default_factory=list)
    canonical_source: str | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.verse_count:
            self.verse_count = len(self.verses)


# ---------------------------------------------------------------------------
# ETL intermediate records
# ---------------------------------------------------------------------------


class RawPoem(BaseModel):
    """
    A poem exactly as it came from a source — no normalisation applied.
    Stored in the raw lake before any transformation.
    """

    source: str                        # e.g. "hf:arbml/ashaar", "crawl:diwan.com"
    source_id: str                     # ID within that source
    raw: dict                          # original record verbatim


class ProcessedPoem(BaseModel):
    """
    A poem after schema mapping and light cleaning — ready for dedup + load.
    """

    source: str
    source_id: str
    title: str | None = None
    poet_name: str | None = None
    poet_description: str | None = None  # biography text from source
    poet_era_raw: str | None = None
    poet_location_raw: str | None = None
    poet_url: str | None = None
    verses: list[str]
    meter_raw: str | None = None
    rhyme_raw: str | None = None
    theme_raw: str | None = None
    language_type_raw: str | None = None
    tradition: Tradition = Tradition.FUSHA
    form: str = "qasida"     # qasida / muhawara / zajal / tafila / nathr / ardah / samri
    tariq: str | None = None # محاورة melody-meter path
    occasion: str | None = None
    source_url: str | None = None
    has_diacritics: bool = False


class EnrichedPoem(BaseModel):
    """
    A poem after normalisation, deduplication, and enrichment.
    Has canonical meter/theme/era values. Ready to load into Postgres + Qdrant.
    This maps 1:1 to the Poem model.
    """

    poem: Poem
    dedup_key: str              # hash for cross-source duplicate detection
    duplicate_of: UUID | None = None  # points to canonical record if dupe
    embedding_model: str | None = None
    embedding: list[float] | None = None
