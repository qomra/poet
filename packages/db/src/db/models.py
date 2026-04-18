"""
SQLAlchemy ORM models — maps directly to the canonical schema.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Poet(Base):
    __tablename__ = "poets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name_ar: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    name_en: Mapped[str | None] = mapped_column(String(256))
    kunya: Mapped[str | None] = mapped_column(String(128))
    laqab: Mapped[str | None] = mapped_column(String(128))
    birth_year: Mapped[int | None] = mapped_column(Integer)
    death_year: Mapped[int | None] = mapped_column(Integer)
    era: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    tradition: Mapped[str] = mapped_column(String(64), default="fusha", index=True)
    region: Mapped[str | None] = mapped_column(String(128))
    tribe: Mapped[str | None] = mapped_column(String(128))
    biography: Mapped[str | None] = mapped_column(Text)
    signature_meters: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    signature_themes: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    diwan_url: Mapped[str | None] = mapped_column(String(512))
    wikipedia_url: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    poems: Mapped[list[Poem]] = relationship("Poem", back_populates="poet")


class Poem(Base):
    __tablename__ = "poems"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(512))
    poet_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("poets.id"), index=True)
    poet_name: Mapped[str | None] = mapped_column(String(256))  # denormalised

    tradition: Mapped[str] = mapped_column(String(64), default="fusha", index=True)
    language_type: Mapped[str] = mapped_column(String(64), default="fusha")
    meter: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    meter_raw: Mapped[str | None] = mapped_column(String(256))
    rhyme_letter: Mapped[str | None] = mapped_column(String(8), index=True)
    rhyme_harakah: Mapped[str | None] = mapped_column(String(32))
    sadr_rhyme: Mapped[str | None] = mapped_column(String(8))   # Nabati dual-rhyme
    ajuz_rhyme: Mapped[str | None] = mapped_column(String(8))
    theme: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    theme_raw: Mapped[str | None] = mapped_column(String(256))
    era: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    dialect: Mapped[str | None] = mapped_column(String(64))
    verse_count: Mapped[int] = mapped_column(Integer, default=0)

    # Structural form — determines evaluation logic and search behaviour
    form: Mapped[str] = mapped_column(String(64), default="qasida", index=True)
    # قصيدة / مقطوعة / محاورة / زجل / تفعيلة / نثر / عرضة / سامري / حداء

    # محاورة-specific: the melody-meter path both poets must follow
    tariq: Mapped[str | None] = mapped_column(String(128))

    # Occasion / context (مناسبة) — for مناسبات and performance poetry
    occasion: Mapped[str | None] = mapped_column(String(256))

    sources: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    canonical_source: Mapped[str | None] = mapped_column(String(128))

    # Qafiya classification — populated by a matching rule (NULL = unclassified)
    qafiya_rawiy: Mapped[str | None] = mapped_column(String(8))
    qafiya_radf: Mapped[str | None] = mapped_column(String(4))
    qafiya_taassis: Mapped[bool | None] = mapped_column(Boolean)
    qafiya_wasl: Mapped[str | None] = mapped_column(String(4))
    qafiya_harakah: Mapped[str | None] = mapped_column(String(32))
    qafiya_type: Mapped[str | None] = mapped_column(String(32))
    qafiya_pattern: Mapped[str | None] = mapped_column(String(32))
    qafiya_confidence: Mapped[str | None] = mapped_column(String(16), index=True)
    qafiya_rule_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("rules.id", ondelete="SET NULL"), index=True
    )

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    poet: Mapped[Poet | None] = relationship("Poet", back_populates="poems")
    verses: Mapped[list[Verse]] = relationship("Verse", back_populates="poem",
                                                cascade="all, delete-orphan")


class Verse(Base):
    """Atomic unit — every verse is individually searchable and attributable."""

    __tablename__ = "verses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    poem_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("poems.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)   # 0-indexed within poem
    text: Mapped[str] = mapped_column(Text, nullable=False)           # no diacritics (search)
    text_diacritized: Mapped[str | None] = mapped_column(Text)        # with tashkeel

    # Per-verse poet attribution — essential for محاورة (alternating poets)
    poet_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("poets.id"), index=True)

    # زجل strophic structure
    stanza_index: Mapped[int | None] = mapped_column(Integer)         # which stanza (0-indexed)
    stanza_role: Mapped[str | None] = mapped_column(String(32))       # matla / dour / qafla

    poem: Mapped[Poem] = relationship("Poem", back_populates="verses")
    poet: Mapped[Poet | None] = relationship("Poet")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)  # YouTube ID
    url: Mapped[str] = mapped_column(String(1024))
    title: Mapped[str | None] = mapped_column(String(512))
    channel: Mapped[str | None] = mapped_column(String(256))
    event_type: Mapped[str | None] = mapped_column(String(64), index=True)  # محاورة / عرضة / etc.
    tradition: Mapped[str | None] = mapped_column(String(64), index=True)
    upload_date: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
    duration_s: Mapped[int | None] = mapped_column(Integer)
    asr_confidence: Mapped[float | None] = mapped_column(Float)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    segments: Mapped[list[VideoSegment]] = relationship("VideoSegment", back_populates="video",
                                                         cascade="all, delete-orphan")


class VideoSegment(Base):
    """A timestamped verse-level segment from a video transcript."""

    __tablename__ = "video_segments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    speaker_id: Mapped[str | None] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    identified_verse_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("verses.id"))
    confidence: Mapped[float | None] = mapped_column(Float)

    video: Mapped[Video] = relationship("Video", back_populates="segments")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str | None] = mapped_column(String(256), index=True)
    language: Mapped[str] = mapped_column(String(8), default="ar")
    tradition_context: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(),
                                                  onupdate=func.now())

    messages: Mapped[list[Message]] = relationship("Message", back_populates="session",
                                                    cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)   # user / assistant / tool
    content: Mapped[str | None] = mapped_column(Text)
    tool_calls_json: Mapped[str | None] = mapped_column(Text)       # JSON array
    tool_results_json: Mapped[str | None] = mapped_column(Text)     # JSON array
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[Session] = relationship("Session", back_populates="messages")


class QafiyaAnnotation(Base):
    """Human validation of the mechanistic qafiya classifier output."""

    __tablename__ = "qafiya_annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    poem_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("poems.id", ondelete="CASCADE"), index=True)
    verdict: Mapped[str] = mapped_column(String(16))  # accept / reject / partial / skip
    # If reject/partial: what the correct values should be
    correct_rawiy: Mapped[str | None] = mapped_column(String(8))
    correct_radf: Mapped[str | None] = mapped_column(String(4))
    correct_wasl: Mapped[str | None] = mapped_column(String(4))
    correct_harakah: Mapped[str | None] = mapped_column(String(32))
    correct_type: Mapped[str | None] = mapped_column(String(32))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Rule(Base):
    """A qafiya classification rule — proposed by the annotator, coded as a Python
    function, and applied to the unclassified pool. Each matching poem gets its
    qafiya_* fields populated and is linked back here via poems.qafiya_rule_id.
    """

    __tablename__ = "rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code: Mapped[str | None] = mapped_column(String(16), unique=True)  # r001, r002 — set when coded
    title_ar: Mapped[str] = mapped_column(Text, nullable=False)
    description_ar: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="proposed", index=True)
    # proposed → coded → applied
    function_name: Mapped[str | None] = mapped_column(String(128))
    poems_matched: Mapped[int] = mapped_column(Integer, default=0)
    example_poem_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("poems.id", ondelete="SET NULL")
    )
    example_qafiya_json: Mapped[str | None] = mapped_column(Text)  # expected qafiya for the example
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    coded_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
    applied_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))
