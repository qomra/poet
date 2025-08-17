# poet/models/constraints.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class QafiyaType(Enum):
    """Types of Arabic qafiya (rhyme) patterns"""
    MUTAWATIR = "متواتر"  # One vowel between two consonants
    MUTARAKIB = "متراكب"  # Three vowels between two consonants
    MUTADAARIK = "متدارك"  # Two vowels between two consonants
    MUTAKAASIS = "متكاوس"  # Four vowels between two consonants
    MUTARADIF = "مترادف"   # Two consonants together

class QafiyaTypeDescriptionAndExamples(Enum):
    """
    Detailed descriptions and examples of Arabic qafiya types.

    Each member contains a single string describing the qafiya pattern, where '/'
    represents a vowel (haraka) and 'o' represents a cessation (sukun).
    """
    MUTAKAASIS = "////o - Mutakaathis (المُتَكاثِس): A succession of four voweled letters followed by a sukun. This is the most complex and rarest type, suggesting a 'piling up' of vowels. Examples: الزَمانُ صَدَعَكَ, الإلَهُ فَجَبَرَ"
    MUTARAKIB = "///o - Mutarakib (المُتراكِب): A succession of three voweled letters followed by a sukun. Its name means 'compounded,' and it creates a flowing, consecutive rhythm. Examples: سَمَرُ, فَتَنُ, نَظَمُ, هَزَلُ"
    MUTADAARIK = "//o - Mutadaarik (المُتدارِك): A succession of two voweled letters followed by a sukun. Meaning 'the one that follows,' it's a very common and balanced qafiya type. Examples: مُنْتَهِي, مٌقْتَفِي, نَاظِمِ, رَاسِمُ"
    MUTAWATIR = "/o - Mutawatir (المُتواتِر): A single voweled letter followed by a sukun. Its name means 'alternating,' and it is one of the most frequent and simple qafiya patterns, creating a crisp ending. Examples: جَميلُ, كَريمُ, جَمالُ, جَمانُ"
    MUTARADIF = "oo - Mutaradif (المُترادِف): Two consecutive sukuns at the end of the verse. This pattern creates an abrupt stop and often occurs when a long vowel precedes the final consonant. Examples: دِينْ, عَيْنْ, حينْ, أَيْنْ"


@dataclass
class Constraints:
    """
    Represents user requirements and constraints for poem generation.
    
    This class encapsulates all the constraints a user might specify
    when requesting a poem, including prosodic, thematic, and stylistic
    requirements.
    """    
    # Prosodic constraints
    meter: Optional[str] = None
    meeter_tafeelat: Optional[str] = None
    qafiya: Optional[str] = None
    qafiya_harakah: Optional[str] = None  # مفتوح، مكسور، مضموم، ساكن
    qafiya_type: Optional[QafiyaType] = None
    qafiya_type_description_and_examples: Optional[str] = field(default=None, init=False) # infered not passed by user
    line_count: Optional[int] = None
    
    # Thematic constraints  
    theme: Optional[str] = None
    tone: Optional[str] = None
    imagery: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Structural constraints
    sections: List[str] = field(default_factory=list)  # e.g., مقدمة، غزل، تخلص
    
    # Style constraints
    register: Optional[str] = None  # formal, colloquial, etc.
    era: Optional[str] = None  # classical, modern
    poet_style: Optional[str] = None  # specific poet to emulate
    
    # Meta information
    ambiguities: List[str] = field(default_factory=list)
    
    # LLM extraction metadata (not part of __post_init__ processing)
    llm_suggestions: Optional[str] = field(default=None, init=False)
    llm_reasoning: Optional[str] = field(default=None, init=False)
    original_prompt: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Basic validation after initialization"""
        # Strip whitespace from string fields
        string_fields = ['meter', 'qafiya', 'theme', 'tone', 'register', 'era', 'poet_style']
        for field_name in string_fields:
            value = getattr(self, field_name)
            if value:
                setattr(self, field_name, value.strip())
        
        # infer qafiya_type_examples from qafiya_type
        if self.qafiya_type:
            # convert qafiya_type to QafiyaTypeDescriptionAndExamples enum
            self.qafiya_type_description_and_examples = QafiyaTypeDescriptionAndExamples[self.qafiya_type.name].value
            
        # Basic validation
        if self.line_count is not None and self.line_count <= 0:
            raise ValueError("Line count must be positive")
    
    def has_ambiguities(self) -> bool:
        """Check if constraints have ambiguities"""
        return len(self.ambiguities) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary"""
        return {
            "meter": self.meter,
            "meeter_tafeelat": self.meeter_tafeelat,
            "qafiya": self.qafiya,
            "qafiya_harakah": self.qafiya_harakah,
            "qafiya_type": self.qafiya_type.value if self.qafiya_type is not None else None,
            "qafiya_type_description_and_examples": self.qafiya_type_description_and_examples,
            "line_count": self.line_count,
            "theme": self.theme,
            "tone": self.tone,
            "imagery": self.imagery,
            "keywords": self.keywords,
            "sections": self.sections,
            "register": self.register,
            "era": self.era,
            "poet_style": self.poet_style,
            "ambiguities": self.ambiguities,
            "llm_suggestions": self.llm_suggestions,
            "llm_reasoning": self.llm_reasoning,
            "original_prompt": self.original_prompt
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constraints':
        """Create constraints from dictionary"""
        # Handle qafiya_type enum conversion
        qafiya_type = None
        if data.get("qafiya_type"):
            try:
                qafiya_type = QafiyaType(data["qafiya_type"])
            except ValueError:
                # If the value doesn't match any enum, try to find by name
                for qtype in QafiyaType:
                    if qtype.value == data["qafiya_type"]:
                        qafiya_type = qtype
                        break
        
        return cls(
            meter=data.get("meter"),
            meeter_tafeelat=data.get("meeter_tafeelat"),
            qafiya=data.get("qafiya"),
            qafiya_harakah=data.get("qafiya_harakah"),
            qafiya_type=qafiya_type,
            line_count=data.get("line_count"),
            theme=data.get("theme"),
            tone=data.get("tone"),
            imagery=data.get("imagery", []),
            keywords=data.get("keywords", []),
            sections=data.get("sections", []),
            register=data.get("register"),
            era=data.get("era"),
            poet_style=data.get("poet_style"),
            ambiguities=data.get("ambiguities", []),
            original_prompt=data.get("original_prompt")
        )
    

    

    
    def __str__(self) -> str:
        """String representation of constraints"""
        parts = []
        if self.meter:
            parts.append(f"البحر: {self.meter}")
        if self.meeter_tafeelat:
            parts.append(f"تفعيلات البحر: {self.meeter_tafeelat}")
        if self.qafiya:
            qafiya_info = f"القافية: {self.qafiya}"
            if self.qafiya_harakah:
                qafiya_info += f" ({self.qafiya_harakah})"
            if self.qafiya_type:
                qafiya_info += f" ({self.qafiya_type})"
            if self.qafiya_type_description_and_examples:
                qafiya_info += f" {self.qafiya_type_description_and_examples}"
            parts.append(qafiya_info)
        if self.line_count:
            parts.append(f"الأبيات: {self.line_count}")
        if self.theme:
            parts.append(f"الموضوع: {self.theme}")
        if self.tone:
            parts.append(f"النبرة: {self.tone}")
            
        return " | ".join(parts) if parts else "No constraints specified"