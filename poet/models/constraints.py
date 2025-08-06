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


@dataclass
class UserConstraints:
    """
    Represents user requirements and constraints for poem generation.
    
    This class encapsulates all the constraints a user might specify
    when requesting a poem, including prosodic, thematic, and stylistic
    requirements.
    """    
    # Prosodic constraints
    meter: Optional[str] = None
    qafiya: Optional[str] = None
    qafiya_harakah: Optional[str] = None  # مفتوح، مكسور، مضموم، ساكن
    qafiya_type: Optional[QafiyaType] = None
    qafiya_pattern: Optional[str] = None  # Exact pattern like "عُ", "قَ", etc.
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
            "qafiya": self.qafiya,
            "qafiya_harakah": self.qafiya_harakah,
            "qafiya_type": self.qafiya_type.value if self.qafiya_type else None,
            "qafiya_pattern": self.qafiya_pattern,
            "line_count": self.line_count,
            "theme": self.theme,
            "tone": self.tone,
            "imagery": self.imagery,
            "keywords": self.keywords,
            "sections": self.sections,
            "register": self.register,
            "era": self.era,
            "poet_style": self.poet_style,
            "ambiguities": self.ambiguities
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserConstraints':
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
            qafiya=data.get("qafiya"),
            qafiya_harakah=data.get("qafiya_harakah"),
            qafiya_type=qafiya_type,
            qafiya_pattern=data.get("qafiya_pattern"),
            line_count=data.get("line_count"),
            theme=data.get("theme"),
            tone=data.get("tone"),
            imagery=data.get("imagery", []),
            keywords=data.get("keywords", []),
            sections=data.get("sections", []),
            register=data.get("register"),
            era=data.get("era"),
            poet_style=data.get("poet_style"),
            ambiguities=data.get("ambiguities", [])
        )
    

    

    
    def __str__(self) -> str:
        """String representation of constraints"""
        parts = []
        if self.meter:
            parts.append(f"البحر: {self.meter}")
        if self.qafiya:
            qafiya_info = f"القافية: {self.qafiya}"
            if self.qafiya_harakah:
                qafiya_info += f" ({self.qafiya_harakah})"
            if self.qafiya_pattern:
                qafiya_info += f" [{self.qafiya_pattern}]"
            parts.append(qafiya_info)
        if self.line_count:
            parts.append(f"الأبيات: {self.line_count}")
        if self.theme:
            parts.append(f"الموضوع: {self.theme}")
        if self.tone:
            parts.append(f"النبرة: {self.tone}")
            
        return " | ".join(parts) if parts else "No constraints specified"