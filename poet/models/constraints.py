# poet/models/constraints.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum

# Import DataExample types only for type checking to avoid circular imports
if TYPE_CHECKING:
    from poet.models.search import CorpusExample, WebExample

class QafiyaType(Enum):
    """Types of Arabic qafiya (rhyme) patterns"""
    MUTAWATIR = "متواتر"  # One vowel between two consonants
    MUTARAKIB = "متراكب"  # Three vowels between two consonants
    MUTADAARIK = "متدارك"  # Two vowels between two consonants
    MUTAKAASIS = "متكاوس"  # Four vowels between two consonants
    MUTARADIF = "مترادف"   # Two consonants together

class QafiyaTypeDescriptionAndExamples(Enum):
    """
    Types of Arabic qafiya (rhyme) patterns with detailed descriptions and examples
    """
    MUTAWATIR = "قافیة - Mutawatir (المُتواتِر): One vowel between the last two consonants. This creates a melodic ending. Examples: رَبَبْ, شَجَنْ, حَیاةْ, وَطَنْ"
    MUTARAKIB = "قافیة - Mutarakib (المُتراكِب): Three vowels between the last two consonants. This pattern is more complex and creates flowing endings. Examples: عَزیزانْ, حَبیبانْ, خَضِرُونْ, قادِمُونْ"
    MUTADAARIK = "قافیة - Mutadaarik (المُتدارِك): Two vowels between the last two consonants. This pattern offers moderate complexity. Examples: مَکانُهْ, سَماؤُهْ, حِکایَتْ, کِتابَتْ"
    MUTAKAASIS = "قافیة - Mutakaasis (المُتكاوِس): Four vowels between the last two consonants. This is the most complex pattern with flowing, cascading endings. Examples: العُذیوبَهْ, المحبوبَهْ, العجیبَهْ, الحکیمَهْ"
    MUTARADIF = "oo - Mutaradif (المُترادِف): Two consecutive sukuns at the end of the verse. This pattern creates an abrupt stop and often occurs when a long vowel precedes the final consonant. Examples: دِینْ, عَیْنْ, حینْ, أَیْنْ"


def get_letter_name(letter):
    name_map = {
        'ا': 'الف',
        'ب': 'الباء',
        'ت': 'التاء',
        'ث': 'الثاء',
        'ج': 'الجيم',
        'ح': 'الحاء',
        'خ': 'الخاء',
        'د': 'الدال',
        'ذ': 'الذال',
        'ر': 'الراء',
        'ز': 'الزاي',
        'س': 'السين',
        'ش': 'الشين',
        'ص': 'الصاد',
        'ض': 'الضاد',
        'ط': 'الطاء',
        'ظ': 'الظاء',
        'ع': 'العين',
        'غ': 'الغين',
        'ف': 'الفاء',
        'ق': 'القاف',
        'ك': 'الكاف',
        'ل': 'اللام',
        'م': 'الميم',
        'ن': 'النون',
        'ه': 'الهاء',
        'و': 'الواو',
        'ي': 'الياء',
        "ألف": "الف",
        "باء": "الباء",
        "تاء": "التاء",
        "ثاء": "الثاء",
        "جيم": "الجيم",
        "حاء": "الحاء",
        "خاء": "الخاء",
        "دال": "الدال",
        "ذال": "الذال",
        "راء": "الراء",
        "زاي": "الزاي",
        "سين": "السين",
        "شين": "الشين",
        "صاد": "الصاد",
        "ضاد": "الضاد",
        "طاء": "الطاء",
        "ظاء": "الظاء",
        "عين": "العين",
        "غين": "الغين",
        "فاء": "الفاء",
        "قاف": "القاف",
        "كاف": "الكاف",
        "لام": "اللام",
        "ميم": "الميم",
        "نون": "النون",
        "هاء": "الهاء",
        "واو": "الواو",
        "ياء": "الياء",
    }

    return name_map.get(letter, letter)



@dataclass
class ExampleData:
    """Type-safe structure for example data containing retrieved poems"""
    corpus_examples: List['CorpusExample'] = field(default_factory=list)
    web_examples: List['WebExample'] = field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


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
    
    # Data enrichment - retrieved examples from corpus and web search
    example_data: Optional[ExampleData] = field(default=None, init=False)
    
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
        
        if self.qafiya:
            self.qafiya = get_letter_name(self.qafiya)
    def has_ambiguities(self) -> bool:
        """Check if constraints have ambiguities"""
        return len(self.ambiguities) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary"""
        # Serialize example_data with proper DataExample handling
        serialized_example_data = None
        if self.example_data:
            serialized_example_data = {}
            
            # Serialize corpus examples
            corpus_examples = self.example_data.corpus_examples
            if corpus_examples:
                serialized_corpus = []
                for example in corpus_examples:
                    if hasattr(example, '__dict__'):
                        # It's a DataExample object, convert to dict
                        example_dict = {
                            'search_criteria': example.search_criteria,
                            'metadata': example.metadata,
                            'title': example.title,
                            'verses': example.verses,
                            'meter': example.meter,
                            'qafiya': example.qafiya,
                            'theme': example.theme,
                            'poet_name': example.poet_name,
                            'poet_era': example.poet_era
                        }
                        serialized_corpus.append(example_dict)
                    else:
                        # Already a dict
                        serialized_corpus.append(example)
                serialized_example_data["corpus_examples"] = serialized_corpus
            
            # Serialize web examples  
            web_examples = self.example_data.web_examples
            if web_examples:
                serialized_web = []
                for example in web_examples:
                    if hasattr(example, '__dict__'):
                        # It's a DataExample object, convert to dict
                        example_dict = {
                            'search_criteria': example.search_criteria,
                            'metadata': example.metadata,
                            'title': example.title,
                            'content': example.content,
                            'url': example.url,
                            'relevance_score': example.relevance_score
                        }
                        serialized_web.append(example_dict)
                    else:
                        # Already a dict
                        serialized_web.append(example)
                serialized_example_data["web_examples"] = serialized_web
            
            # Copy over other metadata
            serialized_example_data["retrieval_metadata"] = self.example_data.retrieval_metadata
        
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
            "original_prompt": self.original_prompt,
            "example_data": serialized_example_data
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
        
        # Create constraints with init=True fields only
        constraints = cls(
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
        
        # Set init=False fields after initialization
        constraints.llm_suggestions = data.get("llm_suggestions")
        constraints.llm_reasoning = data.get("llm_reasoning")
        
        # Reconstruct example_data with proper DataExample objects
        example_data_dict = data.get("example_data")
        if example_data_dict:
            from poet.models.search import CorpusExample, WebExample
            
            reconstructed_example_data = ExampleData(
                corpus_examples=[],
                web_examples=[],
                retrieval_metadata={}
            )
            
            # Reconstruct corpus examples
            corpus_examples = example_data_dict.get("corpus_examples", [])
            if corpus_examples:
                reconstructed_corpus = []
                for example_dict in corpus_examples:
                    if isinstance(example_dict, dict):
                        # Reconstruct CorpusExample from dict
                        corpus_example = CorpusExample(
                            search_criteria=example_dict.get("search_criteria", []),
                            metadata=example_dict.get("metadata", {}),
                            title=example_dict.get("title", ""),
                            verses=example_dict.get("verses", ""),
                            meter=example_dict.get("meter", ""),
                            qafiya=example_dict.get("qafiya", ""),
                            theme=example_dict.get("theme", ""),
                            poet_name=example_dict.get("poet_name", ""),
                            poet_era=example_dict.get("poet_era", "")
                        )
                        reconstructed_corpus.append(corpus_example)
                    else:
                        # Already a CorpusExample object
                        reconstructed_corpus.append(example_dict)
                reconstructed_example_data.corpus_examples = reconstructed_corpus
            
            # Reconstruct web examples
            web_examples = example_data_dict.get("web_examples", [])
            if web_examples:
                reconstructed_web = []
                for example_dict in web_examples:
                    if isinstance(example_dict, dict):
                        # Reconstruct WebExample from dict
                        web_example = WebExample(
                            search_criteria=example_dict.get("search_criteria", []),
                            metadata=example_dict.get("metadata", {}),
                            title=example_dict.get("title", ""),
                            content=example_dict.get("content", ""),
                            url=example_dict.get("url", ""),
                            relevance_score=example_dict.get("relevance_score")
                        )
                        reconstructed_web.append(web_example)
                    else:
                        # Already a WebExample object
                        reconstructed_web.append(example_dict)
                reconstructed_example_data.web_examples = reconstructed_web
            
            # Copy over other metadata
            reconstructed_example_data.retrieval_metadata = example_data_dict.get("retrieval_metadata", {})
            
            constraints.example_data = reconstructed_example_data
        else:
            constraints.example_data = data.get("example_data")
        
        return constraints
    

    

    
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