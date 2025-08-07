from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from .prosody import ProsodyValidationResult
from .qafiya import QafiyaValidationResult
from .line_count import LineCountValidationResult
from .tashkeel import TashkeelValidationResult


@dataclass
class QualityAssessment:
    """Centralized validation issues container"""
    
    prosody_issues: List[str]
    line_count_issues: List[str]
    qafiya_issues: List[str]
    overall_score: float
    is_acceptable: bool
    recommendations: List[str]
    tashkeel_issues: List[str] = None
    
    # Detailed validation results
    prosody_validation: Optional[ProsodyValidationResult] = None
    qafiya_validation: Optional[QafiyaValidationResult] = None
    line_count_validation: Optional[LineCountValidationResult] = None
    tashkeel_validation: Optional[TashkeelValidationResult] = None
    
    def __post_init__(self):
        if self.tashkeel_issues is None:
            self.tashkeel_issues = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "prosody_issues": self.prosody_issues,
            "line_count_issues": self.line_count_issues,
            "qafiya_issues": self.qafiya_issues,
            "tashkeel_issues": self.tashkeel_issues,
            "overall_score": self.overall_score,
            "is_acceptable": self.is_acceptable,
            "recommendations": self.recommendations,
            "prosody_validation": self.prosody_validation.to_dict() if self.prosody_validation else None,
            "qafiya_validation": self.qafiya_validation.to_dict() if self.qafiya_validation else None,
            "line_count_validation": self.line_count_validation.to_dict() if self.line_count_validation else None,
            "tashkeel_validation": self.tashkeel_validation.to_dict() if self.tashkeel_validation else None
        } 