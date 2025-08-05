from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class BaitValidationResult:
    """Result of validating a single bait (verse pair)"""
    
    bait_text: str
    is_valid: bool
    pattern: str
    error_details: Optional[str] = None
    diacritized_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "bait_text": self.bait_text,
            "is_valid": self.is_valid,
            "pattern": self.pattern,
            "error_details": self.error_details,
            "diacritized_text": self.diacritized_text
        }


@dataclass
class ProsodyValidationResult:
    """Result of prosody validation for an entire poem"""
    
    overall_valid: bool
    total_baits: int
    valid_baits: int
    invalid_baits: int
    bait_results: List[BaitValidationResult]
    bahr_used: str
    validation_summary: str
    diacritics_applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "overall_valid": self.overall_valid,
            "total_baits": self.total_baits,
            "valid_baits": self.valid_baits,
            "invalid_baits": self.invalid_baits,
            "bait_results": [result.to_dict() for result in self.bait_results],
            "bahr_used": self.bahr_used,
            "validation_summary": self.validation_summary,
            "diacritics_applied": self.diacritics_applied
        } 