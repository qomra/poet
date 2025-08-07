
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TashkeelBaitResult:
    """Result of validating tashkeel for a single bait (verse pair)"""
    
    bait_number: int
    is_valid: bool
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "bait_number": self.bait_number,
            "is_valid": self.is_valid,
            "error_details": self.error_details
        }
@dataclass
class TashkeelValidationResult:
    """Result of tashkeel validation for an entire poem"""
    
    overall_valid: bool
    total_baits: int
    valid_baits: int
    invalid_baits: int
    bait_results: List[TashkeelBaitResult]
    validation_summary: str
    issues: List[str] = None
    
    def __post_init__(self):
        """Initialize issues from bait_results if not provided"""
        if self.issues is None:
            self.issues = []
            if self.bait_results:
                for bait_result in self.bait_results:
                    if not bait_result.is_valid and bait_result.error_details:
                        self.issues.append(bait_result.error_details)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "overall_valid": self.overall_valid,
            "total_baits": self.total_baits,
            "valid_baits": self.valid_baits,
            "invalid_baits": self.invalid_baits,
            "bait_results": [result.to_dict() for result in self.bait_results],
            "validation_summary": self.validation_summary,
            "issues": self.issues
        }