# poet/models/qafiya.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class QafiyaBaitResult:
    """Result of validating qafiya for a single bait (verse pair)"""
    
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
class QafiyaValidationResult:
    """Result of qafiya validation for an entire poem"""
    
    overall_valid: bool
    total_baits: int
    valid_baits: int
    invalid_baits: int
    bait_results: List[QafiyaBaitResult]
    validation_summary: str
    misaligned_bait_numbers: List[int]
    expected_qafiya: Optional[str] = None
    qafiya_harakah: Optional[str] = None
    qafiya_type: Optional[str] = None
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
            "misaligned_bait_numbers": self.misaligned_bait_numbers,
            "expected_qafiya": self.expected_qafiya,
            "qafiya_harakah": self.qafiya_harakah,
            "qafiya_type": self.qafiya_type,
            "issues": self.issues
        } 