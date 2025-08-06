# poet/models/line_count.py

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LineCountValidationResult:
    """Result of line count validation"""
    
    is_valid: bool
    line_count: int
    expected_even: bool
    validation_summary: str
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "is_valid": self.is_valid,
            "line_count": self.line_count,
            "expected_even": self.expected_even,
            "validation_summary": self.validation_summary,
            "error_details": self.error_details
        } 