# poet/evaluation/line_count_validator.py

import logging
from poet.models.poem import LLMPoem
from poet.models.line_count import LineCountValidationResult


class LineCountValidator:
    """
    Validates that a poem has the correct line count (even number of lines).
    
    This validator should be run before other validators to ensure
    the poem structure is valid for further analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_line_count(self, poem: LLMPoem) -> LineCountValidationResult:
        """
        Validate that the poem has an even number of lines.
        
        Args:
            poem: LLMPoem object to validate
            
        Returns:
            LineCountValidationResult with validation details
        """
        line_count = len(poem.verses)
        is_valid = line_count % 2 == 0
        
        if is_valid:
            bait_count = line_count // 2
            validation_summary = f"عدد الأبيات صحيح: {bait_count} بيت ({line_count} شطر)"
            error_details = None
        else:
            validation_summary = f"عدد الأبيات يجب أن يكون زوجياً، الحالي: {line_count} شطر"
            error_details = "عدد الأبيات يجب أن يكون زوجياً"
        
        return LineCountValidationResult(
            is_valid=is_valid,
            line_count=line_count,
            expected_even=True,
            validation_summary=validation_summary,
            error_details=error_details
        ) 