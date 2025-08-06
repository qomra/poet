# tests/unit/test_line_count_validator.py

import pytest
from poet.evaluation.line_count_validator import LineCountValidator
from poet.models.line_count import LineCountValidationResult
from poet.models.poem import LLMPoem


class TestLineCountValidator:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = LineCountValidator()
    
    def test_validate_line_count_valid_poem(self):
        """Test line count validation with valid poem (even number of lines)"""
        valid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
            ],
            llm_provider="test",
            model_name="test-model"
        )
        
        result = self.validator.validate_line_count(valid_poem)
        
        assert isinstance(result, LineCountValidationResult)
        assert result.is_valid is True
        assert result.line_count == 4
        assert result.expected_even is True
        assert "صحيح" in result.validation_summary
        assert result.error_details is None
    
    def test_validate_line_count_invalid_poem(self):
        """Test line count validation with invalid poem (odd number of lines)"""
        invalid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا"
            ],
            llm_provider="test",
            model_name="test-model"
        )
        
        result = self.validator.validate_line_count(invalid_poem)
        
        assert isinstance(result, LineCountValidationResult)
        assert result.is_valid is False
        assert result.line_count == 3
        assert result.expected_even is True
        assert "يجب أن يكون زوجياً" in result.validation_summary
        assert result.error_details == "عدد الأبيات يجب أن يكون زوجياً"
    
    def test_validate_line_count_empty_poem(self):
        """Test line count validation with empty poem"""
        empty_poem = LLMPoem(
            verses=[],
            llm_provider="test",
            model_name="test-model"
        )
        
        result = self.validator.validate_line_count(empty_poem)
        
        assert isinstance(result, LineCountValidationResult)
        assert result.is_valid is True  # 0 is even
        assert result.line_count == 0
        assert result.expected_even is True
    
    def test_to_dict_serialization(self):
        """Test that LineCountValidationResult can be serialized to dict"""
        result = LineCountValidationResult(
            is_valid=True,
            line_count=4,
            expected_even=True,
            validation_summary="عدد الأبيات صحيح: 4 بيت"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['is_valid'] is True
        assert result_dict['line_count'] == 4
        assert result_dict['expected_even'] is True
        assert result_dict['validation_summary'] == "عدد الأبيات صحيح: 4 بيت"
        assert result_dict['error_details'] is None 