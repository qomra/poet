# tests/unit/test_qafiya_evaluator.py

import pytest
from unittest.mock import Mock, patch
from poet.evaluation.qafiya_evaluator import QafiyaValidator, QafiyaValidationError
from poet.models.qafiya import QafiyaValidationResult, QafiyaBaitResult
from poet.models.poem import LLMPoem
from poet.llm.base_llm import LLMConfig


class TestQafiyaValidator:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_llm.config = LLMConfig(model_name="test-model")
        self.validator = QafiyaValidator(self.mock_llm)
        
        # Sample poem with valid qafiya
        self.valid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
            ],
            llm_provider="test",
            model_name="test-model"
        )
        
        # Sample poem with invalid qafiya
        self.invalid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ",
                "هَذَا بَيْتَ خَاطِئٌ فِي الْقَافِيَةِ",
                "وَهَذَا بَيْتَ آخَرُ خَاطِئٌ أَيْضًا"
            ],
            llm_provider="test",
            model_name="test-model"
        )
    
    def test_validate_qafiya_valid_poem(self):
        """Test qafiya validation with a valid poem"""
        # Mock LLM response for valid poem
        mock_response = '''
        ```json
        {
            "misaligned_lines": [],
            "issues": []
        }
        ```
        '''
        self.mock_llm.generate.return_value = mock_response
        
        result = self.validator.validate_qafiya(self.valid_poem, expected_qafiya="ل")
        
        assert isinstance(result, QafiyaValidationResult)
        assert result.overall_valid is True
        assert result.total_baits == 2
        assert result.valid_baits == 2
        assert result.invalid_baits == 0
        assert result.expected_qafiya == "ل"
        assert result.misaligned_bait_numbers == []
        assert len(result.bait_results) == 2
        
        # Check all bait results are valid
        for bait_result in result.bait_results:
            assert isinstance(bait_result, QafiyaBaitResult)
            assert bait_result.is_valid is True
    
    def test_validate_qafiya_invalid_poem(self):
        """Test qafiya validation with an invalid poem"""
        # Mock LLM response for invalid poem
        mock_response = '''
        ```json
        {
            "misaligned_lines": [3],
            "issues": ["البيت الثالث لا يتبع نفس القافية"]
        }
        ```
        '''
        self.mock_llm.generate.return_value = mock_response
        
        result = self.validator.validate_qafiya(self.invalid_poem, expected_qafiya="ل")
        
        assert isinstance(result, QafiyaValidationResult)
        assert result.overall_valid is False
        assert result.total_baits == 3
        assert result.valid_baits == 2
        assert result.invalid_baits == 1
        assert result.misaligned_bait_numbers == [3]
        
        # Check bait results
        assert len(result.bait_results) == 3
        assert result.bait_results[0].is_valid is True  # Bait 1
        assert result.bait_results[1].is_valid is True  # Bait 2
        assert result.bait_results[2].is_valid is False  # Bait 3
    
    def test_validate_qafiya_invalid_line_count(self):
        """Test qafiya validation with invalid line count (should be handled by LineCountValidator)"""
        invalid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا"
            ],
            llm_provider="test",
            model_name="test-model"
        )
        
        result = self.validator.validate_qafiya(invalid_poem, expected_qafiya="ل")
        
        # Should return invalid result for odd number of lines
        assert isinstance(result, QafiyaValidationResult)
        assert result.total_baits == 0  # No complete baits
        assert result.valid_baits == 0
        assert result.invalid_baits == 0
        assert result.validation_summary == "لا توجد أبيات للتحقق من القافية"
    
    def test_validate_qafiya_empty_poem(self):
        """Test qafiya validation with empty poem"""
        empty_poem = LLMPoem(
            verses=[],
            llm_provider="test",
            model_name="test-model"
        )
        
        result = self.validator.validate_qafiya(empty_poem, expected_qafiya="ل")
        
        assert result.overall_valid is False
        assert result.validation_summary == "لا توجد أبيات للتحقق من القافية"
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response from LLM"""
        mock_response = '''
        ```json
        {
            "misaligned_lines": [2, 4],
            "issues": ["البيت الثاني لا يتبع نفس النمط", "البيت الرابع لا يتبع نفس النمط"]
        }
        ```
        '''
        
        result = self.validator._parse_llm_response(mock_response)
        
        assert result['misaligned_lines'] == [2, 4]
        assert result['issues'] == ["البيت الثاني لا يتبع نفس النمط", "البيت الرابع لا يتبع نفس النمط"]
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response from LLM"""
        mock_response = "This is not JSON"
        
        with pytest.raises(Exception):
            self.validator._parse_llm_response(mock_response)
    
    def test_parse_llm_response_missing_required_field(self):
        """Test parsing response missing required field"""
        mock_response = '''
        ```json
        {
            "issues": []
        }
        ```
        '''
        
        with pytest.raises(QafiyaValidationError, match="Missing required fields"):
            self.validator._parse_llm_response(mock_response)
    
    def test_generate_validation_summary_all_valid(self):
        """Test validation summary generation for all valid baits"""
        summary = self.validator._generate_validation_summary(4, 0)
        assert "جميع الأبيات (4) صحيحة قافياً" in summary
    
    def test_generate_validation_summary_all_invalid(self):
        """Test validation summary generation for all invalid baits"""
        summary = self.validator._generate_validation_summary(0, 4)
        assert "جميع الأبيات (4) خاطئة قافياً" in summary
    
    def test_generate_validation_summary_mixed(self):
        """Test validation summary generation for mixed results"""
        summary = self.validator._generate_validation_summary(2, 2)
        assert "2 من 4 أبيات صحيحة قافياً" in summary
        assert "الأبيات الخاطئة: 1، 2" in summary
    
    def test_to_dict_serialization(self):
        """Test that QafiyaValidationResult can be serialized to dict"""
        bait_result = QafiyaBaitResult(
            bait_number=1,
            is_valid=True
        )
        
        validation_result = QafiyaValidationResult(
            overall_valid=True,
            total_baits=2,
            valid_baits=2,
            invalid_baits=0,
            bait_results=[bait_result],
            validation_summary="جميع الأبيات صحيحة",
            misaligned_bait_numbers=[],
            expected_qafiya="ل"
        )
        
        result_dict = validation_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['overall_valid'] is True
        assert result_dict['total_baits'] == 2
        assert result_dict['expected_qafiya'] == "ل"
        assert len(result_dict['bait_results']) == 1
        assert result_dict['bait_results'][0]['bait_number'] == 1 