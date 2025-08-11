# tests/unit/test_qafiya_evaluator.py

import pytest
from unittest.mock import Mock, patch
from poet.evaluation.qafiya import QafiyaEvaluator, QafiyaValidationError
from poet.models.qafiya import QafiyaValidationResult, QafiyaBaitResult
from poet.models.poem import LLMPoem
from poet.llm.base_llm import LLMConfig


class TestQafiyaEvaluator:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_llm.config = LLMConfig(model_name="test-model")
        self.validator = QafiyaEvaluator(self.mock_llm)
        
        # Sample poem with valid qafiya (2 baits)
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
        
        # Sample poem with invalid qafiya (3 baits, last one invalid)
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
    
    def test_evaluate_qafiya_valid_poem(self):
        """Test qafiya validation with a valid poem - each bait evaluated individually"""
        # Mock LLM responses for each bait (2 baits, both valid)
        mock_responses = [
            '{"is_valid": true, "issue": null}',
            '{"is_valid": true, "issue": null}'
        ]
        self.mock_llm.generate.side_effect = mock_responses
        
        with patch('poet.prompts.prompt_manager.PromptManager.format_prompt') as mock_format_prompt:
            mock_format_prompt.return_value = "test qafiya validation prompt"
            
            result = self.validator.evaluate_qafiya(
                self.valid_poem, 
                expected_qafiya="ل",
                qafiya_harakah="مكسور",
                qafiya_type="متواتر",
                qafiya_type_description_and_examples="متواتر: متحرك واحد بين ساكنين"
            )
            
            # Verify LLM was called twice (once for each bait)
            assert self.mock_llm.generate.call_count == 2
            
            # Verify result
            assert isinstance(result, QafiyaValidationResult)
            assert result.overall_valid is True
            assert result.total_baits == 2
            assert result.valid_baits == 2
            assert result.invalid_baits == 0
            assert result.expected_qafiya == "ل"
            assert result.qafiya_harakah == "مكسور"
            assert result.qafiya_type == "متواتر"
            assert result.misaligned_bait_numbers == []
            assert len(result.bait_results) == 2
            
            # Check all bait results are valid
            for bait_result in result.bait_results:
                assert isinstance(bait_result, QafiyaBaitResult)
                assert bait_result.is_valid is True
                assert bait_result.error_details is None
    
    def test_evaluate_qafiya_invalid_poem(self):
        """Test qafiya validation with an invalid poem - each bait evaluated individually"""
        # Mock LLM responses for each bait (3 baits, last one invalid)
        mock_responses = [
            '{"is_valid": true, "issue": null}',
            '{"is_valid": true, "issue": null}',
            '{"is_valid": false, "issue": "البيت الثالث لا يتبع نفس القافية"}'
        ]
        self.mock_llm.generate.side_effect = mock_responses
        
        with patch('poet.prompts.prompt_manager.PromptManager.format_prompt') as mock_format_prompt:
            mock_format_prompt.return_value = "test qafiya validation prompt"
            
            result = self.validator.evaluate_qafiya(
                self.invalid_poem, 
                expected_qafiya="ل",
                qafiya_harakah="مكسور",
                qafiya_type="متواتر",
                qafiya_type_description_and_examples="متواتر: متحرك واحد بين ساكنين"
            )
            
            # Verify LLM was called three times (once for each bait)
            assert self.mock_llm.generate.call_count == 3
            
            # Verify result
            assert isinstance(result, QafiyaValidationResult)
            assert result.overall_valid is False
            assert result.total_baits == 3
            assert result.valid_baits == 2
            assert result.invalid_baits == 1
            assert result.misaligned_bait_numbers == [3]
            
            # Check bait results
            assert len(result.bait_results) == 3
            assert result.bait_results[0].is_valid is True  # Bait 1
            assert result.bait_results[0].error_details is None
            assert result.bait_results[1].is_valid is True  # Bait 2
            assert result.bait_results[1].error_details is None
            assert result.bait_results[2].is_valid is False  # Bait 3
            assert result.bait_results[2].error_details == "البيت الثالث لا يتبع نفس القافية"
    
    def test_evaluate_qafiya_invalid_line_count(self):
        """Test qafiya validation with invalid line count (odd number of lines)"""
        invalid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا"
            ],
            llm_provider="test",
            model_name="test-model"
        )
        
        result = self.validator.evaluate_qafiya(invalid_poem, expected_qafiya="ل")
        
        # Should return invalid result for odd number of lines
        assert isinstance(result, QafiyaValidationResult)
        assert result.total_baits == 0  # No complete baits
        assert result.valid_baits == 0
        assert result.invalid_baits == 0
        assert result.validation_summary == "لا توجد أبيات للتحقق من القافية"
        assert result.overall_valid is False
    
    def test_evaluate_qafiya_empty_poem(self):
        """Test qafiya validation with empty poem"""
        empty_poem = LLMPoem(
            verses=[],
            llm_provider="test",
            model_name="test-model"
        )
        
        result = self.validator.evaluate_qafiya(empty_poem, expected_qafiya="ل")
        
        assert result.overall_valid is False
        assert result.validation_summary == "لا توجد أبيات للتحقق من القافية"
        assert result.total_baits == 0
        assert result.valid_baits == 0
        assert result.invalid_baits == 0
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response from LLM"""
        mock_response = '''
        ```json
        {
            "is_valid": false,
            "issue": "البيت لا يتبع نفس النمط"
        }
        ```
        '''
        
        result = self.validator._parse_llm_response(mock_response)
        
        assert result['is_valid'] is False
        assert result['issue'] == "البيت لا يتبع نفس النمط"
    
    def test_parse_llm_response_valid_json_no_issue(self):
        """Test parsing valid JSON response with no issue"""
        mock_response = '''
        ```json
        {
            "is_valid": true,
            "issue": null
        }
        ```
        '''
        
        result = self.validator._parse_llm_response(mock_response)
        
        assert result['is_valid'] is True
        assert result['issue'] is None
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response from LLM"""
        mock_response = "This is not JSON"
        
        with pytest.raises(QafiyaValidationError):
            self.validator._parse_llm_response(mock_response)
    
    def test_parse_llm_response_missing_required_field(self):
        """Test parsing response missing required field"""
        mock_response = '''
        ```json
        {
            "issue": "some issue"
        }
        ```
        '''
        
        with pytest.raises(QafiyaValidationError, match="Missing required fields"):
            self.validator._parse_llm_response(mock_response)
    
    def test_parse_llm_response_wrong_field_type(self):
        """Test parsing response with wrong field type"""
        mock_response = '''
        ```json
        {
            "is_valid": "not a boolean",
            "issue": null
        }
        ```
        '''
        
        with pytest.raises(QafiyaValidationError, match="'is_valid' must be a boolean"):
            self.validator._parse_llm_response(mock_response)
    
    def test_generate_validation_summary_all_valid(self):
        """Test validation summary generation for all valid baits"""
        summary = self.validator._generate_validation_summary(4, 0, expected_qafiya="ل")
        assert "جميع الأبيات (4) صحيحة قافياً" in summary
        assert "على القافية المطلوبة (ل)" in summary
    
    def test_generate_validation_summary_all_invalid(self):
        """Test validation summary generation for all invalid baits"""
        summary = self.validator._generate_validation_summary(0, 4, expected_qafiya="ع")
        assert "جميع الأبيات (4) خاطئة قافياً" in summary
        assert "على القافية المطلوبة (ع)" in summary
    
    def test_generate_validation_summary_mixed_few_invalid(self):
        """Test validation summary generation for mixed results with few invalid baits"""
        summary = self.validator._generate_validation_summary(2, 2, expected_qafiya="ق")
        assert "2 من 4 أبيات صحيحة قافياً" in summary
        assert "الأبيات الخاطئة: 1، 2" in summary
        assert "على القافية المطلوبة (ق)" in summary
    
    def test_generate_validation_summary_mixed_many_invalid(self):
        """Test validation summary generation for mixed results with many invalid baits"""
        summary = self.validator._generate_validation_summary(1, 5, expected_qafiya="ر")
        assert "1 من 6 أبيات صحيحة قافياً" in summary
        assert "عدد الأبيات الخاطئة: 5" in summary
        assert "على القافية المطلوبة (ر)" in summary
    
    def test_generate_validation_summary_no_expected_qafiya(self):
        """Test validation summary generation without expected qafiya"""
        summary = self.validator._generate_validation_summary(3, 1)
        assert "3 من 4 أبيات صحيحة قافياً" in summary
        assert "على القافية المطلوبة" not in summary
    
    def test_to_dict_serialization(self):
        """Test that QafiyaValidationResult can be serialized to dict"""
        bait_result = QafiyaBaitResult(
            bait_number=1,
            is_valid=True,
            error_details=None
        )
        
        validation_result = QafiyaValidationResult(
            overall_valid=True,
            total_baits=2,
            valid_baits=2,
            invalid_baits=0,
            bait_results=[bait_result],
            validation_summary="جميع الأبيات صحيحة",
            misaligned_bait_numbers=[],
            expected_qafiya="ل",
            qafiya_harakah="مكسور",
            qafiya_type="متواتر",
        )
        
        result_dict = validation_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['overall_valid'] is True
        assert result_dict['total_baits'] == 2
        assert result_dict['expected_qafiya'] == "ل"
        assert result_dict['qafiya_harakah'] == "مكسور"
        assert result_dict['qafiya_type'] == "متواتر"
        assert len(result_dict['bait_results']) == 1
        assert result_dict['bait_results'][0]['bait_number'] == 1
        assert result_dict['bait_results'][0]['is_valid'] is True
        assert result_dict['bait_results'][0]['error_details'] is None
    
    def test_evaluate_qafiya_llm_error_handling(self):
        """Test qafiya validation when LLM fails"""
        # Mock LLM to raise exception
        self.mock_llm.generate.side_effect = Exception("LLM API error")
        
        with pytest.raises(QafiyaValidationError, match="Qafiya validation failed"):
            self.validator.evaluate_qafiya(self.valid_poem, expected_qafiya="ل")
    
    def test_evaluate_qafiya_with_prompt_manager_injection(self):
        """Test qafiya validation with custom prompt manager"""
        mock_prompt_manager = Mock()
        mock_prompt_manager.format_prompt.return_value = "custom prompt"
        
        validator = QafiyaEvaluator(self.mock_llm, mock_prompt_manager)
        
        # Mock LLM response
        self.mock_llm.generate.return_value = '{"is_valid": true, "issue": null}'
        
        result = validator.evaluate_qafiya(self.valid_poem, expected_qafiya="ل")
        
        # Verify custom prompt manager was used
        mock_prompt_manager.format_prompt.assert_called()
        assert result.overall_valid is True
    
    def test_evaluate_qafiya_partial_specifications(self):
        """Test qafiya validation with partial specifications"""
        # Mock LLM response
        self.mock_llm.generate.return_value = '{"is_valid": true, "issue": null}'
        
        with patch('poet.prompts.prompt_manager.PromptManager.format_prompt') as mock_format_prompt:
            mock_format_prompt.return_value = "test prompt"
            
            # Test with only expected_qafiya
            result = self.validator.evaluate_qafiya(self.valid_poem, expected_qafiya="ل")
            
            assert result.expected_qafiya == "ل"
            assert result.qafiya_harakah is None
            assert result.qafiya_type is None
            
            # Test with only 
            result2 = self.validator.evaluate_qafiya(
                self.valid_poem, 
                qafiya_harakah="مكسور",
                qafiya_type="متواتر",
                qafiya_type_description_and_examples="متواتر: متحرك واحد بين ساكنين"
            )
            
            assert result2.expected_qafiya is None
    
    def test_evaluate_qafiya_single_bait_poem(self):
        """Test qafiya validation with a single bait poem"""
        single_bait_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
            ],
            llm_provider="test",
            model_name="test-model"
        )
        
        # Mock LLM response for single bait
        self.mock_llm.generate.return_value = '{"is_valid": true, "issue": null}'
        
        with patch('poet.prompts.prompt_manager.PromptManager.format_prompt') as mock_format_prompt:
            mock_format_prompt.return_value = "test prompt"
            
            result = self.validator.evaluate_qafiya(single_bait_poem, expected_qafiya="ل")
            
            # Verify LLM was called once
            assert self.mock_llm.generate.call_count == 1
            
            # Verify result
            assert result.total_baits == 1
            assert result.valid_baits == 1
            assert result.invalid_baits == 0
            assert result.overall_valid is True
            assert len(result.bait_results) == 1
            assert result.bait_results[0].bait_number == 1
            assert result.bait_results[0].is_valid is True 