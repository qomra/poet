# tests/unit/test_tashkeel_refiner.py

import pytest
import os
from unittest.mock import Mock, patch
from poet.refinement.tashkeel_refiner import TashkeelRefiner
from poet.models.poem import LLMPoem
from poet.llm.base_llm import LLMConfig


class TestTashkeelRefiner:
    """Unit tests for TashkeelRefiner"""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM provider"""
        llm = Mock()
        llm.generate.return_value = '''
        ```json
        {
            "diacritized_verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
            ]
        }
        ```
        '''
        return llm
    
    @pytest.fixture
    def tashkeel_refiner(self, mock_llm):
        """TashkeelRefiner instance with mock LLM"""
        return TashkeelRefiner(mock_llm)
    
    @pytest.fixture
    def sample_poem(self):
        """Sample LLMPoem without diacritics"""
        return LLMPoem(
            verses=[
                "قفا نبك من ذكرى حبيب ومنزل",
                "بسقط اللوى بين الدخول فحومل"
            ],
            llm_provider="mock",
            model_name="test-model"
        )
    
    def test_apply_tashkeel_success(self, tashkeel_refiner, sample_poem, mock_llm):
        """Test successful tashkeel application"""
        with patch('poet.prompts.prompt_manager.PromptManager.format_prompt') as mock_format_prompt:
            mock_format_prompt.return_value = "test tashkeel prompt"
            
            result = tashkeel_refiner.apply_tashkeel(sample_poem)
            
            # Verify LLM was called
            mock_llm.generate.assert_called_once_with("test tashkeel prompt")
            
            # Verify result
            assert isinstance(result, LLMPoem)
            assert result.verses == [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
            ]
            assert result.llm_provider == sample_poem.llm_provider
            assert result.model_name == sample_poem.model_name
            assert result.constraints == sample_poem.constraints
    
    def test_apply_tashkeel_empty_poem(self, tashkeel_refiner):
        """Test tashkeel application with empty poem"""
        empty_poem = LLMPoem(
            verses=[],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = tashkeel_refiner.apply_tashkeel(empty_poem)
        
        # Should return original poem unchanged
        assert result == empty_poem
        assert result.verses == []
    
    def test_apply_tashkeel_llm_error(self, mock_llm):
        """Test tashkeel application when LLM fails"""
        mock_llm.generate.side_effect = Exception("LLM error")
        refiner = TashkeelRefiner(mock_llm)
        
        poem = LLMPoem(
            verses=["قفا نبك من ذكرى حبيب ومنزل"],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = refiner.apply_tashkeel(poem)
        
        # Should return original poem on error
        assert result == poem
        assert result.verses == ["قفا نبك من ذكرى حبيب ومنزل"]
    
    def test_clean_diacritics(self, tashkeel_refiner):
        """Test the _clean_diacritics method"""
        # Test haraka + shaddah sequences (kasra + shaddah)
        text_with_haraka_shaddah = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        # Add a kasra + shaddah sequence manually
        text_with_haraka_shaddah = text_with_haraka_shaddah.replace("كِ", "كِّ")  # kasra + shaddah
        cleaned = tashkeel_refiner._clean_diacritics(text_with_haraka_shaddah)
        # Should remove kasra + shaddah, keeping only shaddah
        assert "كِّ" not in cleaned  # kasra + shaddah should be removed
        assert "كّ" in cleaned  # only shaddah should remain
        
        # Test shaddah + haraka sequences (shaddah + fatha)
        text_with_shaddah_haraka = "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
        # Add a shaddah + fatha sequence manually
        text_with_shaddah_haraka = text_with_shaddah_haraka.replace("لِ", "لِّ")  # shaddah + kasra
        cleaned2 = tashkeel_refiner._clean_diacritics(text_with_shaddah_haraka)
        # Should remove shaddah + kasra, keeping only shaddah
        assert "لِّ" not in cleaned2  # shaddah + kasra should be removed
        assert "لّ" in cleaned2  # only shaddah should remain
        
        # Test text without problematic sequences
        normal_text = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        cleaned3 = tashkeel_refiner._clean_diacritics(normal_text)
        assert cleaned3 == normal_text  # should be unchanged
    
    def test_parse_llm_response_valid_json(self, tashkeel_refiner):
        """Test parsing valid JSON response"""
        response = '''
        ```json
        {
            "diacritized_verses": [
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
            ]
        }
        ```
        '''
        
        result = tashkeel_refiner._parse_llm_response(response)
        
        assert result == [
            "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
            "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
        ]
    
    def test_parse_llm_response_invalid_json(self, tashkeel_refiner):
        """Test parsing invalid JSON response"""
        response = "invalid json response"

        with pytest.raises(ValueError, match="Invalid response format"):
            tashkeel_refiner._parse_llm_response(response)
    
    def test_parse_llm_response_missing_field(self, tashkeel_refiner):
        """Test parsing response with missing required field"""
        response = '''
        ```json
        {
            "other_field": ["some verses"]
        }
        ```
        '''
        
        with pytest.raises(ValueError, match="Missing 'diacritized_verses' field"):
            tashkeel_refiner._parse_llm_response(response)
    
    def test_parse_llm_response_wrong_field_type(self, tashkeel_refiner):
        """Test parsing response with wrong field type"""
        response = '''
        ```json
        {
            "diacritized_verses": "not a list"
        }
        ```
        '''
        
        with pytest.raises(ValueError, match="'diacritized_verses' must be a list"):
            tashkeel_refiner._parse_llm_response(response)
    
    def test_apply_tashkeel_preserves_poem_metadata(self, tashkeel_refiner, sample_poem, mock_llm):
        """Test that tashkeel application preserves poem metadata"""
        # Add some metadata to the original poem
        sample_poem.prosody_validation = Mock()
        sample_poem.quality = Mock()
        sample_poem.generation_timestamp = "2024-01-01T00:00:00"
        
        with patch('poet.prompts.prompt_manager.PromptManager.format_prompt') as mock_format_prompt:
            mock_format_prompt.return_value = "test prompt"
            
            result = tashkeel_refiner.apply_tashkeel(sample_poem)
            
            # Verify metadata is preserved
            assert result.quality == sample_poem.quality
            assert result.generation_timestamp == sample_poem.generation_timestamp
            assert result.constraints == sample_poem.constraints
    
    @pytest.mark.skipif(
        not os.getenv("TEST_REAL_LLMS"),
        reason="Real LLM tests require TEST_REAL_LLMS environment variable"
    )
    def test_apply_tashkeel_real_llm(self, real_llm):
        """Test tashkeel application using real LLM"""
        refiner = TashkeelRefiner(real_llm)
        
        poem = LLMPoem(
            verses=[
                "قفا نبك من ذكرى حبيب ومنزل",
                "بسقط اللوى بين الدخول فحومل"
            ],
            llm_provider="real",
            model_name="test-model"
        )
        
        result = refiner.apply_tashkeel(poem)
        
        # Verify result
        assert isinstance(result, LLMPoem)
        assert len(result.verses) == 2
        
        # Check that diacritics were applied
        for verse in result.verses:
            # Should contain diacritics
            assert any(char in verse for char in ['َ', 'ِ', 'ُ', 'ْ', 'ً', 'ٍ', 'ٌ', 'ّ'])
            # Should not be empty
            assert len(verse.strip()) > 0
    
    def test_apply_tashkeel_with_custom_prompt_manager(self):
        """Test tashkeel application with custom prompt manager"""
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"diacritized_verses": ["test"]}'
        
        mock_prompt_manager = Mock()
        mock_prompt_manager.format_prompt.return_value = "custom prompt"
        
        refiner = TashkeelRefiner(mock_llm, mock_prompt_manager)
        
        poem = LLMPoem(
            verses=["test verse"],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = refiner.apply_tashkeel(poem)
        
        # Verify custom prompt manager was used
        mock_prompt_manager.format_prompt.assert_called_once_with('tashkeel', text="test verse")
        mock_llm.generate.assert_called_once_with("custom prompt") 